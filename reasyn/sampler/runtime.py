# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import os
import pathlib
import pickle
from collections.abc import Iterable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.models.reasyn import ReaSyn


PathLike = str | os.PathLike[str]


@dataclasses.dataclass
class SamplingRuntime:
    models: list[ReaSyn]
    fpindex: FingerprintIndex
    rxn_matrix: ReactantReactionMatrix
    config: DictConfig


def set_process_affinity_to_all_cpus() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, range(os.cpu_count() or 1))


def resolve_path(
    path: PathLike,
    asset_dir: PathLike | None = None,
    *,
    must_exist: bool = False,
    description: str = "path",
) -> pathlib.Path:
    p = pathlib.Path(path)
    if p.is_absolute() or asset_dir is None:
        candidates = [p]
    else:
        base = pathlib.Path(asset_dir)
        candidates = [base / p, p]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    resolved = candidates[0]
    if must_exist:
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find {description}: {checked}")
    return resolved


def parse_model_paths(model_path: str | PathLike | Iterable[PathLike], asset_dir: PathLike | None = None) -> list[pathlib.Path]:
    if isinstance(model_path, (str, os.PathLike)):
        paths = str(model_path).split(",")
    else:
        paths = list(model_path)

    resolved = [
        resolve_path(pathlib.Path(path), asset_dir, must_exist=True, description="model checkpoint")
        for path in paths
    ]
    if len(resolved) != 2:
        raise ValueError("ReaSyn inference requires two checkpoints: autoregressive and editflow.")
    return resolved


def load_sampling_runtime(
    model_path: str | PathLike | Iterable[PathLike],
    *,
    device: str | torch.device = "cuda",
    asset_dir: PathLike | None = None,
    fpindex_path: PathLike | None = None,
    rxn_matrix_path: PathLike | None = None,
    add_bb_path: PathLike | None = None,
    verbose: bool = True,
) -> SamplingRuntime:
    model_paths = parse_model_paths(model_path, asset_dir)

    models: list[ReaSyn] = []
    config = None
    for checkpoint_path in model_paths:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        model = ReaSyn(config.model).to(device)
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        model.eval()
        models.append(model)

    if config is None:
        raise ValueError("No checkpoints were loaded.")

    fpindex_path = resolve_path(
        fpindex_path or config.chem.fpindex,
        asset_dir,
        must_exist=True,
        description="fingerprint index",
    )
    rxn_matrix_path = resolve_path(
        rxn_matrix_path or config.chem.rxn_matrix,
        asset_dir,
        must_exist=True,
        description="reaction matrix",
    )

    with open(fpindex_path, "rb") as f:
        fpindex: FingerprintIndex = pickle.load(f)
    with open(rxn_matrix_path, "rb") as f:
        rxn_matrix: ReactantReactionMatrix = pickle.load(f)

    if add_bb_path is not None:
        add_path = resolve_path(add_bb_path, asset_dir, must_exist=True, description="additional building block index")
        with open(add_path, "rb") as f:
            fpindex_add: FingerprintIndex = pickle.load(f)
        num_orig_bb = len(fpindex._molecules)
        fpindex._molecules = tuple(fpindex._molecules) + tuple(fpindex_add._molecules)
        fpindex._smiles = list(fpindex._smiles) + list(fpindex_add._smiles)
        fpindex._fp = np.vstack([fpindex._fp, fpindex_add._fp])
        fpindex._tree = fpindex._init_tree()
        fpindex.fp_cuda.cache_clear()
        if verbose:
            print(f"BB expanded: {num_orig_bb} -> {len(fpindex._molecules)}")

    return SamplingRuntime(models=models, fpindex=fpindex, rxn_matrix=rxn_matrix, config=config)
