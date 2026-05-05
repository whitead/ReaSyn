# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from reasyn.chem.mol import Molecule
from reasyn.sampler.runtime import PathLike, SamplingRuntime, load_sampling_runtime
from reasyn.sampler.sampler import Sampler
from reasyn.utils.sample_utils import TimeLimit


class ReaSynInference:
    """Reusable single-process inference runtime.

    This is intended for service deployments where model weights and chemistry
    indexes should be loaded once per container.
    """

    def __init__(
        self,
        model_path: str | PathLike | list[PathLike],
        *,
        device: str = "cuda",
        asset_dir: PathLike | None = None,
        fpindex_path: PathLike | None = None,
        rxn_matrix_path: PathLike | None = None,
        add_bb_path: PathLike | None = None,
        verbose: bool = True,
    ) -> None:
        self.runtime: SamplingRuntime = load_sampling_runtime(
            model_path,
            device=device,
            asset_dir=asset_dir,
            fpindex_path=fpindex_path,
            rxn_matrix_path=rxn_matrix_path,
            add_bb_path=add_bb_path,
            verbose=verbose,
        )

    def sample(
        self,
        smiles: str,
        *,
        search_width: int = 24,
        exhaustiveness: int = 64,
        max_evolve_steps: int = 8,
        max_results: int = 100,
        time_limit: int = 1000,
        exact_break: bool = True,
        num_cycles: int = 1,
        num_editflow_samples: int = 10,
        num_editflow_steps: int = 100,
        mols_to_filter: list[Molecule] | None = None,
        filter_sim: float = 0.8,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> pd.DataFrame:
        mol = Molecule(smiles)
        sampler = Sampler(
            fpindex=self.runtime.fpindex,
            rxn_matrix=self.runtime.rxn_matrix,
            mol=mol,
            model=self.runtime.models,
            factor=search_width,
            max_active_states=exhaustiveness,
            exact_break=exact_break,
            mols_to_filter=mols_to_filter,
            filter_sim=filter_sim,
        )
        timer = TimeLimit(time_limit)
        t_start = time.time()
        sampler.evolve(
            gpu_lock=None,
            time_limit=timer,
            num_cycles=num_cycles,
            max_evolve_steps=max_evolve_steps,
            num_editflow_samples=num_editflow_samples,
            num_editflow_steps=num_editflow_steps,
            progress_callback=progress_callback,
        )
        df = sampler.get_dataframe()[:max_results]
        df["time"] = time.time() - t_start
        return df
