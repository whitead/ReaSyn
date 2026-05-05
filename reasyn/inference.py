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


def _exact_rows(df: pd.DataFrame, target_csmiles: str) -> pd.DataFrame:
    if df.empty or "smiles" not in df:
        return df.head(0)

    matches = []
    for smiles in df["smiles"]:
        try:
            matches.append(Molecule(str(smiles)).csmiles == target_csmiles)
        except Exception:
            matches.append(False)
    return df.loc[matches]


def _dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "synthesis" not in df:
        return df
    return df.drop_duplicates(subset=["synthesis"], keep="first")


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
        exact_target_only: bool = False,
        min_exact_results: int = 1,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> pd.DataFrame:
        mol = Molecule(smiles)

        def new_sampler() -> Sampler:
            return Sampler(
                fpindex=self.runtime.fpindex,
                rxn_matrix=self.runtime.rxn_matrix,
                mol=mol,
                model=self.runtime.models,
                factor=search_width,
                max_active_states=exhaustiveness,
                exact_break=exact_break and not exact_target_only,
                mols_to_filter=mols_to_filter,
                filter_sim=filter_sim,
            )

        sampler = new_sampler()
        timer = TimeLimit(time_limit)
        t_start = time.time()

        round_index = 0
        df = pd.DataFrame()
        target_csmiles = mol.csmiles
        exact_df = pd.DataFrame()
        while True:
            round_index += 1
            if progress_callback is not None and exact_target_only:
                progress_callback(
                    {
                        "event": "exact_search_round_started",
                        "round": round_index,
                        "requested_exact_routes": min_exact_results,
                    }
                )

            sampler.evolve(
                gpu_lock=None,
                time_limit=timer,
                num_cycles=num_cycles,
                max_evolve_steps=max_evolve_steps,
                num_editflow_samples=num_editflow_samples,
                num_editflow_steps=num_editflow_steps,
                progress_callback=progress_callback,
            )
            df = sampler.get_dataframe()

            if not exact_target_only:
                df = df[:max_results]
                break

            exact_df = _dedupe_rows(pd.concat([exact_df, _exact_rows(df, target_csmiles)], ignore_index=True))
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "exact_routes_status",
                        "round": round_index,
                        "exact_routes": len(exact_df),
                        "requested_exact_routes": min_exact_results,
                    }
                )

            if len(exact_df) >= min_exact_results:
                df = exact_df[:max_results]
                break
            if timer.exceeded():
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "exact_search_timeout",
                            "round": round_index,
                            "exact_routes": len(exact_df),
                            "requested_exact_routes": min_exact_results,
                        }
                    )
                df = exact_df[:max_results]
                break

            active_states = getattr(sampler, "_active", [])
            finished_states = getattr(sampler, "_finished", [])
            has_reaction_finished = any(state.stack.count_reactions() for state in finished_states)
            if not active_states and not has_reaction_finished:
                sampler = new_sampler()
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "exact_search_restarted",
                            "round": round_index,
                            "exact_routes": len(exact_df),
                            "requested_exact_routes": min_exact_results,
                        }
                    )

        df["time"] = time.time() - t_start
        return df
