# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib

import click

from reasyn.chem.mol import Molecule, read_mol_file
from reasyn.sampler.parallel import run_parallel_sampling
from reasyn.sampler.runtime import parse_model_paths


def _input_mols_option(path: str) -> list[Molecule]:
    return list(read_mol_file(path))


@click.command(context_settings={"show_default": True})
@click.option("--input", "-i", type=_input_mols_option, required=True)
@click.option("--output", "-o", type=click.Path(exists=False, path_type=pathlib.Path), required=True)
@click.option("--model-path", "--model_path", "-m", type=str, required=True, help="Comma-separated AR and EB checkpoint paths.")
@click.option("--asset-dir", type=click.Path(file_okay=False, path_type=pathlib.Path), default=None)
@click.option("--search-width", "--search_width", type=int, default=4)
@click.option("--exhaustiveness", type=int, default=8)
@click.option("--num-gpus", "--num_gpus", type=int, default=-1)
@click.option("--num-workers-per-gpu", "--num_workers_per_gpu", type=int, default=8)
@click.option("--task-qsize", "--task_qsize", type=int, default=0)
@click.option("--result-qsize", "--result_qsize", type=int, default=0)
@click.option("--time-limit", "--time_limit", type=int, default=10000)
@click.option("--add-bb-path", "--add_bb_path", type=str, default=None, help="Optional extra FingerprintIndex pickle.")
@click.option("--no-exact-break", "--no_exact_break", is_flag=True)
@click.option("--num-cycles", "--num_cycles", type=int, default=1)
@click.option("--num-editflow-samples", "--num_editflow_samples", type=int, default=100)
@click.option("--num-editflow-steps", "--num_editflow_steps", type=int, default=100)
@click.option("--mols-to-filter", "--mols_to_filter", type=str, default=None)
@click.option("--filter-sim", "--filter_sim", type=float, default=0.8)
def main(
    input: list[Molecule],
    output: pathlib.Path,
    model_path: str,
    asset_dir: pathlib.Path | None,
    search_width: int,
    exhaustiveness: int,
    num_gpus: int,
    num_workers_per_gpu: int,
    task_qsize: int,
    result_qsize: int,
    time_limit: int,
    add_bb_path: str | None,
    no_exact_break: bool,
    num_cycles: int,
    num_editflow_samples: int,
    num_editflow_steps: int,
    mols_to_filter: str | None,
    filter_sim: float,
) -> None:
    resolved_model_paths = parse_model_paths(model_path, asset_dir)

    run_parallel_sampling(
        input=input,
        output=output,
        model_path=resolved_model_paths,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        num_gpus=num_gpus,
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        time_limit=time_limit,
        add_bb_path=add_bb_path,
        exact_break=not no_exact_break,
        num_cycles=num_cycles,
        num_editflow_samples=num_editflow_samples,
        num_editflow_steps=num_editflow_steps,
        mols_to_filter=_input_mols_option(mols_to_filter) if mols_to_filter is not None else None,
        filter_sim=filter_sim,
        asset_dir=asset_dir,
    )
