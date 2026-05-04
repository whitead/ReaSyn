# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib

import click

from reasyn.inference import ReaSynInference


@click.command(context_settings={"show_default": True})
@click.option("--smiles", "-s", required=True, help="Input molecule SMILES.")
@click.option("--model-path", "--model_path", "-m", required=True, help="Comma-separated AR and EB checkpoint paths.")
@click.option("--output", "-o", type=click.Path(exists=False, path_type=pathlib.Path), default=None)
@click.option("--asset-dir", type=click.Path(file_okay=False, path_type=pathlib.Path), default=None)
@click.option("--add-bb-path", "--add_bb_path", type=str, default=None, help="Optional extra FingerprintIndex pickle.")
@click.option("--device", default="cuda")
@click.option("--search-width", "--search_width", type=int, default=24)
@click.option("--exhaustiveness", type=int, default=64)
@click.option("--max-evolve-steps", "--max_evolve_steps", type=int, default=8)
@click.option("--max-results", "--max_results", type=int, default=100)
@click.option("--time-limit", "--time_limit", type=int, default=1000)
@click.option("--no-exact-break", "--no_exact_break", is_flag=True)
@click.option("--num-cycles", "--num_cycles", type=int, default=1)
@click.option("--num-editflow-samples", "--num_editflow_samples", type=int, default=10)
@click.option("--num-editflow-steps", "--num_editflow_steps", type=int, default=100)
def main(
    smiles: str,
    model_path: str,
    output: pathlib.Path | None,
    asset_dir: pathlib.Path | None,
    add_bb_path: str | None,
    device: str,
    search_width: int,
    exhaustiveness: int,
    max_evolve_steps: int,
    max_results: int,
    time_limit: int,
    no_exact_break: bool,
    num_cycles: int,
    num_editflow_samples: int,
    num_editflow_steps: int,
) -> None:
    engine = ReaSynInference(
        model_path,
        device=device,
        asset_dir=asset_dir,
        add_bb_path=add_bb_path,
    )
    df = engine.sample(
        smiles,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        max_evolve_steps=max_evolve_steps,
        max_results=max_results,
        time_limit=time_limit,
        exact_break=not no_exact_break,
        num_cycles=num_cycles,
        num_editflow_samples=num_editflow_samples,
        num_editflow_steps=num_editflow_steps,
    )
    if output is None:
        click.echo(df.to_string(index=False))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, float_format="%.3f", index=False)
