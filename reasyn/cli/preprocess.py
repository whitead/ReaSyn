# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib

import click
from omegaconf import DictConfig, OmegaConf

from reasyn.chem.fpindex import create_fingerprint_index_cache
from reasyn.chem.matrix import create_reactant_reaction_matrix_cache
from reasyn.chem.mol import FingerprintOption


@click.command(context_settings={"show_default": True})
@click.option("--model-config", type=OmegaConf.load, required=True)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing caches without prompting.")
def main(model_config: DictConfig, force: bool) -> None:
    building_block_path = pathlib.Path(model_config.chem.building_block_path)
    reaction_path = pathlib.Path(model_config.chem.reaction_path)
    out_fp = pathlib.Path(model_config.chem.fpindex)
    out_rxn = pathlib.Path(model_config.chem.rxn_matrix) if "rxn_matrix" in model_config.chem else None

    if force or not out_fp.exists() or click.confirm(f"{out_fp} already exists. Overwrite?"):
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        fp_option = FingerprintOption(**model_config.chem.fp_option)
        fpindex = create_fingerprint_index_cache(
            molecule_path=building_block_path,
            cache_path=out_fp,
            fp_option=fp_option,
        )
        click.echo(f"Number of molecules: {len(fpindex.molecules)}")
        click.echo(f"Saved index to {out_fp}")

    if out_rxn is not None and (force or not out_rxn.exists() or click.confirm(f"{out_rxn} already exists. Overwrite?")):
        out_rxn.parent.mkdir(parents=True, exist_ok=True)
        matrix = create_reactant_reaction_matrix_cache(
            reactant_path=building_block_path,
            reaction_path=reaction_path,
            cache_path=out_rxn,
        )
        click.echo(f"Number of reactants: {len(matrix.reactants)}")
        click.echo(f"Number of reactions: {len(matrix.reactions)}")
        click.echo(f"Saved matrix to {out_rxn}")
