from __future__ import annotations

import pathlib
import pickle

import click
import pandas as pd

from reasyn.chem.fpindex import FingerprintIndex
from reasyn.chem.matrix import create_reactant_reaction_matrix_cache
from reasyn.chem.mol import FingerprintOption, Molecule


def _select_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    try:
        index = int(column)
    except ValueError as exc:
        raise click.ClickException(f"Column {column!r} is not present in the CSV.") from exc
    if index < 0 or index >= len(df.columns):
        raise click.ClickException(f"Column index {index} is outside the CSV width of {len(df.columns)}.")
    return df.iloc[:, index]


@click.command(context_settings={"show_default": True})
@click.option("--input-csv", type=click.Path(dir_okay=False, path_type=pathlib.Path), default="mcule_unique_bb_instock_library_260130.csv")
@click.option("--smiles-column", default="5", help="Column name or zero-based index containing SMILES.")
@click.option("--has-header/--no-header", default=False)
@click.option("--output-smiles", type=click.Path(dir_okay=False, path_type=pathlib.Path), default="data/building_blocks/building_blocks_mcule.txt")
@click.option("--output-fpindex", type=click.Path(dir_okay=False, path_type=pathlib.Path), default="data/processed/mcule_2048/fpindex.pkl")
@click.option("--reaction-path", type=click.Path(dir_okay=False, path_type=pathlib.Path), default="data/rxn_templates/comprehensive.txt")
@click.option("--output-rxn-matrix", type=click.Path(dir_okay=False, path_type=pathlib.Path), default="data/processed/mcule_2048/matrix.pkl")
@click.option("--no-fpindex", is_flag=True, help="Only write the canonical SMILES text file.")
@click.option("--no-rxn-matrix", is_flag=True, help="Skip the reactant/reaction matrix.")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing outputs.")
def main(
    input_csv: pathlib.Path,
    smiles_column: str,
    has_header: bool,
    output_smiles: pathlib.Path,
    output_fpindex: pathlib.Path,
    reaction_path: pathlib.Path,
    output_rxn_matrix: pathlib.Path,
    no_fpindex: bool,
    no_rxn_matrix: bool,
    force: bool,
) -> None:
    if output_smiles.exists() and not force:
        raise click.ClickException(f"{output_smiles} already exists. Pass --force to overwrite.")
    if output_fpindex.exists() and not no_fpindex and not force:
        raise click.ClickException(f"{output_fpindex} already exists. Pass --force to overwrite.")
    if output_rxn_matrix.exists() and not no_rxn_matrix and not force:
        raise click.ClickException(f"{output_rxn_matrix} already exists. Pass --force to overwrite.")

    df = pd.read_csv(input_csv, header=0 if has_header else None)
    smiles_values = _select_column(df, smiles_column).dropna()

    molecules: list[Molecule] = []
    seen: set[str] = set()
    for raw_smiles in smiles_values:
        mol = Molecule(str(raw_smiles)).major_molecule
        if not mol.is_valid:
            continue
        canonical = mol.csmiles
        if canonical in seen:
            continue
        seen.add(canonical)
        molecules.append(Molecule(canonical))

    output_smiles.parent.mkdir(parents=True, exist_ok=True)
    with open(output_smiles, "w") as f:
        for mol in molecules:
            f.write(f"{mol.csmiles}\n")
    click.echo(f"Wrote {len(molecules)} canonical building blocks to {output_smiles}")

    if no_fpindex:
        return

    output_fpindex.parent.mkdir(parents=True, exist_ok=True)
    fpindex = FingerprintIndex(
        molecules,
        fp_option=FingerprintOption(type="morgan", morgan_radius=2, morgan_n_bits=2048),
    )
    with open(output_fpindex, "wb") as f:
        pickle.dump(fpindex, f)
    click.echo(f"Saved MCule fingerprint index to {output_fpindex}")

    if no_rxn_matrix:
        return

    if not reaction_path.exists():
        raise click.ClickException(f"Reaction template file does not exist: {reaction_path}")
    output_rxn_matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix = create_reactant_reaction_matrix_cache(
        reactant_path=output_smiles,
        reaction_path=reaction_path,
        cache_path=output_rxn_matrix,
    )
    click.echo(f"Number of reactants: {len(matrix.reactants)}")
    click.echo(f"Number of reactions: {len(matrix.reactions)}")
    click.echo(f"Saved MCule reaction matrix to {output_rxn_matrix}")
