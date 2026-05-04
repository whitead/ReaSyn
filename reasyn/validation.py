from __future__ import annotations

import dataclasses
import itertools
import re
from collections.abc import Sequence

from reasyn.chem.mol import Molecule
from reasyn.chem.reaction import Reaction


RXN_TOKEN = re.compile(r"^R(\d+)$")


@dataclasses.dataclass(frozen=True)
class ForwardStep:
    rxn_id: int
    reactants: list[str]
    product: str
    reaction_smiles: str
    reaction_smarts: str


@dataclasses.dataclass(frozen=True)
class ForwardValidation:
    valid: bool
    expected_product: str
    final_candidate_count: int
    steps: list[ForwardStep]
    error: str | None = None


@dataclasses.dataclass
class _Node:
    smiles: str
    rxn_id: int | None = None
    rxn_smarts: str | None = None
    reactants: tuple["_Node", ...] = ()


def _canon(smiles: str) -> str:
    return Molecule(smiles).csmiles


def _run_reaction(rxn: Reaction, ordered_reactants: tuple[_Node, ...]) -> dict[str, _Node]:
    products = rxn([Molecule(reactant.smiles) for reactant in ordered_reactants])
    nodes = {}
    for product in products:
        product_smiles = product.csmiles
        nodes.setdefault(product_smiles, _Node(product_smiles, reactants=ordered_reactants))
    return nodes


def _collect_steps(node: _Node) -> list[ForwardStep]:
    steps: list[ForwardStep] = []
    seen = set()

    def visit(current: _Node) -> None:
        for reactant in current.reactants:
            visit(reactant)
        if current.rxn_id is None or current.rxn_smarts is None:
            return

        reactants = [reactant.smiles for reactant in current.reactants]
        key = (current.rxn_id, tuple(reactants), current.smiles)
        if key in seen:
            return
        seen.add(key)
        steps.append(
            ForwardStep(
                rxn_id=current.rxn_id,
                reactants=reactants,
                product=current.smiles,
                reaction_smiles=f"{'.'.join(reactants)}>>{current.smiles}",
                reaction_smarts=current.rxn_smarts,
            )
        )

    visit(node)
    return steps


def validate_route_forward(
    synthesis: str,
    expected_product: str,
    reactions: Sequence[Reaction],
) -> ForwardValidation:
    """Forward-execute a ReaSyn stack route and verify the expected product."""
    try:
        expected_canon = _canon(expected_product)
        stack: list[dict[str, _Node]] = []

        for token in synthesis.split(";"):
            token = token.strip()
            if not token:
                continue

            rxn_match = RXN_TOKEN.match(token)
            if rxn_match is None:
                smiles = _canon(token)
                stack.append({smiles: _Node(smiles)})
                continue

            rxn_id = int(rxn_match.group(1))
            rxn = reactions[rxn_id]
            num_reactants = rxn.num_reactants
            if len(stack) < num_reactants:
                return ForwardValidation(
                    valid=False,
                    expected_product=expected_canon,
                    final_candidate_count=0,
                    steps=[],
                    error=f"R{rxn_id} requires {num_reactants} reactants but stack has {len(stack)}.",
                )

            reactant_sets = [stack[-1 - i] for i in range(num_reactants)]
            if num_reactants == 1:
                ordered_groups = [(node,) for node in reactant_sets[0].values()]
            else:
                ordered_groups = []
                for combo in itertools.product(*[list(reactants.values()) for reactants in reactant_sets]):
                    ordered_groups.extend(itertools.permutations(combo, num_reactants))

            products_by_smiles: dict[str, _Node] = {}
            for ordered_reactants in ordered_groups:
                for product_smiles in _run_reaction(rxn, ordered_reactants):
                    products_by_smiles.setdefault(
                        product_smiles,
                        _Node(
                            product_smiles,
                            rxn_id=rxn_id,
                            rxn_smarts=rxn.smarts,
                            reactants=ordered_reactants,
                        ),
                    )

            if not products_by_smiles:
                return ForwardValidation(
                    valid=False,
                    expected_product=expected_canon,
                    final_candidate_count=0,
                    steps=[],
                    error=f"R{rxn_id} produced no RDKit products.",
                )

            del stack[-num_reactants:]
            stack.append(products_by_smiles)

        final_products = stack[-1] if stack else {}
        final_node = final_products.get(expected_canon)
        return ForwardValidation(
            valid=final_node is not None,
            expected_product=expected_canon,
            final_candidate_count=len(final_products),
            steps=_collect_steps(final_node) if final_node is not None else [],
            error=None if final_node is not None else "Expected product was not among final RDKit products.",
        )
    except Exception as exc:
        return ForwardValidation(
            valid=False,
            expected_product=expected_product,
            final_candidate_count=0,
            steps=[],
            error=str(exc),
        )
