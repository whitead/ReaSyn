from __future__ import annotations

import csv
import dataclasses
import pathlib
from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class ReactionAnnotation:
    template_id: int
    template_smarts: str
    reaction_class_id: str
    reaction_name: str
    annotation_method: str
    annotation_confidence: float | None
    example_reactants: list[str]
    example_product: str

    def model_dump(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def load_reaction_annotations(path: str | pathlib.Path) -> Mapping[int, ReactionAnnotation]:
    path = pathlib.Path(path)
    annotations: dict[int, ReactionAnnotation] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "template_id",
            "template_smarts",
            "reaction_class_id",
            "reaction_name",
            "annotation_method",
            "annotation_confidence",
            "example_reactants",
            "example_product",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            missing_s = ", ".join(sorted(missing))
            raise ValueError(f"Reaction annotation file {path} is missing columns: {missing_s}")

        for row in reader:
            template_id = int(row["template_id"])
            if template_id in annotations:
                raise ValueError(f"Duplicate reaction annotation template_id: {template_id}")
            confidence_s = row["annotation_confidence"].strip()
            annotations[template_id] = ReactionAnnotation(
                template_id=template_id,
                template_smarts=row["template_smarts"].strip(),
                reaction_class_id=row["reaction_class_id"].strip(),
                reaction_name=row["reaction_name"].strip(),
                annotation_method=row["annotation_method"].strip(),
                annotation_confidence=float(confidence_s) if confidence_s else None,
                example_reactants=[
                    reactant.strip()
                    for reactant in row["example_reactants"].split("|")
                    if reactant.strip()
                ],
                example_product=row["example_product"].strip(),
            )
    return annotations
