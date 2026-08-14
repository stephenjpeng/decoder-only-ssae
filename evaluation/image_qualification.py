"""Image qualification prompt planning and manifest generation for native model gates.

Loads a declarative YAML spec, validates property and context uniqueness, builds
deterministic row plans for audit and bakeoff designs, and generates stable row IDs
for traceability and blinded scoring.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


EXPECTED_PROPERTY_COUNT = 26
EXPECTED_AUDIT_CONTEXT_COUNT = 2
EXPECTED_AUDIT_SEED_COUNT = 2
EXPECTED_AUDIT_ROW_COUNT = 104
EXPECTED_BAKEOFF_PROPERTY_COUNT = 12
EXPECTED_BAKEOFF_CONTEXT_COUNT = 4
EXPECTED_BAKEOFF_SEED_COUNT = 4
EXPECTED_BAKEOFF_MODEL_COUNT = 3
EXPECTED_BAKEOFF_ROW_COUNT = 576


@dataclass
class Property:
    """A single property from the qualification spec."""

    id: str
    category: str
    original_phrase: str
    rendering_phrase: str
    critical: bool
    high_risk: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Property:
        return cls(
            id=data["id"],
            category=data["category"],
            original_phrase=data["original_phrase"],
            rendering_phrase=data["rendering_phrase"],
            critical=data["critical"],
            high_risk=data["high_risk"],
        )


@dataclass
class Context:
    """A prompt context template."""

    id: str
    description: str
    template: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Context:
        return cls(
            id=data["id"],
            description=data["description"],
            template=data["template"],
        )

    def render(self, property_phrase: str) -> str:
        """Render a property phrase into this context template."""
        return self.template.format(property=property_phrase)


@dataclass
class AuditDesign:
    """Prompt audit design: all properties x contexts x seeds."""

    seeds: list[int]
    context_ids: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditDesign:
        return cls(
            seeds=data["seeds"],
            context_ids=data["context_ids"],
        )


@dataclass
class BakeoffDesign:
    """Bakeoff design: frozen high-risk properties x contexts x seeds x models."""

    property_ids: list[str]
    seeds: list[int]
    context_ids: list[str]
    model_backbones: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BakeoffDesign:
        return cls(
            property_ids=data["property_ids"],
            seeds=data["seeds"],
            context_ids=data["context_ids"],
            model_backbones=data["model_backbones"],
        )


@dataclass
class PromptSpec:
    """Complete prompt specification for image qualification."""

    properties: list[Property]
    contexts: list[Context]
    audit_design: AuditDesign
    bakeoff_design: BakeoffDesign

    @classmethod
    def load(cls, path: str | Path) -> PromptSpec:
        """Load and validate a prompt spec from YAML."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        properties = [Property.from_dict(p) for p in data["properties"]]
        contexts = [Context.from_dict(c) for c in data["contexts"]]
        audit_design = AuditDesign.from_dict(data["audit_design"])
        bakeoff_design = BakeoffDesign.from_dict(data["bakeoff_design"])

        spec = cls(
            properties=properties,
            contexts=contexts,
            audit_design=audit_design,
            bakeoff_design=bakeoff_design,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        """Validate uniqueness and cross-references."""
        # freeze the property inventory so every source property remains traceable
        property_ids = [p.id for p in self.properties]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError(f"duplicate property IDs: {property_ids}")
        if len(property_ids) != EXPECTED_PROPERTY_COUNT:
            raise ValueError(
                f"prompt spec must contain exactly {EXPECTED_PROPERTY_COUNT} properties, "
                f"got {len(property_ids)}"
            )

        # duplicate factor values would create repeated experimental rows
        context_ids = [c.id for c in self.contexts]
        if len(context_ids) != len(set(context_ids)):
            raise ValueError(f"duplicate context IDs: {context_ids}")
        self._validate_unique_factors()
        self._validate_design_dimensions()

        # check audit context references
        for ctx_id in self.audit_design.context_ids:
            if ctx_id not in context_ids:
                raise ValueError(
                    f"audit_design references unknown context {ctx_id!r}; "
                    f"known: {context_ids}"
                )

        # check bakeoff property references
        for prop_id in self.bakeoff_design.property_ids:
            if prop_id not in property_ids:
                raise ValueError(
                    f"bakeoff_design references unknown property {prop_id!r}; "
                    f"known: {property_ids}"
                )

        # check bakeoff context references
        for ctx_id in self.bakeoff_design.context_ids:
            if ctx_id not in context_ids:
                raise ValueError(
                    f"bakeoff_design references unknown context {ctx_id!r}; "
                    f"known: {context_ids}"
                )

        # verify all bakeoff properties are marked high_risk
        prop_map = {p.id: p for p in self.properties}
        for prop_id in self.bakeoff_design.property_ids:
            if not prop_map[prop_id].high_risk:
                raise ValueError(
                    f"bakeoff property {prop_id!r} must be marked high_risk=true"
                )

    def _validate_unique_factors(self) -> None:
        """Reject repeated design factors before they create duplicate rows."""
        factors = {
            "audit seeds": self.audit_design.seeds,
            "audit contexts": self.audit_design.context_ids,
            "bakeoff properties": self.bakeoff_design.property_ids,
            "bakeoff seeds": self.bakeoff_design.seeds,
            "bakeoff contexts": self.bakeoff_design.context_ids,
            "bakeoff models": self.bakeoff_design.model_backbones,
        }
        for name, values in factors.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique, got {values}")

    def _validate_design_dimensions(self) -> None:
        """Enforce the frozen qualification design dimensions."""
        expected_dimensions = {
            "audit contexts": (
                len(self.audit_design.context_ids),
                EXPECTED_AUDIT_CONTEXT_COUNT,
            ),
            "audit seeds": (len(self.audit_design.seeds), EXPECTED_AUDIT_SEED_COUNT),
            "bakeoff properties": (
                len(self.bakeoff_design.property_ids),
                EXPECTED_BAKEOFF_PROPERTY_COUNT,
            ),
            "bakeoff contexts": (
                len(self.bakeoff_design.context_ids),
                EXPECTED_BAKEOFF_CONTEXT_COUNT,
            ),
            "bakeoff seeds": (
                len(self.bakeoff_design.seeds),
                EXPECTED_BAKEOFF_SEED_COUNT,
            ),
            "bakeoff models": (
                len(self.bakeoff_design.model_backbones),
                EXPECTED_BAKEOFF_MODEL_COUNT,
            ),
        }
        for name, (actual, expected) in expected_dimensions.items():
            if actual != expected:
                raise ValueError(
                    f"{name} must contain exactly {expected} values, got {actual}"
                )

    def fingerprint(self) -> str:
        """Compute a stable fingerprint of the prompt spec content.

        The fingerprint is a short SHA256 hash of the canonical JSON serialization
        of properties, contexts, and designs. Changes to property phrases, context
        templates, or experimental designs yield a different fingerprint.
        """
        canonical = {
            "properties": [
                {
                    "id": p.id,
                    "category": p.category,
                    "original_phrase": p.original_phrase,
                    "rendering_phrase": p.rendering_phrase,
                    "critical": p.critical,
                    "high_risk": p.high_risk,
                }
                for p in self.properties
            ],
            "contexts": [
                {"id": c.id, "description": c.description, "template": c.template}
                for c in self.contexts
            ],
            "audit_design": {
                "seeds": self.audit_design.seeds,
                "context_ids": self.audit_design.context_ids,
            },
            "bakeoff_design": {
                "property_ids": self.bakeoff_design.property_ids,
                "seeds": self.bakeoff_design.seeds,
                "context_ids": self.bakeoff_design.context_ids,
                "model_backbones": self.bakeoff_design.model_backbones,
            },
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()
        return digest[:12]

    def get_property(self, prop_id: str) -> Property:
        """Retrieve a property by ID."""
        for p in self.properties:
            if p.id == prop_id:
                return p
        raise KeyError(f"unknown property {prop_id!r}")

    def get_context(self, ctx_id: str) -> Context:
        """Retrieve a context by ID."""
        for c in self.contexts:
            if c.id == ctx_id:
                return c
        raise KeyError(f"unknown context {ctx_id!r}")


@dataclass
class RenderRow:
    """A single row in the render plan."""

    row_id: str
    stage: str
    model_backbone: str
    property_id: str
    context_id: str
    seed: int
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSONL output."""
        return {
            "row_id": self.row_id,
            "stage": self.stage,
            "model_backbone": self.model_backbone,
            "property_id": self.property_id,
            "context_id": self.context_id,
            "seed": self.seed,
            "prompt": self.prompt,
        }


def build_audit_plan(spec: PromptSpec) -> list[RenderRow]:
    """Build deterministic audit plan: all 26 properties x contexts x seeds.

    Row IDs are stable and encode property, context, and seed for traceability.
    """
    rows: list[RenderRow] = []
    for prop in spec.properties:
        for ctx_id in spec.audit_design.context_ids:
            ctx = spec.get_context(ctx_id)
            for seed in spec.audit_design.seeds:
                row_id = f"audit_{prop.id}_{ctx_id}_s{seed}"
                prompt = ctx.render(prop.rendering_phrase)
                rows.append(
                    RenderRow(
                        row_id=row_id,
                        stage="native",
                        model_backbone="sd35_large_turbo",  # audit uses default model
                        property_id=prop.id,
                        context_id=ctx_id,
                        seed=seed,
                        prompt=prompt,
                    )
                )
    return rows


def build_bakeoff_plan(spec: PromptSpec) -> list[RenderRow]:
    """Build deterministic bakeoff plan: 12 high-risk properties x contexts x seeds x models.

    Row IDs are stable and encode property, context, seed, and model for traceability.
    """
    rows: list[RenderRow] = []
    for prop_id in spec.bakeoff_design.property_ids:
        prop = spec.get_property(prop_id)
        for ctx_id in spec.bakeoff_design.context_ids:
            ctx = spec.get_context(ctx_id)
            for seed in spec.bakeoff_design.seeds:
                for model in spec.bakeoff_design.model_backbones:
                    row_id = f"bakeoff_{prop.id}_{ctx_id}_s{seed}_{model}"
                    prompt = ctx.render(prop.rendering_phrase)
                    rows.append(
                        RenderRow(
                            row_id=row_id,
                            stage="native",
                            model_backbone=model,
                            property_id=prop.id,
                            context_id=ctx_id,
                            seed=seed,
                            prompt=prompt,
                        )
                    )
    return rows


def validate_coverage(
    audit_rows: list[RenderRow], bakeoff_rows: list[RenderRow], spec: PromptSpec
) -> None:
    """Validate the frozen plan dimensions and row ID uniqueness."""
    spec._validate_design_dimensions()
    if len(spec.properties) != EXPECTED_PROPERTY_COUNT:
        raise ValueError(
            f"prompt spec must contain exactly {EXPECTED_PROPERTY_COUNT} properties, "
            f"got {len(spec.properties)}"
        )
    if len(audit_rows) != EXPECTED_AUDIT_ROW_COUNT:
        raise ValueError(
            f"audit plan has {len(audit_rows)} rows, expected {EXPECTED_AUDIT_ROW_COUNT}"
        )
    if len(bakeoff_rows) != EXPECTED_BAKEOFF_ROW_COUNT:
        raise ValueError(
            f"bakeoff plan has {len(bakeoff_rows)} rows, "
            f"expected {EXPECTED_BAKEOFF_ROW_COUNT}"
        )

    # check row ID uniqueness across both plans
    all_row_ids = [r.row_id for r in audit_rows] + [r.row_id for r in bakeoff_rows]
    if len(all_row_ids) != len(set(all_row_ids)):
        raise ValueError(
            f"duplicate row IDs detected in combined audit + bakeoff plans"
        )
