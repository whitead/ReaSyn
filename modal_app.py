from __future__ import annotations

import hmac
import json
import os
import pathlib
import queue
import shutil
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

import modal
from fastapi import Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field


APP_NAME = "reasyn"
VOLUME_NAME = "reasyn-assets"
AUTH_SECRET_NAME = "reasyn-web-auth"
AUTH_ENV_VAR = "REASYN_AUTH_TOKEN"

ASSET_DIR = pathlib.Path("/vol/reasyn")
AR_CKPT = ASSET_DIR / "data/trained_model/nv-reasyn-ar-166m-v2.ckpt"
EB_CKPT = ASSET_DIR / "data/trained_model/nv-reasyn-eb-174m-v2.ckpt"
MCULE_FPINDEX = ASSET_DIR / "data/processed/mcule_2048/fpindex.pkl"
MCULE_MATRIX = ASSET_DIR / "data/processed/mcule_2048/matrix.pkl"

BATCH_MAX_REQUESTS = 4
BATCH_WAIT_MS = 750
CACHE_TTL_SECONDS = 5 * 60
CACHE_MAX_ITEMS = 256

Effort = Literal["low", "medium", "high"]

EFFORT_PRESETS: dict[Effort, dict[str, int]] = {
    "low": {
        "search_width": 2,
        "exhaustiveness": 4,
        "num_cycles": 1,
        "num_editflow_samples": 5,
        "num_editflow_steps": 25,
        "time_limit": 120,
    },
    "medium": {
        "search_width": 4,
        "exhaustiveness": 8,
        "num_cycles": 2,
        "num_editflow_samples": 10,
        "num_editflow_steps": 50,
        "time_limit": 300,
    },
    "high": {
        "search_width": 8,
        "exhaustiveness": 32,
        "num_cycles": 4,
        "num_editflow_samples": 25,
        "num_editflow_steps": 100,
        "time_limit": 600,
    },
}


class SearchOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_width: int | None = Field(default=None, ge=1, le=64)
    exhaustiveness: int | None = Field(default=None, ge=1, le=256)
    num_cycles: int | None = Field(default=None, ge=1, le=12)
    num_editflow_samples: int | None = Field(default=None, ge=1, le=100)
    num_editflow_steps: int | None = Field(default=None, ge=1, le=200)
    time_limit: int | None = Field(default=None, ge=0, le=1800)


class RoutesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smiles: list[str] = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    effort: Effort = "medium"
    verbose: bool = False
    overrides: SearchOverrides | None = None


class MoleculeRoutes(BaseModel):
    input: str
    routes: list[Any]
    error: str | None = None


class RoutesResponse(BaseModel):
    results: list[MoleculeRoutes]


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

worker_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgomp1")
    .uv_sync()
    .uv_pip_install("fastapi[standard]>=0.115,<1")
    .add_local_python_source("reasyn")
)

web_image = modal.Image.debian_slim(python_version="3.10").uv_pip_install("fastapi[standard]>=0.115,<1")
asset_image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "fastapi[standard]>=0.115,<1",
    "huggingface-hub[hf_xet]>=0.36,<1",
)


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def _require_auth(authorization: str | None) -> None:
    expected = os.environ.get(AUTH_ENV_VAR)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Modal secret {AUTH_SECRET_NAME!r} must define {AUTH_ENV_VAR}.",
        )

    token = _extract_bearer_token(authorization)
    if token is None or not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _compact_routes(routes: list[dict[str, Any]]) -> list[list[str]]:
    return [route["reaction_smiles"] for route in routes]


def _sse(event: dict[str, Any]) -> str:
    event_name = str(event.get("event", "message"))
    data = json.dumps(event, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data}\n\n"


def _effective_search(request: RoutesRequest) -> dict[str, int]:
    preset = EFFORT_PRESETS[request.effort].copy()
    if request.overrides is None:
        return preset

    for key, value in request.overrides.model_dump(exclude_none=True).items():
        preset[key] = value
    return preset


def _route_cache_key(smiles: str, request: RoutesRequest) -> tuple[Any, ...]:
    effective_search = _effective_search(request)
    return (
        smiles,
        request.k,
        request.effort,
        tuple(sorted(effective_search.items())),
    )


def _synthesis_tokens(synthesis: str) -> list[str]:
    return [token.strip() for token in synthesis.split(";") if token.strip()]


def _building_blocks(synthesis: str) -> list[str]:
    return [token for token in _synthesis_tokens(synthesis) if not token.startswith("R")]


def _multistep_reaction_smiles(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""

    chain = steps[0]["reaction_smiles"]
    previous_product = steps[0]["product"]
    for step in steps[1:]:
        extra_reactants = [reactant for reactant in step["reactants"] if reactant != previous_product]
        if extra_reactants:
            chain += "." + ".".join(extra_reactants)
        chain += ">>" + step["product"]
        previous_product = step["product"]
    return chain


def _reaction_details(reaction: Any) -> dict[str, Any]:
    return {
        "smarts": reaction.smarts,
        "num_reactants": reaction.num_reactants,
        "num_agents": reaction.num_agents,
        "num_products": reaction.num_products,
        "reactant_templates": [template.smarts for template in reaction.reactant_templates],
        "agent_templates": [template.smarts for template in reaction.agent_templates],
        "product_templates": [template.smarts for template in reaction.product_templates],
        "metadata_note": (
            "The bundled ReaSyn template set contains SMARTS templates, not named "
            "reagents, catalysts, solvents, or reaction conditions."
        ),
    }


class _ReaSynRuntimeMixin:
    def _load_engine(self) -> None:
        from reasyn.inference import ReaSynInference

        missing = [path for path in (AR_CKPT, EB_CKPT, MCULE_FPINDEX, MCULE_MATRIX) if not path.exists()]
        if missing:
            missing_s = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing ReaSyn Modal assets: {missing_s}")

        self.engine = ReaSynInference(
            [AR_CKPT, EB_CKPT],
            asset_dir=ASSET_DIR,
            fpindex_path=MCULE_FPINDEX,
            rxn_matrix_path=MCULE_MATRIX,
            device="cuda",
        )
        self._route_cache: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}

    def _stream_payload(self, request: dict[str, Any]):
        routes_request = RoutesRequest.model_validate(request)
        request_started = time.time()
        yield {
            "event": "request_started",
            "input_count": len(routes_request.smiles),
            "k": routes_request.k,
            "effort": routes_request.effort,
            "search": _effective_search(routes_request),
        }

        results: list[dict[str, Any]] = []
        for index, smiles in enumerate(routes_request.smiles, start=1):
            yield {
                "event": "molecule_started",
                "input_index": index,
                "input_count": len(routes_request.smiles),
                "smiles": smiles,
                "elapsed_seconds": round(time.time() - request_started, 3),
            }
            for event in self._stream_one(smiles, routes_request, index, len(routes_request.smiles)):
                if event.get("event") in {"molecule_completed", "molecule_failed"} and isinstance(event.get("result"), dict):
                    results.append(event["result"])
                yield event

        yield {
            "event": "request_completed",
            "elapsed_seconds": round(time.time() - request_started, 3),
            "result": RoutesResponse.model_validate({"results": results}).model_dump(),
        }

    def _sample_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        routes_request = RoutesRequest.model_validate(request)
        results = [
            self._sample_one(smiles, routes_request)
            for smiles in routes_request.smiles
        ]
        return RoutesResponse(results=results).model_dump()

    def _sample_one(self, smiles: str, request: RoutesRequest) -> MoleculeRoutes:
        try:
            routes, cache_hit = self._get_or_sample_routes(smiles, request)
        except Exception as exc:
            return MoleculeRoutes(input=smiles, routes=[], error=str(exc))

        if not request.verbose:
            return MoleculeRoutes(input=smiles, routes=_compact_routes(routes))

        for route in routes:
            route["search"]["cache_hit"] = cache_hit
        return MoleculeRoutes(input=smiles, routes=routes)

    def _stream_one(
        self,
        smiles: str,
        request: RoutesRequest,
        input_index: int,
        input_count: int,
    ):
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        started = time.time()

        def emit(event: dict[str, Any]) -> None:
            events.put(
                {
                    "input_index": input_index,
                    "input_count": input_count,
                    "smiles": smiles,
                    "elapsed_seconds": round(time.time() - started, 3),
                    **event,
                }
            )

        def run() -> None:
            try:
                routes, cache_hit = self._get_or_sample_routes(smiles, request, progress_callback=emit)
                if not request.verbose:
                    route_payload: list[Any] = _compact_routes(routes)
                else:
                    for route in routes:
                        route["search"]["cache_hit"] = cache_hit
                    route_payload = routes
                emit(
                    {
                        "event": "molecule_completed",
                        "cache_hit": cache_hit,
                        "result": MoleculeRoutes(input=smiles, routes=route_payload).model_dump(),
                    }
                )
            except Exception as exc:
                emit(
                    {
                        "event": "molecule_failed",
                        "error": str(exc),
                        "result": MoleculeRoutes(input=smiles, routes=[], error=str(exc)).model_dump(),
                    }
                )
            finally:
                events.put(None)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        while True:
            event = events.get()
            if event is None:
                break
            yield event
        thread.join()

    def _get_or_sample_routes(
        self,
        smiles: str,
        request: RoutesRequest,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        cache_key = _route_cache_key(smiles, request)
        now = time.time()
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            expires_at, routes = cached
            if expires_at > now:
                if progress_callback is not None:
                    progress_callback({"event": "cache_hit"})
                return [self._copy_route(route) for route in routes], True
            del self._route_cache[cache_key]

        if progress_callback is not None:
            progress_callback({"event": "cache_miss"})
        routes = self._sample_routes(smiles, request, progress_callback=progress_callback)
        self._route_cache[cache_key] = (now + CACHE_TTL_SECONDS, [self._copy_route(route) for route in routes])
        if len(self._route_cache) > CACHE_MAX_ITEMS:
            self._evict_expired_or_oldest_cache_entry()
        return routes, False

    def _sample_routes(
        self,
        smiles: str,
        request: RoutesRequest,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        from reasyn.validation import validate_route_forward

        preset = _effective_search(request)
        df = self.engine.sample(
            smiles,
            search_width=preset["search_width"],
            exhaustiveness=preset["exhaustiveness"],
            max_results=request.k,
            time_limit=preset["time_limit"],
            num_cycles=preset["num_cycles"],
            num_editflow_samples=preset["num_editflow_samples"],
            num_editflow_steps=preset["num_editflow_steps"],
            progress_callback=progress_callback,
        )

        routes = []
        for rank, row in enumerate(df.head(request.k).to_dict(orient="records"), start=1):
            validation = validate_route_forward(
                synthesis=str(row["synthesis"]),
                expected_product=str(row["smiles"]),
                reactions=self.engine.runtime.rxn_matrix.reactions,
            )
            reaction_smiles = [step.reaction_smiles for step in validation.steps]
            forward_steps = [
                {
                    "rxn_id": step.rxn_id,
                    "reactants": step.reactants,
                    "product": step.product,
                    "reaction_smiles": step.reaction_smiles,
                    "reaction_smarts": step.reaction_smarts,
                    "reaction": _reaction_details(self.engine.runtime.rxn_matrix.reactions[step.rxn_id]),
                }
                for step in validation.steps
            ]
            synthesis = str(row["synthesis"])
            routes.append(
                {
                    "rank": rank,
                    "reaction_smiles": reaction_smiles,
                    "multistep_reaction_smiles": _multistep_reaction_smiles(forward_steps),
                    "synthesis_tokens": _synthesis_tokens(synthesis),
                    "building_blocks": _building_blocks(synthesis),
                    "num_forward_steps": len(forward_steps),
                    "effort": request.effort,
                    "search": {
                        **preset,
                        "overrides_applied": request.overrides is not None,
                        "modal_request_batch_size": getattr(self, "_last_batch_size", 1),
                        "cache_hit": False,
                        "cache_ttl_seconds": CACHE_TTL_SECONDS,
                    },
                    **{key: _jsonable(value) for key, value in row.items()},
                    "forward_valid": validation.valid,
                    "forward_error": validation.error,
                    "forward_candidate_count": validation.final_candidate_count,
                    "forward_steps": forward_steps,
                }
            )
        return routes

    def _evict_expired_or_oldest_cache_entry(self) -> None:
        now = time.time()
        for key, (expires_at, _) in list(self._route_cache.items()):
            if expires_at <= now:
                del self._route_cache[key]
        if len(self._route_cache) <= CACHE_MAX_ITEMS:
            return

        oldest_key = min(self._route_cache, key=lambda key: self._route_cache[key][0])
        del self._route_cache[oldest_key]

    @staticmethod
    def _copy_route(route: dict[str, Any]) -> dict[str, Any]:
        copied = {}
        for key, value in route.items():
            if isinstance(value, dict):
                copied[key] = value.copy()
            elif isinstance(value, list):
                copied[key] = [item.copy() if isinstance(item, dict) else item for item in value]
            else:
                copied[key] = value
        return copied


@app.cls(
    image=worker_image,
    gpu="A10G",
    timeout=60 * 60,
    scaledown_window=5 * 60,
    volumes={"/vol": volume.read_only()},
)
class ReaSynService(_ReaSynRuntimeMixin):
    @modal.enter()
    def load(self) -> None:
        self._load_engine()

    @modal.batched(max_batch_size=BATCH_MAX_REQUESTS, wait_ms=BATCH_WAIT_MS)
    def sample_many(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._last_batch_size = len(requests)
        return [self._sample_payload(request) for request in requests]


@app.cls(
    image=worker_image,
    gpu="A10G",
    timeout=60 * 60,
    scaledown_window=5 * 60,
    volumes={"/vol": volume.read_only()},
)
class ReaSynStreamService(_ReaSynRuntimeMixin):
    @modal.enter()
    def load(self) -> None:
        self._load_engine()

    @modal.method()
    def sample_stream(self, request: dict[str, Any]):
        yield from self._stream_payload(request)


@app.function(
    image=asset_image,
    timeout=6 * 60 * 60,
    volumes={"/vol": volume},
)
def hydrate_checkpoints() -> list[str]:
    from huggingface_hub import hf_hub_download

    downloads = [
        ("nvidia/NV-ReaSyn-AR-166M-v2", "nv-reasyn-ar-166m-v2.ckpt", AR_CKPT),
        ("nvidia/NV-ReaSyn-EB-174M-v2", "nv-reasyn-eb-174m-v2.ckpt", EB_CKPT),
    ]
    messages = []
    for repo_id, filename, target in downloads:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            messages.append(f"exists: {target}")
            continue
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copyfile(cached_path, target)
        messages.append(f"downloaded: {repo_id}/{filename} -> {target}")

    volume.commit()
    return messages


@app.function(
    image=web_image,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name(AUTH_SECRET_NAME)],
)
@modal.fastapi_endpoint(method="POST", label="reasyn-routes", docs=True)
def routes(request: RoutesRequest, authorization: str | None = Header(default=None)) -> RoutesResponse:
    _require_auth(authorization)
    return RoutesResponse.model_validate(ReaSynService().sample_many.remote(request.model_dump()))


@app.function(
    image=web_image,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name(AUTH_SECRET_NAME)],
)
@modal.fastapi_endpoint(method="POST", label="reasyn-routes-stream", docs=True)
def routes_stream(request: RoutesRequest, authorization: str | None = Header(default=None)) -> StreamingResponse:
    _require_auth(authorization)

    def events():
        yield _sse({"event": "stream_connected"})
        for event in ReaSynStreamService().sample_stream.remote_gen(request.model_dump()):
            yield _sse(event)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.local_entrypoint()
def main(
    smiles: str = "O=C(Nc1ccc(F)cc1)N(Cc1noc(C2CC2)n1)c1ccc(Cl)cc1Cl",
    k: int = 3,
    effort: Effort = "low",
    verbose: bool = False,
) -> None:
    payload = RoutesRequest(
        smiles=[s.strip() for s in smiles.split(",") if s.strip()],
        k=k,
        effort=effort,
        verbose=verbose,
    )
    print(ReaSynService().sample_many.remote(payload.model_dump()))


@app.local_entrypoint()
def hydrate() -> None:
    for message in hydrate_checkpoints.remote():
        print(message)
