# ReaSyn Modal Endpoint Assets

This folder contains the public source data used by the Modal deployment in
`modal_app.py`. It is meant to make the Modal integration portable to another
repo without checking generated pickle assets into Git.

## Files

- `source_data/mcule_unique_bb_instock_library_260130.csv`: raw MCule supplier
  data. The hydrate step extracts zero-based column `5` as SMILES.
- `source_data/comprehensive_named.tsv`: reaction template annotations. The
  hydrate step also derives `data/rxn_templates/comprehensive.txt` from this
  file, preserving `template_id` order.

Generated runtime files are not stored here. Modal builds them in the volume:

- `/vol/reasyn/data/building_blocks/building_blocks_mcule.txt`
- `/vol/reasyn/data/processed/mcule_2048/fpindex.pkl`
- `/vol/reasyn/data/processed/mcule_2048/matrix.pkl`
- `/vol/reasyn/data/trained_model/*.ckpt`

## Deploy The Endpoint

Use the fork or copy `modal_app.py` plus the patched `reasyn` package into your
repo. Then install Modal support and authenticate:

```bash
uv sync --extra modal
uv run --extra modal modal setup
```

Create the volume and hydrate it. Hydration downloads checkpoints from Hugging
Face, fetches this folder's source data from the public fork, and builds the
MCule runtime assets inside Modal:

```bash
uv run --extra modal modal volume create reasyn-assets
uv run --extra modal modal run modal_app.py::hydrate
```

By default, `modal_app.py` fetches source data from:

```text
https://raw.githubusercontent.com/whitead/ReaSyn/reasyn_v2/dist/source_data
```

If you move these files to another public location, set this before running
`hydrate`:

```bash
export REASYN_SOURCE_DATA_BASE_URL="https://<host>/<path>/dist/source_data"
```

Create the bearer-token secret and deploy:

```bash
export REASYN_AUTH_TOKEN="$(openssl rand -hex 32)"
uv run --extra modal modal secret create reasyn-web-auth \
  REASYN_AUTH_TOKEN="$REASYN_AUTH_TOKEN" \
  --force

uv run --extra modal modal deploy modal_app.py --name reasyn
```

## Call The Endpoint

The JSON endpoint is:

```text
https://<workspace>--reasyn-routes.modal.run
```

Example:

```bash
curl -X POST "https://<workspace>--reasyn-routes.modal.run" \
  -H "Authorization: Bearer $REASYN_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": ["CCO"],
    "k": 1,
    "effort": "low",
    "timeout_seconds": 300,
    "verbose": false
  }'
```

Use the SSE endpoint for progress events:

```text
https://<workspace>--reasyn-routes-stream.modal.run
```

Request fields:

- `smiles`: one or more target SMILES strings.
- `k`: number of exact routes to return per molecule.
- `effort`: `low`, `medium`, or `high`.
- `timeout_seconds`: per-molecule search timeout.
- `verbose`: when `false`, returns compact reaction SMILES routes; when `true`,
  includes search metadata, building blocks, forward-validation data, and
  reaction annotation details.

Only routes that RDKit-forward execute to the requested target are returned.
