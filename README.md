<h1 align="center">Exploring Sythesizable Chemical Space with Iterative Pathway Refinements</h1>

This is the official code repository for the paper titled [Exploring Sythesizable Chemical Space with Iterative Pathway Refinements](https://arxiv.org/abs/2509.16084).

<p align="center">
    <img width="750" src="assets/concept.png"/>
</p>

<p align="center">
    <img width="750" src="assets/concept2.png"/>
</p>

*Abstract:*
A well-known pitfall of molecular generative models is that they are not guaranteed
to generate synthesizable molecules. Existing solutions for this problem often struggle
to effectively navigate exponentially large combinatorial space of synthesizable
molecules and suffer from poor coverage. To address this problem, we introduce
ReaSyn, an iterative generative pathway refinement framework that obtains synthesizable
analogs to input molecules by projecting them onto synthesizable space.
Specifically, we propose a simple synthetic pathway representation that allows for
generating pathways in both bottom-up and top-down traversal of synthetic trees.
We design ReaSyn so that both bottom-up and top-down pathways can be sampled
with a single unified autoregressive model. ReaSyn can thus iteratively refine subtrees
of generated synthetic trees in a bidirectional manner. Further, we introduce a
discrete flow model that refines the generated pathway at the entire pathway level
with edit operations: insertion, deletion, and substitution. The iterative refinement
cycle of (1) bottom-up decoding, (2) top-down decoding, and (3) holistic editing
constitutes a powerful pathway reasoning strategy, allowing the model to explore
the vast space of synthesizable molecules. Experimentally, ReaSyn achieves the
highest reconstruction rate and pathway diversity in synthesizable molecule reconstruction
and the highest optimization performance in synthesizable goal-directed
molecular optimization, and significantly outperforms previous synthesizable projection
methods in synthesizable hit expansion. These results highlight ReaSyn’s
superior ability to navigate combinatorially-large synthesizable chemical space.

Find the Model Card++ for ReaSyn [here](model_card/overview.md).

## Installation

For inference-focused development, use `uv`:
```bash
uv sync --group dev
```

This installs the default inference dependencies. Training and benchmark-only
dependencies are optional:
```bash
uv sync --extra train --extra eval --group dev
```

The packaged command line entry points are:
```bash
uv run reasyn-sample --help
uv run reasyn-sample-one --help
uv run reasyn-preprocess --help
uv run reasyn-prepare-mcule-building-blocks --help
uv run --extra ui reasyn-ui --help
```

The original Conda environment is retained for reference:
```bash
conda env create -f env.yml
conda activate reasyn
```

## Inference Runtime Assets

Inference needs these runtime assets, which are not committed to this repository:

- `data/trained_model/nv-reasyn-ar-166m-v2.ckpt`
- `data/trained_model/nv-reasyn-eb-174m-v2.ckpt`
- `data/processed/comp_2048/fpindex.pkl`
- `data/processed/comp_2048/matrix.pkl`

Checkpoint configs refer to the processed chemistry files using relative paths.
For Docker or Modal volume mounts, pass `--asset-dir` to resolve those paths
under the mounted asset root.

Example batch inference:
```bash
uv run reasyn-sample \
  --asset-dir . \
  -m data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt \
  -i data/test_zinc250k.txt \
  -o results/zinc250k.csv \
  --num-cycles 16
```

Example single-molecule inference:
```bash
uv run reasyn-sample-one \
  --asset-dir . \
  -m data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt \
  -s 'O=C(Nc1ccc(F)cc1)N(Cc1noc(C2CC2)n1)c1ccc(Cl)cc1Cl'
```

### MCule Building Blocks

The MCule CSV in this checkout is raw supplier data. ReaSyn's `--add-bb-path`
expects a pickled `FingerprintIndex`, not a CSV. Prepare it with:
```bash
uv run reasyn-prepare-mcule-building-blocks --force
```

By default this reads `mcule_unique_bb_instock_library_260130.csv`, extracts
zero-based column `5` as SMILES, writes canonical SMILES to
`data/building_blocks/building_blocks_mcule.txt`, writes the MCule
building-block index to `data/processed/mcule_2048/fpindex.pkl`, and writes
the MCule reaction matrix to `data/processed/mcule_2048/matrix.pkl`.

Use it during inference with:
```bash
uv run reasyn-sample-one \
  --asset-dir . \
  --add-bb-path data/processed/mcule_2048/fpindex.pkl \
  -m data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt \
  -s 'CCO'
```

### Docker

Build the image:
```bash
docker build -t reasyn .
```

Run with model and chemistry assets mounted into `/app/data`:
```bash
docker run --rm --gpus all \
  -v "$PWD/data:/app/data" \
  reasyn \
  --asset-dir /app \
  -m data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt \
  -i data/test_zinc250k.txt \
  -o results/zinc250k.csv
```

### Modal Integration

`modal_app.py` contains a deployable Modal app with separable parts:

- `ReaSynService`: the GPU worker that loads checkpoints, MCule assets, runs
  inference, validates routes by forward-executing RDKit reactions, and uses
  Modal dynamic batching.
- `ReaSynStreamService`: a generator worker with the same runtime that emits
  sampler progress events while inference is running.
- `routes`: a small FastAPI HTTP endpoint with bearer-token auth. This is a
  convenience wrapper around `ReaSynService`, not a requirement.
- `routes_stream`: a Server-Sent Events endpoint for UI/demo clients that want
  intermediate sampler progress. The regular `routes` endpoint remains the
  best choice for high-throughput batched inference.

The public request schema is intentionally small:

```json
{
  "smiles": ["CCO", "c1ccccc1"],
  "k": 2,
  "effort": "medium",
  "timeout_seconds": 300,
  "verbose": false,
  "overrides": {
    "search_width": 2,
    "exhaustiveness": 4
  }
}
```

`effort` must be `low`, `medium`, or `high`; it controls the search shape
(`search_width`, `exhaustiveness`, cycles, and editflow settings), not the wall
clock budget. `timeout_seconds` controls how long the worker keeps searching for
exact RDKit-forward routes to the requested target. `overrides` is optional and
is intended for demos or debugging. Responses only include routes whose
forward-executed product exactly matches the requested target; if none are
found before timeout, the molecule returns an empty route list with an error
message. `verbose=false` returns only RDKit-forward reaction SMILES for each
exact route. `verbose=true` also returns the generated molecule, scores,
effective search settings, stack tokens, building blocks, a chained multistep
reaction SMILES string, `forward_valid`, `forward_error`,
`forward_candidate_count`, and `forward_steps`.

The GPU worker keeps a short per-container cache for `(smiles, k, effort,
timeout_seconds, effective search settings)`. `verbose` is intentionally not
part of the cache key, so a client can first request compact output and then
re-request the same inputs with `verbose=true` to get details without rerunning
inference when the request lands on the same warm worker.

#### Deploy The Included Endpoint

Install Modal support and authenticate:

```bash
uv sync --extra modal
uv run --extra modal modal setup
```

Prepare the MCule runtime assets locally:

```bash
uv run reasyn-prepare-mcule-building-blocks --force
```

Create the Modal volume and upload the MCule assets:

```bash
uv run --extra modal modal volume create reasyn-assets
uv run --extra modal modal volume put reasyn-assets \
  data/processed/mcule_2048 \
  /reasyn/data/processed/mcule_2048 \
  -f
```

Hydrate model checkpoints into the same volume. This downloads the NVIDIA
ReaSyn checkpoints from Hugging Face inside Modal:

```bash
uv run --extra modal modal run modal_app.py::hydrate
```

Create or rotate the bearer-token secret used by the included endpoint:

```bash
export REASYN_AUTH_TOKEN="$(openssl rand -hex 32)"
uv run --extra modal modal secret create reasyn-web-auth \
  REASYN_AUTH_TOKEN="$REASYN_AUTH_TOKEN" \
  --force
```

Deploy:

```bash
uv run --extra modal modal deploy modal_app.py --name reasyn
```

Test:

```bash
curl -X POST https://<workspace>--reasyn-routes.modal.run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $REASYN_AUTH_TOKEN" \
  -d '{
    "smiles": ["O=C(O)c1ccccc1"],
    "k": 2,
    "effort": "medium",
    "timeout_seconds": 300,
    "verbose": false
  }'
```

The streaming endpoint uses the same auth and request schema, but returns
`text/event-stream`:

```bash
curl -N -X POST https://<workspace>--reasyn-routes-stream.modal.run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $REASYN_AUTH_TOKEN" \
  -d '{
    "smiles": ["O=C(O)c1ccccc1"],
    "k": 2,
    "effort": "medium",
    "timeout_seconds": 300,
    "verbose": true
  }'
```

Progress events include cache status, phase boundaries, autoregressive step
completion, active/finished/aborted state counts, best score so far, and final
per-molecule route payloads. Verbose route steps include the actual forward
reactants/products, reaction template SMARTS, template-level annotations from
`data/rxn_templates/comprehensive_named.tsv`, reactant template SMARTS, agent
template SMARTS when present, and product template SMARTS. The bundled template
annotations describe reaction classes; they still do not provide exact named
reagents, catalysts, solvents, or conditions. Streaming is intended for
inspection and UI feedback; it does not use Modal dynamic batching because
generator calls need to flush events as they happen.

#### Bring Your Own FastAPI/Auth

If you already have server logic, keep `ReaSynService` and replace or ignore the
included `routes` function. Your own Modal web function can call the same
batched GPU worker:

```python
import modal
from fastapi import Header, HTTPException

from modal_app import ReaSynService, RoutesRequest, RoutesResponse, app, web_image


@app.function(image=web_image, secrets=[modal.Secret.from_name("your-secret")])
@modal.fastapi_endpoint(method="POST", label="your-routes")
def your_routes(request: RoutesRequest, authorization: str | None = Header(default=None)) -> RoutesResponse:
    # Apply your own API keys, JWTs, tenant checks, rate limits, logging, etc.
    if not your_auth_check(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")

    return RoutesResponse.model_validate(
        ReaSynService().sample_many.remote(request.model_dump())
    )
```

For a non-Modal API server, deploy only the Modal worker app and call the remote
method from your service process:

```python
from modal_app import ReaSynService

payload = {
    "smiles": ["O=C(O)c1ccccc1"],
    "k": 2,
    "effort": "medium",
    "timeout_seconds": 300,
    "verbose": True,
}
result = ReaSynService().sample_many.remote(payload)
```

For a custom progress endpoint, wrap the worker generator in your own SSE or
WebSocket logic:

```python
import json

from fastapi.responses import StreamingResponse
from modal_app import ReaSynStreamService


def sse(event: dict) -> str:
    return f"event: {event.get('event', 'message')}\ndata: {json.dumps(event)}\n\n"


def your_stream(payload: dict) -> StreamingResponse:
    def events():
        for event in ReaSynStreamService().sample_stream.remote_gen(payload):
            yield sse(event)

    return StreamingResponse(events(), media_type="text/event-stream")
```

Keep these constraints in mind when bringing your own server:

- The worker expects assets in Modal volume `reasyn-assets` under `/vol/reasyn`.
- `sample_many` accepts a list of request dictionaries because it is dynamically
  batched by Modal; each dictionary uses the same schema as the HTTP endpoint.
- Leave batching on the worker side. Your server can send one request per user
  call; Modal coalesces concurrent calls up to `BATCH_MAX_REQUESTS`.
- For progress UIs, expose your own SSE/WebSocket route and forward events from
  `ReaSynStreamService().sample_stream.remote_gen(...)`. Use `sample_many` for
  throughput-oriented JSON APIs and `sample_stream` for observability.
- If you change auth or request models, keep `RoutesRequest` compatible or add
  an adapter before calling `sample_many.remote(...)`.

### Local Route Explorer UI

The local UI is a small FastAPI app that proxies requests to the Modal endpoint
and renders molecules/reactions through `http://mol2txt.app`. The browser calls
only the local proxy, so the Modal bearer token stays server-side and CORS is not
an issue.

Run it:

```bash
export REASYN_AUTH_TOKEN="<your Modal bearer token>"
export REASYN_MODAL_ENDPOINT="https://<workspace>--reasyn-routes.modal.run"
export REASYN_MODAL_STREAM_ENDPOINT="https://<workspace>--reasyn-routes-stream.modal.run"
uv run --extra ui reasyn-ui
```

Then open:

```text
http://127.0.0.1:8765
```

The UI always sends `verbose=true` to Modal and uses the SSE endpoint by default
so it can show how the algorithm is behaving: timeout, search profile, optional
overrides, cache hits, current sampler phase, active/finished state counts, exact
route count, best score, generated product, stack tokens, building blocks, RDKit
validation status, reaction class annotations, reaction template SMARTS, and the
concrete forward reaction SMILES for each step. It only displays exact
RDKit-forward routes to the requested target and always sends reactions and
molecules to `mol2txt.app` as PNG images.

Environment variables:

- `REASYN_AUTH_TOKEN`: bearer token for the deployed Modal endpoint. If unset,
  the UI also checks `/tmp/reasyn_web_auth_token`.
- `REASYN_MODAL_ENDPOINT`: Modal endpoint URL. Defaults to the deployed example
  endpoint used during development.
- `REASYN_MODAL_STREAM_ENDPOINT`: Modal SSE endpoint URL. Defaults to the
  deployed example stream endpoint used during development.
- `REASYN_RENDERER_ENDPOINT`: molecule/reaction renderer. Defaults to
  `http://mol2txt.app/`.

## Data Preparation

### Reaction Teamplates
We use the 115 [reaction templates](https://github.com/wenhao-gao/synformer/tree/main/data/rxn_templates) used in [SynFormer](https://github.com/wenhao-gao/synformer). Place the data as `data/rxn_templates/comprehensive.txt`.

### Enamine Building Blocks
The building blocks used in the paper are from Enamine US Stock catalog, which are available upon request.<br>
After requesting the data from [Enamine](https://enamine.net/building-blocks/building-blocks-catalog), place the data as `data/building_blocks/building_blocks.txt`.<br>
Then, run the following command to preprocess the data:
```bash
python scripts/preprocess.py --model-config configs/train.yml
```

Alternatively, you can directly use preprocessed building block data.<br>
To resolve pickle path compatibility, first clone the [SynFormer](https://github.com/wenhao-gao/synformer) repository into the top-level directory for ReaSyn (`/ReaSyn`):
```bash
git clone https://github.com/wenhao-gao/synformer.git
cd synformer
pip install --no-deps -e .
pip install scikit-learn==1.6.0 # 1.6.0 is required to load fpindex.pkl
cd ..
```
Then, download the [preprocessed data](https://huggingface.co/whgao/synformer). Place `fpindex.pkl` and `matrix.pkl` in the folder `data/processed/comp_2048`.<br>
Then, run the following command:
```bash
python scripts/convert_processed_1.py
pip install scikit-learn==1.2.2 # 1.2.2 is required for hit expansion later
pip uninstall synformer         # optional; you may delete the synformer package
rm -rf synformer                # optional; you may delete the synformer folder
python scripts/convert_processed_2.py
```

### ZINC250k Building Blocks
For the synthesizable molecule reconstruction task on ZINC250k, we provide additional building blocks in `data/building_blocks/building_blocks_zinc250k.txt`.<br>
These are the molecules from ZINC250k that have more than 18 heavy atoms.<br>
Run the following command to preprocess the data:
```bash
python scripts/preprocess.py --model-config configs/preprocess_zinc250k.yml
```

## Training

We provide the trained model checkpoint via NGC ([AR](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/nv-reasyn-ar-166m-v2.ckpt?version=2.0) and [EB](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/nv-reasyn-eb-174m-v2.ckpt?version=2.0)) and HuggingFace ([AR](https://huggingface.co/nvidia/NV-ReaSyn-AR-166M-v2) and [EB](https://huggingface.co/nvidia/NV-ReaSyn-EB-174M-v2)). Place `nv-reasyn-ar-166m-v2.ckpt` and `nv-reasyn-eb-174m-v2.ckpt` in the `data/trained_model` directory.

### Autoregressive Model
Run the following command to train ReaSyn's autoregressive model for bottom-up and top-down pathway generation:
```bash
torchrun --nnodes $NUM_NODES --nproc_per_node $SUBMIT_GPUS \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK \
         scripts/train.py -n ${exp_name} -c configs/train_ar.yml
```
We used 8 NVIDIA A100 GPUs. Training for 500k steps took 5~6 days.

### Edit Bridge Model

To train ReaSyn's Edit Bridge model, we generated the dataset of `(target molecule, AR-predicted pathway, true pathway)` triplets offline.<br>
Specifically, given a `(target molecule, true pathway)` pair, we first generated `AR-predicted pathway` with the trained AR model.<br>
Then, the aligned `(AR-predicted pathway, true pathway)` pair is obtained via the alignment process (Section B of the paper).<br>

Run the following command to prepare the training data for ReaSyn's Edit Bridge model:
```bash
python scripts/editflow_data_generate_x0.py -c configs/train_eb.yml -m ${pretrained_path} -d ${data_path}
python scripts/editflow_data_align.py -c configs/train_eb.yml -d ${data_path}
```
`pretrained_path` is the trained AR model path, e.g., `data/trained_model/nv-reasyn-ar-166m-v2.ckpt`.<br>
Running the first command creates a temporary folder `data_path_x0`. You may delete this folder after running the second command.<br>
Generating 10.5M data points with 120 NVIDIA A100 GPUs took ~3 days.

Run the following command to train ReaSyn's Edit Bridge model for holistic pathway editing:
```bash
torchrun --nnodes $NUM_NODES --nproc_per_node $SUBMIT_GPUS \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK \
         scripts/train.py -n ${exp_name} -c configs/train_eb.yml -b 128 -d ${data_path}
```
We used 8 NVIDIA A100 GPUs. Training for 500k steps took ~5 days.

## Inference

### Synthesizable Molecule Reconstruction
Our paper evaluated ReaSyn on three test sets.<br>
For the Enamine and ChEMBL test sets, place `enamine_smiles_1k.txt` and `chembl_filtered_1k.txt` from [SynFormer](https://github.com/wenhao-gao/synformer/tree/main/data) in the `data` folder.<br>
For the ZINC250k test set, we provide `data/test_zinc250k.txt`.

Run the following command to conduct synthesizable molecule reconstruction:
```bash
python scripts/sample.py -m ${model_path} -i ${testset_path} -o ${output_path} --num_cycles ${num_cycles}
# python scripts/sample.py -m ${model_path} -i data/enamine_smiles_1k.txt -o results/enamine.txt --num_cycles 12
# python scripts/sample.py -m ${model_path} -i data/chembl_filtered_1k.txt -o results/chembl.txt --num_cycles 24
# python scripts/sample.py -m ${model_path} -i data/test_zinc250k.txt -o results/zinc250k.txt --num_cycles 16 --add_bb_path data/processed/zinc250k_2048/fpindex.pkl
python scripts/eval_recon.py ${output_path}
```
`model_path` is a comma-separated string of the AR and EB model paths, e.g., `data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt`.<br>
We recommend using multiple GPUs for parallelized synthesizable molecule reconstruction.

### Synthesizable Goal-directed Optimization of TDC Oracles
Run the following command to conduct synthesizable goal-directed optimization of TDC oracles:
```bash
python scripts/optimize_tdc.py -m ${model_path} -o ${oracle}
```
`model_path` is a comma-separated string of the AR and EB model paths, e.g., `data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt`.

### Synthesizable Hit Expansion
Run the following command to conduct synthesizable hit expansion:
```bash
python scripts/sample.py -m ${model_path} -i data/jnk3_hit.txt -o ${output_path} --search_width 12 --exhaustiveness 128 --num_cycles 12 --no_exact_break
python scripts/eval_hit.py ${output_path}
```
`model_path` is a comma-separated string of the AR and EB model paths, e.g., `data/trained_model/nv-reasyn-ar-166m-v2.ckpt,data/trained_model/nv-reasyn-eb-174m-v2.ckpt`.

### (Optional) Filtering Pathways
We additionally provide the functionality to filter out generated pathways that lead to molecules that users want to avoid (e.g., toxic molecules). We provide an example catalog of toxic molecules in `data/mols_to_filter.txt`. Set `mols_to_filter` and `filter_sim` arguments to filter synthetic pathways for molecules whose Tanimoto similarity to `mols_to_filter` is greater than `filter_sim`.<br>
For example:
```bash
python scripts/sample.py -m ${model_path} -i ${testset_path} -o ${output_path} --num_cycles ${num_cycles} --mols_to_filter data/mols_to_filter.txt --filter_sim ${filter_sim}
```

## License
Copyright @ 2025, NVIDIA Corporation. All rights reserved.<br>
The source code is made available under Apache-2.0.<br>
The model weights are made available under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Contributing
This project is currently not accepting external contributions.

## Citation
If you find this repository and our paper useful, we kindly request to cite our work.
```BibTex
@article{lee2025reasyn,
  title     = {Exploring Sythesizable Chemical Space with Iterative Pathway Refinements},
  author    = {Lee, Seul and Kreis, Karsten and Veccham, Srimukh Prasad and Liu, Meng and Reidenbach, Danny and Paliwal, Saee and Nie, Weili and Vahdat, Arash},
  journal   = {arXiv},
  year      = {2025}
}
```
