# Environment variable matrix (Anemll 0.1.1 vs Stage-C overlay)

This recipe defaults to the prebuilt image:

```text
ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

A large set of `VLLM_DSPARK_*` / extra B12X knobs still appear in historical
Stage-C docs and in `recipe/overlay/vllm/envs.py`. **Those symbols are
registered in the Stage-C overlay build**, not necessarily in the Anemll
prebuilt image.

vLLM validates process environment keys that start with `VLLM_`. Unknown keys
log:

```text
Unknown vLLM environment variable detected: VLLM_…
```

and are **ignored** (warning only; serve still starts).

> **Important:** missing env registration does **not** mean DSpark or the Keys
> concurrency patches are absent from Anemll. Logic may be baked into the image
> without exposing every Stage-C kill-switch. Conversely, setting a Stage-C-only
> env on Anemll does **not** enable that kill-switch.

Audit date: **2026-07-29**, image tag **`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`**,
by inspecting `vllm.envs.environment_variables` inside the container and
comparing to `recipe/overlay/vllm/envs.py` in this repo.

Re-check after image bumps:

```bash
docker run --rm --entrypoint python3 ghcr.io/anemll/dspark-vllm-gx10:0.1.1 - <<'PY'
import pathlib, vllm
ns = {}
exec(compile((pathlib.Path(vllm.__file__).parent / "envs.py").read_text(), "envs.py", "exec"), ns)
keys = ns["environment_variables"]
for k in sorted(keys):
    if any(s in k for s in ("B12", "DSPARK", "DSV4", "SPARSE_INDEXER", "FLASHINFER_SAMPLER")):
        print(k)
PY
```

---

## Compose / `.env` knobs by lane

### A. Safe on Anemll 0.1.1 (registered `VLLM_*` or non-`VLLM_` runtime)

| Variable | Role |
|----------|------|
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Allow long context configs |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | Sparse indexer workspace cap |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | Profiler / capture estimate |
| `VLLM_USE_FLASHINFER_SAMPLER` | FlashInfer sampler |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | Set `0` to opt out of DS4's automatic breakable-graph mode and retain regular CUDA graphs |
| `VLLM_USE_B12X_MOE` | Enable B12X MoE path |
| `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM` | Experimental W4A16 selector |
| `VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M` | Experimental W4A16 selector |
| `VLLM_B12X_W4A16_FORCE_TILE_CONFIG` | Experimental W4A16 selector |
| `VLLM_HOST_IP` | Distributed bind address |
| `VLLM_CACHE_ROOT` | vLLM cache root (compose sets path) |
| `TRITON_CACHE_DIR` | Persistent Triton JIT cache under the node-local `HF_CACHE` mount |
| `TILELANG_CACHE_DIR` | Persistent TileLang kernel cache under the node-local `HF_CACHE` mount |
| `CUTE_DSL_ARCH` | **Not** `VLLM_*` — CuTeDSL/b12x compile target (`sm_121a` on GB10) |
| `TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST` | Build/JIT arch lists |
| `NCCL_*` / `TP_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | Fabric |
| `HF_*` / `TRANSFORMERS_OFFLINE` | Hub cache behavior |
| `MTP_NUM_TOKENS` | Consumed by compose command line (not a vLLM env registry key) |

### B. Stage-C / overlay-registered only (warn + no-op on Anemll 0.1.1)

These appear in `recipe/overlay/vllm/envs.py` and in older validated Stage-C
lanes. On Anemll **0.1.1** they are **not** in `environment_variables` and only
produce unknown-env warnings if injected.

| Variable | Stage-C intent (summary) |
|----------|---------------------------|
| `VLLM_USE_B12X_WO_PROJECTION` | B12X WO projection path |
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` | Draft confidence threshold |
| `VLLM_DSPARK_CONFIDENCE_SCHEDULER` | Confidence scheduler mode |
| `VLLM_DSPARK_LOCAL_ARGMAX` | Local argmax draft path |
| `VLLM_DSPARK_REPLICATE_MARKOV_W1` | Markov W1 replicate |
| `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` | Fused Markov argmax |
| `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK` | GPU rejected-context mask (Keys ragged path switch in overlay) |
| `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` | Reference KV quant/dequant |
| `VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP` | Hardware scheduler early stop |
| `VLLM_DSV4_B12X_COMPRESSED_MLA` | Compressed MLA experiment |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE` | Defer target cudagraph capture |
| `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT` | Exact defer variant |

The optimized Anemll compose **does not** inject these. For Stage-C images, merge:

```bash
docker compose --env-file .env.dspark \
  -f docker-compose.dspark.yml \
  -f docker-compose.stage-c.override.yml \
  up -d
```

(see `docker-compose.stage-c.override.yml`).

### C. Not registered as `VLLM_*` on either lane (or host-only)

| Variable | Notes |
|----------|--------|
| `VLLM_TRITON_MLA_SPARSE` | Not in Anemll 0.1.1 registry; not found as overlay registration in the same form — avoid on Anemll |
| `VLLM_SKIP_INIT_MEMORY_CHECK` | Not in Anemll 0.1.1 registry — avoid on Anemll |
| `DSPARK_SLOT_CLAMP` | Non-`VLLM_` prefix (no unknown-`VLLM_` warning). Only meaningful if the image reads it; treat as Stage-C/overlay unless confirmed |
| `B12X_W4A16_TC_DECODE` | Non-`VLLM_` package/debug knob |
| `VLLM_HOST` / `VLLM_PORT` | Used by **compose command substitution** / start scripts, not as in-process vLLM config envs in the same way as registry keys |
| `DSPARK_MODEL`, `DSPARK_VLLM_IMAGE`, `ENABLE_VLLM_GB10_PATCH`, … | Launcher / compose only |

---

## Recommended defaults by image

### Optimized Anemll 416 image (feature-branch default)

`vllm-dspark-runtime:anemll-nvfp4-416-experimental` is built from
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` and retains its environment registry.

Keep the slim set in `.env.dspark.example` + `docker-compose.dspark.yml`:

- Serve profile: `MAX_NUM_SEQS=4`, `MTP_NUM_TOKENS=5`, capture `max_num_seqs * (k+1)`, `GPU_MEMORY_UTILIZATION≈0.80`
- `LONG_PREFILL_TOKEN_THRESHOLD=2048`, `SCHEDULING_POLICY=priority`, `DEFAULT_THINKING=max`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (explicit opt-out; omission auto-enables the slower breakable path on DS4)
- `VLLM_USE_B12X_MOE=1`
- `CUTE_DSL_ARCH=sm_121a` (GB10 CuTeDSL target; prevents slower JIT fallbacks)
- Do **not** rely on Stage-C-only `VLLM_DSPARK_*` for behavior on this tag

### Stage-C `vllm-dspark-runtime:dspark-nvfp4-stage-c`

- Build via `./build-dspark-vllm-runtime.sh`
- Set `DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c`
- Enable the Stage-C override compose file and the Stage-C block in `.env.dspark.example`
- Then the Keys-oriented switches (e.g. `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`) are meaningful

### Stage-D `vllm-dspark-runtime:dspark-nvfp4-416-experimental`

- Build with `DSPARK_BUILD_STAGE=stage-d-416 ./build-dspark-vllm-runtime.sh`
- Set `DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-416-experimental`
- Merge `docker-compose.stage-c.override.yml` (the launcher does this
  automatically for `DSPARK_BUILD_STAGE=stage-d-416`) so mixed-length DSpark
  batches use `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- Keep `MAX_MODEL_LEN=1048576`, `ENFORCE_EAGER=1`, `MOE_BACKEND=b12x`, and
  `DG_JIT_NVCC_COMPILER=/opt/env/bin/nvcc`
- The validated two-Spark profile uses `GPU_MEMORY_UTILIZATION=0.835`,
  `MAX_NUM_SEQS=6`, and `MAX_NUM_BATCHED_TOKENS=8192`
- Stage D uses the same Stage-C registry surface plus the true 416-byte
  DeepSeek V4 NVFP4 writer/gather/reference-attention overlay

---

## What this does *not* claim

- It does **not** invalidate published Anemll decode benches. Throughput can be
  real while unused envs only add log noise.
- It does **not** assert Anemll lacks concurrency fixes—only that several
  **env kill-switches** from the overlay are not exposed on 0.1.1.
- Image tags after 0.1.1 may register more keys; re-run the audit snippet above.
