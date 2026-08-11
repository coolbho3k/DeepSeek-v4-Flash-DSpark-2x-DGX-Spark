#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

DSPARK_VLLM_IMAGE="${DSPARK_VLLM_IMAGE:-vllm-dspark-runtime:anemll-nvfp4-416-experimental}"
DSPARK_BASE_IMAGE="${DSPARK_BASE_IMAGE:-vllm-dspark-runtime:mia-raf-pr1}"
DSPARK_BUILD_STAGE="${DSPARK_BUILD_STAGE:-anemll-416}"
DSPARK_STAGE_C_IMAGE="${DSPARK_STAGE_C_IMAGE:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
DSPARK_ANEMLL_BASE_IMAGE="${DSPARK_ANEMLL_BASE_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"
WORKER_BUILD="${WORKER_BUILD:-1}"

case "$DSPARK_BUILD_STAGE" in
  stage-c|stage-d-416|anemll-416) ;;
  *)
    echo "DSPARK_BUILD_STAGE must be stage-c, stage-d-416, or anemll-416 (got $DSPARK_BUILD_STAGE)" >&2
    exit 2
    ;;
esac

"$SCRIPT_DIR/scripts/verify-overlay-sources.sh"

build_one() {
  local host="$1"
  local checkout="$2"
  if [ "$host" = "local" ]; then
    if [ "$DSPARK_BUILD_STAGE" = "anemll-416" ]; then
      docker build \
        --build-arg BASE_IMAGE="$DSPARK_ANEMLL_BASE_IMAGE" \
        -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.anemll-416" \
        -t "$DSPARK_VLLM_IMAGE" \
        "$SCRIPT_DIR"
      docker run --rm --entrypoint python3 "$DSPARK_VLLM_IMAGE" -c \
        "from pathlib import Path; from vllm.models.deepseek_v4.nvidia.nvfp4_cache import RECORD_BYTES; source = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py').read_text(); assert RECORD_BYTES == 416; assert 'attn.kv_cache_dtype == \"nvfp4_ds_mla\"' in source; assert 'rope_store_swa_nvfp4_416' in source; print('dspark anemll nvfp4-416 image ok')"
      return
    fi
    local stage_c_image="$DSPARK_VLLM_IMAGE"
    if [ "$DSPARK_BUILD_STAGE" = "stage-d-416" ]; then
      stage_c_image="$DSPARK_STAGE_C_IMAGE"
    fi
    docker build \
      -f "$SCRIPT_DIR/recipe/Dockerfile.dspark-runtime-overlay" \
      -t "$DSPARK_BASE_IMAGE" \
      "$SCRIPT_DIR/recipe/overlay"
    docker run --rm --entrypoint /opt/env/bin/python "$DSPARK_BASE_IMAGE" -c \
      "import vllm.v1.spec_decode.dspark as d; import vllm.v1.spec_decode.dspark_proposer as p; print('dspark overlay ok', d.__name__, p.__name__)"
    docker build \
      --build-arg BASE_IMAGE="$DSPARK_BASE_IMAGE" \
      -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-a" \
      -t "$DSPARK_BASE_IMAGE-nvfp4-a" \
      "$SCRIPT_DIR"
    docker build \
      --build-arg BASE_IMAGE="$DSPARK_BASE_IMAGE-nvfp4-a" \
      -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-b" \
      -t "$DSPARK_BASE_IMAGE-nvfp4-b" \
      "$SCRIPT_DIR"
    docker build \
      --build-arg BASE_IMAGE="$DSPARK_BASE_IMAGE-nvfp4-b" \
      -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-c" \
      -t "$stage_c_image" \
      "$SCRIPT_DIR"
    docker run --rm --entrypoint /opt/env/bin/python "$stage_c_image" -c \
      "import vllm; print('dspark nvfp4 stage-c image ok', vllm.__version__)"
    if [ "$DSPARK_BUILD_STAGE" = "stage-d-416" ]; then
      docker build \
        --build-arg BASE_IMAGE="$stage_c_image" \
        -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-d-416" \
        -t "$DSPARK_VLLM_IMAGE" \
        "$SCRIPT_DIR"
      docker run --rm --entrypoint /opt/env/bin/python "$DSPARK_VLLM_IMAGE" -c \
        "from vllm.models.deepseek_v4.nvidia.nvfp4_cache import RECORD_BYTES; assert RECORD_BYTES == 416; print('dspark nvfp4 stage-d-416 image ok')"
    fi
  else
    ssh "$host" "mkdir -p '$checkout'"
    rsync -az "$SCRIPT_DIR/" "$host:$checkout/"
    ssh "$host" "cd '$checkout' && DSPARK_BASE_IMAGE='$DSPARK_BASE_IMAGE' DSPARK_VLLM_IMAGE='$DSPARK_VLLM_IMAGE' DSPARK_BUILD_STAGE='$DSPARK_BUILD_STAGE' DSPARK_STAGE_C_IMAGE='$DSPARK_STAGE_C_IMAGE' DSPARK_ANEMLL_BASE_IMAGE='$DSPARK_ANEMLL_BASE_IMAGE' WORKER_BUILD=0 ./build-dspark-vllm-runtime.sh"
  fi
}

build_one local "$SCRIPT_DIR"

if [ "$WORKER_BUILD" = "1" ]; then
  : "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
  build_one "$WORKER_HOST" "${WORKER_CHECKOUT:-${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}}"
fi
