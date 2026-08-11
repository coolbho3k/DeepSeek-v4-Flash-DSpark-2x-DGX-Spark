"""Add decode-first cadence control to the non-DP vLLM EngineCore.

VLLM_PREFILL_DECODE_CADENCE=1 preserves upstream scheduling. Values greater
than one admit prefill compute once per N scheduler iterations while at least
one decode request is running. The scheduler's existing throttle_prefills path
still allows unrestricted prefill when no decode request is active. In that
pure-prefill case, requests sequentially water-fill the complete scheduler
budget. Mixed iterations instead share one global configured prefill budget.
"""

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    source = target.read_text()
    if new in source:
        return
    if old not in source:
        raise SystemExit(
            f"missing prefill/decode cadence patch anchor in {target}: {old!r}"
        )
    target.write_text(source.replace(old, new, 1))


# Register the setting so vLLM's VLLM_* validator accepts it.
replace(
    "envs.py",
    """    VLLM_DSV4_DEMAND_SIZED_KV_POOLS: bool = True
""",
    """    VLLM_DSV4_DEMAND_SIZED_KV_POOLS: bool = True
    VLLM_PREFILL_DECODE_CADENCE: int = 1
""",
)

replace(
    "envs.py",
    """    "VLLM_DSV4_DEMAND_SIZED_KV_POOLS": lambda: os.getenv(
        "VLLM_DSV4_DEMAND_SIZED_KV_POOLS", "1"
    ) == "1",
""",
    """    "VLLM_DSV4_DEMAND_SIZED_KV_POOLS": lambda: os.getenv(
        "VLLM_DSV4_DEMAND_SIZED_KV_POOLS", "1"
    ) == "1",
    "VLLM_PREFILL_DECODE_CADENCE": lambda: int(
        os.getenv("VLLM_PREFILL_DECODE_CADENCE", "1")
    ),
""",
)


# The DP scheduler uses a capacity-bound escape to keep ranks synchronized.
# Interactive cadence must remain authoritative even when other requests wait.
replace(
    "v1/core/sched/scheduler.py",
    """import itertools
import time
""",
    """import itertools
import time

import vllm.envs as envs
""",
)

replace(
    "v1/core/sched/scheduler.py",
    """logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
""",
    """logger = init_logger(__name__)


def _get_prefill_token_threshold(
    configured_threshold: int,
    prefill_token_budget: int,
    interactive_cadence: bool,
    num_remaining_prefills: int,
) -> int:
    \"\"\"Return this request's remainder-aware prefill cap.\"\"\"
    if not interactive_cadence:
        return configured_threshold

    remaining = max(1, num_remaining_prefills)
    return (prefill_token_budget + remaining - 1) // remaining


class Scheduler(SchedulerInterface):
""",
)

replace(
    "v1/core/sched/scheduler.py",
    """        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)
""",
    """        interactive_cadence = envs.VLLM_PREFILL_DECODE_CADENCE > 1
        has_active_decode = any(
            not request.is_prefill_chunk for request in self.running
        )
        num_prefill_candidates = min(
            self.max_num_running_reqs,
            sum(request.is_prefill_chunk for request in self.running)
            + len(self.waiting)
            + len(self.skipped_waiting),
        )
        configured_prefill_threshold = (
            self.scheduler_config.long_prefill_token_threshold
        )
        prefill_token_budget = self.max_num_scheduled_tokens
        if has_active_decode and configured_prefill_threshold > 0:
            prefill_token_budget = configured_prefill_threshold
        num_remaining_prefills = num_prefill_candidates
        defer_prefills = (
            throttle_prefills
            and (interactive_cadence or not self.prefill_capacity_bound)
            and has_active_decode
        )
""",
)

replace(
    "v1/core/sched/scheduler.py",
    """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
""",
    """            threshold = configured_prefill_threshold
            if request.is_prefill_chunk:
                threshold = _get_prefill_token_threshold(
                    configured_prefill_threshold,
                    prefill_token_budget,
                    interactive_cadence,
                    num_remaining_prefills,
                )
                if interactive_cadence:
                    num_remaining_prefills = max(0, num_remaining_prefills - 1)
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold
""",
)

replace(
    "v1/core/sched/scheduler.py",
    """                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    break
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and num_new_tokens == 1
                        and (scheduled_running_reqs and not prefill_scheduled)
                    ):
                        num_new_tokens = 1 + self.num_spec_tokens
                        if (
                            num_new_tokens > token_budget
                            or num_computed_tokens + num_new_tokens > self.max_model_len
                        ):
                            # Prefer to not schedule than schedule un-padded here.
                            break
                        pad_spec_decode = True

                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
""",
    """                is_waiting_prefill = (
                    num_computed_tokens < request.num_tokens - 1
                )

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                    if interactive_cadence and is_waiting_prefill:
                        num_remaining_prefills = max(
                            0, num_remaining_prefills - 1
                        )
                elif defer_prefills and is_waiting_prefill:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    break
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and num_new_tokens == 1
                        and (scheduled_running_reqs and not prefill_scheduled)
                    ):
                        num_new_tokens = 1 + self.num_spec_tokens
                        if (
                            num_new_tokens > token_budget
                            or num_computed_tokens + num_new_tokens > self.max_model_len
                        ):
                            # Prefer to not schedule than schedule un-padded here.
                            break
                        pad_spec_decode = True

                    threshold = configured_prefill_threshold
                    if is_waiting_prefill:
                        threshold = _get_prefill_token_threshold(
                            configured_prefill_threshold,
                            prefill_token_budget,
                            interactive_cadence,
                            num_remaining_prefills,
                        )
                        if interactive_cadence:
                            num_remaining_prefills = max(
                                0, num_remaining_prefills - 1
                            )
                    if 0 < threshold < num_new_tokens:
""",
)

replace(
    "v1/core/sched/scheduler.py",
    """            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1
""",
    """            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            if interactive_cadence and request.is_prefill_chunk:
                prefill_token_budget = max(
                    0, prefill_token_budget - num_new_tokens
                )
            req_index += 1
""",
)

replace(
    "v1/core/sched/scheduler.py",
    """                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
""",
    """                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                if interactive_cadence and is_waiting_prefill:
                    prefill_token_budget = max(
                        0, prefill_token_budget - num_new_tokens
                    )
                request.status = RequestStatus.RUNNING
""",
)



# Capture and validate the cadence after the scheduler has been constructed.
replace(
    "v1/engine/core.py",
    """        self.use_spec_decode = vllm_config.speculative_config is not None
""",
    """        self.prefill_decode_cadence = envs.VLLM_PREFILL_DECODE_CADENCE
        if self.prefill_decode_cadence < 1:
            raise ValueError(
                "VLLM_PREFILL_DECODE_CADENCE must be an integer >= 1"
            )
        self._prefill_decode_cadence_step = 0
        if self.prefill_decode_cadence > 1:
            logger.info(
                "Interactive prefill/decode cadence enabled: one prefill-bearing "
                "iteration every %d scheduler iterations while decode is active.",
                self.prefill_decode_cadence,
            )

        self.use_spec_decode = vllm_config.speculative_config is not None
""",
)


# The scheduler already implements safe prefill deferral. Enable that hook on
# ordinary single-engine/TP deployments according to the configured cadence.
# DPEngineCore overrides this method and retains its existing synchronized
# cross-rank cadence.
replace(
    "v1/engine/core.py",
    '''    def _should_throttle_prefills(self) -> bool:
        """Whether to defer new prefills this step (DP prefill balancing).
        Overridden by the DP engine core; never throttles otherwise."""
        return False
''',
    '''    def _should_throttle_prefills(self) -> bool:
        """Defer prefills on decode-priority iterations for non-DP engines.

        Scheduler.schedule() only acts on this signal when decode and prefill
        coexist, so pure-prefill and pure-decode workloads remain unrestricted.
        """
        if self.prefill_decode_cadence <= 1:
            return False

        release_prefill = (
            self._prefill_decode_cadence_step % self.prefill_decode_cadence == 0
        )
        self._prefill_decode_cadence_step += 1
        return not release_prefill
''',
)
