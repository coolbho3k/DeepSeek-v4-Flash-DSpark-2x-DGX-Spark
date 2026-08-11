"""Add decode-first cadence control to the non-DP vLLM EngineCore.

VLLM_PREFILL_DECODE_CADENCE=1 preserves upstream scheduling. Values greater
than one admit prefill compute once per N scheduler iterations while at least
one decode request is running. Pure-prefill batches use a larger aggregate
budget that tapers with existing context, preserving throughput while bounding
the duration of a single model step. Once decode is active, competing prefills
share the configured long-prefill threshold.
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


_LONE_PREFILL_WORK_TARGET_CONTEXT_TOKENS = 1 << 16
_CONTENDED_PREFILL_WORK_TARGET_CONTEXT_TOKENS = 1 << 18
_CONTENDED_PREFILL_MIN_BUDGET_MULTIPLIER = 4


def _get_prefill_token_budget(
    configured_threshold: int,
    max_scheduled_tokens: int,
    interactive_cadence: bool,
    has_active_decode: bool,
    num_prefill_candidates: int,
    max_prefill_context_tokens: int = 0,
) -> int:
    \"\"\"Return the global prefill budget for this scheduler iteration.\"\"\"
    if not interactive_cadence or configured_threshold <= 0:
        return max_scheduled_tokens
    if has_active_decode:
        return min(configured_threshold, max_scheduled_tokens)
    if num_prefill_candidates <= 1:
        return max_scheduled_tokens

    # Several pure prefills can use a large batch while their contexts are
    # short. Taper the aggregate query-token budget as attention work grows,
    # but retain four configured-threshold chunks of forward progress. With
    # the default 8192/512 profile this yields 8192 through 256K context, then
    # 4096 at 512K and 2048 at 1M.
    context_tokens = max(
        _CONTENDED_PREFILL_WORK_TARGET_CONTEXT_TOKENS,
        max_prefill_context_tokens,
    )
    context_budget = (
        max_scheduled_tokens * _CONTENDED_PREFILL_WORK_TARGET_CONTEXT_TOKENS
        + context_tokens
        - 1
    ) // context_tokens
    min_budget = min(
        max_scheduled_tokens,
        configured_threshold * _CONTENDED_PREFILL_MIN_BUDGET_MULTIPLIER,
    )
    return min(max_scheduled_tokens, max(min_budget, context_budget))


def _get_prefill_context_tokens(request, kv_cache_manager, has_mamba_connector):
    \"\"\"Return KV context already computed or locally available for a prefill.\"\"\"
    if (
        request.num_computed_tokens > 0
        or not kv_cache_manager.enable_caching
        or request.skip_reading_prefix_cache
    ):
        return request.num_computed_tokens

    # Fresh and preempted requests do not receive their local prefix-cache hit
    # count until the waiting loop below. Probe the coordinator directly so the
    # aggregate work budget sees actual available KV without recording the same
    # prefix-cache lookup twice or pretending the uncomputed prompt is context.
    max_cache_hit_length = request.num_tokens - 1
    if has_mamba_connector:
        _, per_group_hits = (
            kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
                request.block_hashes,
                max_cache_hit_length,
            )
        )
        return max(per_group_hits, default=0)

    _, num_computed_tokens = kv_cache_manager.coordinator.find_longest_cache_hit(
        request.block_hashes,
        max_cache_hit_length,
    )
    return num_computed_tokens


def _get_prefill_token_threshold(
    configured_threshold: int,
    prefill_token_budget: int,
    interactive_cadence: bool,
    num_remaining_prefills: int,
    num_computed_tokens: int = 0,
    apply_context_cap: bool = True,
) -> int:
    \"\"\"Return this request's fair, context-aware prefill cap.\"\"\"
    if not interactive_cadence:
        return configured_threshold

    remaining = max(1, num_remaining_prefills)
    fair_share = (prefill_token_budget + remaining - 1) // remaining
    if (
        not apply_context_cap
        or configured_threshold <= 0
        or num_computed_tokens <= _LONE_PREFILL_WORK_TARGET_CONTEXT_TOKENS
    ):
        return fair_share

    # Bound roughly query_tokens * existing_context_tokens for a lone prefill.
    # At the default 8192/512 budgets this yields 8192 through 64K context,
    # then 4096/2048/1024/512 at 128K/256K/512K/1M respectively.
    context_cap = (
        prefill_token_budget * _LONE_PREFILL_WORK_TARGET_CONTEXT_TOKENS
        + num_computed_tokens
        - 1
    ) // num_computed_tokens
    context_cap = max(configured_threshold, context_cap)
    return min(fair_share, context_cap)


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
        running_prefill_contexts = [
            request.num_computed_tokens
            for request in self.running
            if request.is_prefill_chunk
        ]
        waiting_slots = max(0, self.max_num_running_reqs - len(self.running))
        waiting_prefill_contexts = (
            [
                _get_prefill_context_tokens(
                    request,
                    self.kv_cache_manager,
                    self.connector is not None and self.has_mamba_layers,
                )
                for request_queue in (self.waiting, self.skipped_waiting)
                for request in itertools.islice(request_queue, waiting_slots)
            ]
            if interactive_cadence and not has_active_decode
            else []
        )
        max_prefill_context_tokens = max(
            running_prefill_contexts + waiting_prefill_contexts,
            default=0,
        )
        configured_prefill_threshold = (
            self.scheduler_config.long_prefill_token_threshold
        )
        prefill_token_budget = _get_prefill_token_budget(
            configured_prefill_threshold,
            self.max_num_scheduled_tokens,
            interactive_cadence,
            has_active_decode,
            num_prefill_candidates,
            max_prefill_context_tokens,
        )
        apply_lone_prefill_context_cap = (
            not has_active_decode and num_prefill_candidates <= 1
        )
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
                    request.num_computed_tokens,
                    apply_lone_prefill_context_cap,
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
                            num_computed_tokens,
                            apply_lone_prefill_context_cap,
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
