#!/usr/bin/env python3
"""CPU checks for the non-DP interactive prefill/decode cadence."""

import inspect
from types import SimpleNamespace

import vllm.envs as envs
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.scheduler import (
    Scheduler,
    _get_prefill_token_budget,
    _get_prefill_token_threshold,
)
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus


assert "VLLM_PREFILL_DECODE_CADENCE" in envs.environment_variables
schedule_source = inspect.getsource(Scheduler.schedule)
assert "interactive_cadence or not self.prefill_capacity_bound" in schedule_source
assert "prefill_token_threshold" in schedule_source
assert schedule_source.count("prefill_token_budget - num_new_tokens") == 2
assert "is_waiting_prefill" in schedule_source
assert "num_prefill_candidates" in schedule_source
assert "max_prefill_context_tokens" in schedule_source


# Cadence-disabled images retain the configured upstream per-request cap.
assert _get_prefill_token_threshold(512, 8192, False, 3) == 512


def waterfill(
    total_budget: int,
    demands: list[int],
    contexts: list[int] | None = None,
    apply_context_cap: bool = True,
) -> list[int]:
    if contexts is None:
        contexts = [0] * len(demands)
    assert len(contexts) == len(demands)
    available = total_budget
    remaining = len(demands)
    allocations: list[int] = []
    for demand, context in zip(demands, contexts):
        limit = _get_prefill_token_threshold(
            512,
            available,
            True,
            remaining,
            context,
            apply_context_cap,
        )
        scheduled = min(demand, limit)
        allocations.append(scheduled)
        available -= scheduled
        remaining -= 1
    return allocations


# Upstream behavior and an explicit threshold of zero retain the full budget.
assert _get_prefill_token_budget(512, 8192, False, True, 2) == 8192
assert _get_prefill_token_budget(0, 8192, True, True, 2) == 8192

# One pure prefill uses the full budget only while its context is short. Its
# query/context work is then bounded progressively down to the configured cap.
assert _get_prefill_token_budget(512, 8192, True, False, 1) == 8192
assert waterfill(8192, [8192], [0]) == [8192]
assert waterfill(8192, [8192], [65536]) == [8192]
assert waterfill(8192, [8192], [131072]) == [4096]
assert waterfill(8192, [8192], [262144]) == [2048]
assert waterfill(8192, [8192], [524288]) == [1024]
assert waterfill(8192, [8192], [1048576]) == [512]

# Competing pure prefills use the full aggregate budget through 256K context,
# then taper to a 2048-token floor at 1M. The global budget is fair-shared
# without applying the lone-prefill context cap a second time.
assert _get_prefill_token_budget(512, 8192, True, False, 2, 0) == 8192
assert _get_prefill_token_budget(512, 8192, True, False, 2, 262144) == 8192
assert _get_prefill_token_budget(512, 8192, True, False, 2, 524288) == 4096
assert _get_prefill_token_budget(512, 8192, True, False, 2, 850000) == 2527
assert _get_prefill_token_budget(512, 8192, True, False, 2, 1048576) == 2048
assert waterfill(8192, [8192] * 2, apply_context_cap=False) == [4096, 4096]
assert waterfill(
    4096, [8192] * 2, [524288] * 2, apply_context_cap=False
) == [2048, 2048]
assert waterfill(
    2048, [8192] * 2, [1048576] * 2, apply_context_cap=False
) == [1024, 1024]
assert waterfill(
    2527, [8192] * 2, [850000] * 2, apply_context_cap=False
) == [1264, 1263]
assert waterfill(
    2527, [8192] * 4, [850000] * 4, apply_context_cap=False
) == [632, 632, 632, 631]
assert waterfill(
    8192, [100, 8192, 8192], apply_context_cap=False
) == [100, 4046, 4046]

# A mixed iteration shares one global 512-token prefill budget.
assert _get_prefill_token_budget(512, 8192, True, True, 3) == 512
assert waterfill(512, [8192] * 3, apply_context_cap=False) == [171, 171, 170]


class FakeRequest:
    """Minimal request state used to execute Scheduler.schedule()."""

    def __init__(
        self,
        request_id: str,
        num_prompt_tokens: int,
        *,
        num_computed_tokens: int = 0,
        num_tokens: int | None = None,
        status: RequestStatus = RequestStatus.WAITING,
        is_prefill_chunk: bool = False,
        arrival_time: float = 0.0,
    ) -> None:
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self._num_tokens = num_tokens or num_prompt_tokens
        self.num_computed_tokens = num_computed_tokens
        self.status = status
        self.is_prefill_chunk = is_prefill_chunk
        self.arrival_time = arrival_time
        self.priority = 0

        self.num_output_placeholders = 0
        self.max_tokens = 128
        self.next_decode_eligible_step = 0
        self.spec_token_ids: list[int] = []
        self.block_hashes = [request_id]
        self.skip_reading_prefix_cache = False
        self.prefill_stats = None
        self.num_preemptions = int(status == RequestStatus.PREEMPTED)

        self.lora_request = None
        self.mm_features = []
        self.prompt_token_ids = None
        self.prompt_embeds = None
        self.prompt_is_token_ids = None
        self.sampling_params = None
        self.pooling_params = None
        self._all_token_ids: list[int] = []
        self.last_sched_seq = 0

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def num_tokens_with_spec(self) -> int:
        return self._num_tokens + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return max(0, self._num_tokens - self.num_prompt_tokens)

    @property
    def has_encoder_inputs(self) -> bool:
        return False

    @property
    def use_structured_output(self) -> bool:
        return False

    def __lt__(self, other: "FakeRequest") -> bool:
        return (
            self.priority,
            self.arrival_time,
            self.request_id,
        ) < (
            other.priority,
            other.arrival_time,
            other.request_id,
        )


class FakeBlocks:
    blocks = ([],)

    @staticmethod
    def get_block_ids() -> tuple[list[int], ...]:
        return ([],)


class FakeCoordinator:
    def __init__(self, cache_hits: dict[str, int]) -> None:
        self.cache_hits = cache_hits

    def find_longest_cache_hit(
        self,
        block_hashes: list[str],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[int], ...], int]:
        hit = min(self.cache_hits.get(block_hashes[0], 0), max_cache_hit_length)
        return ([],), hit


class FakeKVCacheManager:
    def __init__(self, cache_hits: dict[str, int]) -> None:
        self.enable_caching = True
        self.coordinator = FakeCoordinator(cache_hits)
        self.empty_kv_cache_blocks = FakeBlocks()
        self.log_stats = False

    def new_step_starts(self) -> None:
        pass

    def get_computed_blocks(
        self, request: FakeRequest
    ) -> tuple[FakeBlocks, int]:
        _, hit = self.coordinator.find_longest_cache_hit(
            request.block_hashes,
            request.num_tokens - 1,
        )
        return FakeBlocks(), hit

    def allocate_slots(self, request: FakeRequest, num_new_tokens: int, **kwargs):
        del request, num_new_tokens, kwargs
        return FakeBlocks()

    @staticmethod
    def get_blocks(request_id: str) -> FakeBlocks:
        del request_id
        return FakeBlocks()

    @staticmethod
    def get_num_common_prefix_blocks(request_id: str) -> list[int]:
        del request_id
        return [0]

    @staticmethod
    def take_new_block_ids() -> list[int]:
        return []


class FakeEncoderCacheManager:
    @staticmethod
    def get_freed_mm_hashes() -> list[str]:
        return []


def run_schedule(
    *,
    running: list[FakeRequest] | None = None,
    waiting: list[FakeRequest] | None = None,
    skipped_waiting: list[FakeRequest] | None = None,
    cache_hits: dict[str, int] | None = None,
    max_num_seqs: int = 4,
    cadence: int = 16,
):
    running = running or []
    waiting = waiting or []
    skipped_waiting = skipped_waiting or []
    cache_hits = cache_hits or {}

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.current_step = 0
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.max_num_running_reqs = max_num_seqs
    scheduler.max_model_len = 2 * 1048576
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=512,
        enable_chunked_prefill=True,
    )
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.prefill_capacity_bound = False
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.running = running
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    for request in waiting:
        scheduler.waiting.add_request(request)
    for request in skipped_waiting:
        scheduler.skipped_waiting.add_request(request)

    all_requests = running + waiting + skipped_waiting
    scheduler.requests = {request.request_id: request for request in all_requests}
    scheduler.kv_cache_manager = FakeKVCacheManager(cache_hits)
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[None])
    scheduler.encoder_cache_manager = FakeEncoderCacheManager()
    scheduler.connector = None
    scheduler.connector_prefix_cache_stats = None
    scheduler.ec_connector = None
    scheduler.has_mamba_layers = False
    scheduler.need_mamba_block_aligned_split = False
    scheduler.scheduler_reserve_full_isl = False
    scheduler.is_encoder_decoder = False
    scheduler.lora_config = None
    scheduler.log_stats = False
    scheduler.num_spec_tokens = 0
    scheduler.num_lookahead_tokens = 0
    scheduler.dynamic_sd_lookup = None
    scheduler.use_eagle = False
    scheduler.use_v2_model_runner = False
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.enable_return_routed_experts = False
    scheduler.reset_preempted_req_ids = set()
    scheduler.needs_kv_cache_zeroing = False
    scheduler.defer_block_free = False
    scheduler.sched_step_seq = 0
    scheduler._inflight_prefills = set()
    scheduler._make_cached_request_data = (
        lambda *args, **kwargs: CachedRequestData.make_empty()
    )

    envs.VLLM_PREFILL_DECODE_CADENCE = cadence
    return scheduler.schedule()


# Exercise the real scheduling loop. Fresh uncached prompts have no existing KV
# context, so two 1M-token requests retain the short-context aggregate budget.
fresh_a = FakeRequest("fresh-a", 1048576, arrival_time=1.0)
fresh_b = FakeRequest("fresh-b", 1048576, arrival_time=2.0)
output = run_schedule(
    waiting=[fresh_a, fresh_b],
    max_num_seqs=2,
)
assert output.num_scheduled_tokens == {"fresh-a": 4096, "fresh-b": 4096}

# Running partial prefills and resumed prefills use their actual computed/cache
# context, not their full prompt length.
partial = FakeRequest(
    "partial",
    1048576,
    num_computed_tokens=524288,
    status=RequestStatus.RUNNING,
    is_prefill_chunk=True,
)
output = run_schedule(running=[partial], max_num_seqs=1)
assert output.num_scheduled_tokens == {"partial": 1024}

resumed = FakeRequest(
    "resumed",
    1048576,
    status=RequestStatus.PREEMPTED,
)
output = run_schedule(
    waiting=[resumed],
    cache_hits={"resumed": 524288},
    max_num_seqs=1,
)
assert output.num_scheduled_tokens == {"resumed": 1024}

# Fresh local prefix hits are still visible to the aggregate taper before the
# waiting loop copies the hit count into request.num_computed_tokens.
cached_a = FakeRequest("cached-a", 1048576, arrival_time=1.0)
cached_b = FakeRequest("cached-b", 1048576, arrival_time=2.0)
output = run_schedule(
    waiting=[cached_a, cached_b],
    cache_hits={"cached-a": 524288, "cached-b": 524288},
    max_num_seqs=2,
)
assert output.num_scheduled_tokens == {"cached-a": 2048, "cached-b": 2048}

# A completed asynchronous external-KV receive is represented directly on the
# waiting request and receives the same actual-context cap.
external = FakeRequest(
    "external",
    1048576,
    num_computed_tokens=524288,
    status=RequestStatus.WAITING,
)
output = run_schedule(waiting=[external], max_num_seqs=1)
assert output.num_scheduled_tokens == {"external": 1024}

# Mixed decode/prefill behavior keeps the configured global prefill cap.
decoder = FakeRequest(
    "decode",
    128,
    num_computed_tokens=128,
    num_tokens=129,
    status=RequestStatus.RUNNING,
)
mixed_prefill = FakeRequest(
    "mixed-prefill",
    1048576,
    status=RequestStatus.RUNNING,
    is_prefill_chunk=True,
)
output = run_schedule(
    running=[decoder, mixed_prefill],
    max_num_seqs=2,
)
assert output.num_scheduled_tokens == {"decode": 1, "mixed-prefill": 512}

# Cadence one retains the upstream per-request threshold.
upstream_a = FakeRequest("upstream-a", 1048576, arrival_time=1.0)
upstream_b = FakeRequest("upstream-b", 1048576, arrival_time=2.0)
output = run_schedule(
    waiting=[upstream_a, upstream_b],
    max_num_seqs=2,
    cadence=1,
)
assert output.num_scheduled_tokens == {"upstream-a": 512, "upstream-b": 512}

# Candidate accounting includes both waiting queues exactly as the scheduler
# traverses them; neither queue's fresh prompt is mistaken for existing KV.
waiting_req = FakeRequest("waiting", 1048576, arrival_time=2.0)
skipped_req = FakeRequest("skipped", 1048576, arrival_time=1.0)
output = run_schedule(
    waiting=[waiting_req],
    skipped_waiting=[skipped_req],
    max_num_seqs=2,
)
assert output.num_scheduled_tokens == {"skipped": 4096, "waiting": 4096}


def decisions(cadence: int, count: int) -> list[bool]:
    engine = SimpleNamespace(
        prefill_decode_cadence=cadence,
        _prefill_decode_cadence_step=0,
    )
    return [
        EngineCore._should_throttle_prefills(engine)  # type: ignore[arg-type]
        for _ in range(count)
    ]


# False releases prefill; True asks Scheduler.schedule() to defer it when a
# decoder is active. Upstream behavior remains unchanged at cadence one.
assert decisions(1, 8) == [False] * 8
assert decisions(4, 10) == [
    False,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    False,
    True,
]

print("interactive prefill/decode cadence tests: ok")
