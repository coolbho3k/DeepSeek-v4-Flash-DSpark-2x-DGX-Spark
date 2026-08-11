#!/usr/bin/env python3
"""CPU checks for the non-DP interactive prefill/decode cadence."""

import inspect
from types import SimpleNamespace

import vllm.envs as envs
from vllm.v1.core.sched.scheduler import (
    Scheduler,
    _get_prefill_token_budget,
    _get_prefill_token_threshold,
)
from vllm.v1.engine.core import EngineCore


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
