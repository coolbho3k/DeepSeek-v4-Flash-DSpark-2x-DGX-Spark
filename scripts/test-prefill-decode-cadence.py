#!/usr/bin/env python3
"""CPU checks for the non-DP interactive prefill/decode cadence."""

import inspect
from types import SimpleNamespace

import vllm.envs as envs
from vllm.v1.core.sched.scheduler import (
    Scheduler,
    _get_prefill_token_threshold,
)
from vllm.v1.engine.core import EngineCore


assert "VLLM_PREFILL_DECODE_CADENCE" in envs.environment_variables
schedule_source = inspect.getsource(Scheduler.schedule)
assert "interactive_cadence or not self.prefill_capacity_bound" in schedule_source
assert "prefill_token_threshold" in schedule_source


# Cadence-disabled images retain the configured upstream cap. Mixed workloads
# retain the latency-oriented cap, while pure-prefill workloads fairly divide
# the full token budget across their active streams.
assert _get_prefill_token_threshold(512, 8192, False, False, 1) == 512
assert _get_prefill_token_threshold(512, 8192, True, True, 1) == 512
assert _get_prefill_token_threshold(512, 8192, True, False, 1) == 8192
assert _get_prefill_token_threshold(512, 8192, True, False, 2) == 4096
assert _get_prefill_token_threshold(512, 8192, True, False, 4) == 2048
assert _get_prefill_token_threshold(0, 8192, True, False, 1) == 0


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
