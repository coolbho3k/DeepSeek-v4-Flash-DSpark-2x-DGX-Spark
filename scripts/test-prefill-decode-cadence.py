#!/usr/bin/env python3
"""CPU checks for the non-DP interactive prefill/decode cadence."""

import inspect
from types import SimpleNamespace

import vllm.envs as envs
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore


assert "VLLM_PREFILL_DECODE_CADENCE" in envs.environment_variables
schedule_source = inspect.getsource(Scheduler.schedule)
assert "interactive_cadence or not self.prefill_capacity_bound" in schedule_source


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
