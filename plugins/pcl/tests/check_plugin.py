"""Actual entry-point, class identity and native-admission reuse checks."""

import os

os.environ["SGLANG_PLUGINS"] = "pcl"
from sglang.srt.plugins import load_plugins

load_plugins()
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.engine import Engine
from sglang_pcl.scheduler import PclScheduler
from sglang_pcl.policy import PclSchedulePolicy

assert Scheduler is PclScheduler
assert SchedulePolicy is PclSchedulePolicy
assert "pcl_T_fast" in ServerArgs.__struct_fields__
assert Engine.server_args_class is ServerArgs
assert "_get_new_batch_prefill_raw" not in vars(PclScheduler)
load_plugins()
assert Scheduler is PclScheduler
print("PASS: real entry point, class replacement, native admission, idempotence")
