"""Native PCL CLI fields for Edge406's msgspec ServerArgs record."""

from __future__ import annotations

import dataclasses
from typing import get_type_hints

import msgspec
from sglang.srt.arg_groups.arg_utils import A, NS
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.server_args import ServerArgs
from sglang_pcl.algorithm import PCL_POLICY, PriorityCostLpmConfig


@dataclasses.dataclass
class _PclServerArgsExtension:
    """Fields injected into ServerArgs; defaults/help identical to the
    in-tree PCL modification of sglang 0.5.17."""

    pcl_T_fast: A[
        int,
        "Priority-cost-lpm: instant-tier threshold. Requests with recompute_len <= this enter the instant tier (tokens).",
        NS("schedule"),
    ] = PriorityCostLpmConfig.T_fast
    pcl_T_demote: A[
        int,
        "Priority-cost-lpm: background-tier demotion threshold. Requests with recompute_len > this are demoted to the background tier (tokens).",
        NS("schedule"),
    ] = PriorityCostLpmConfig.T_demote
    pcl_W_low: A[
        float,
        "Priority-cost-lpm: KV-cache occupancy watermark. Background-tier requests are skipped when occupancy >= this value.",
        NS("schedule"),
    ] = PriorityCostLpmConfig.W_low
    pcl_T_aging: A[
        float,
        "Priority-cost-lpm: aging promotion threshold. Background-tier requests waiting longer than this (ms) are permanently promoted to the normal tier.",
        NS("schedule"),
    ] = PriorityCostLpmConfig.T_aging
    pcl_T_failfast: A[
        float,
        "Priority-cost-lpm: fast-fail threshold (ms). Waiting-queue requests queued longer than this, or whose SLO deadline has passed, are aborted with a 503. 0 disables. Must stay larger than pcl_T_aging with headroom, otherwise BACKGROUND requests are failed before aging promotion can schedule them.",
        NS("schedule"),
    ] = PriorityCostLpmConfig.T_failfast
    pcl_c_pre: A[
        float,
        "Priority-cost-lpm: per-token prefill cost coefficient for the C_miss profile field.",
        NS("schedule"),
    ] = PriorityCostLpmConfig.c_pre


def _make_server_args():
    """Build on import so a fresh spawned process can unpickle the class."""
    hints = get_type_hints(_PclServerArgsExtension, include_extras=True)
    fields = [
        (f.name, hints[f.name], f.default)
        for f in dataclasses.fields(_PclServerArgsExtension)
    ]
    return msgspec.defstruct(
        "PclServerArgs",
        fields,
        bases=(ServerArgs,),
        namespace={
            "__module__": __name__,
            "_NS_BY_FIELD": {
                **ServerArgs._NS_BY_FIELD,
                **{name: "schedule" for name, _, _ in fields},
            },
        },
    )


PclServerArgs = _make_server_args()


def extend_server_args():
    """Replace both the module binding and Engine's captured class attribute."""
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs", PclServerArgs, HookType.REPLACE
    )
    HookRegistry.register(
        "sglang.srt.entrypoints.engine.Engine.server_args_class",
        PclServerArgs,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.add_cli_args",
        add_cli_args_pcl,
        HookType.AROUND,
    )


def add_cli_args_pcl(original_fn, parser):
    original_fn(parser)
    action = next(a for a in parser._actions if a.dest == "schedule_policy")
    if PCL_POLICY not in action.choices:
        action.choices = [*action.choices, PCL_POLICY]


def check_server_args_pcl(original_fn, self):
    """Whitelist compatibility through the resolved view, not frozen inputs.

    The native method only tests the policy in its fcfs/lof whitelist. All
    other validation still executes. Remove our temporary declaration even
    on failure, retaining any unrelated declarations the validator emits.
    """
    if self.schedule_policy != PCL_POLICY:
        return original_fn(self)
    marker = ("pcl.validation", {"schedule_policy": "fcfs"})
    self._resolved_overrides = [*getattr(self, "_resolved_overrides", []), marker]
    try:
        return original_fn(self)
    finally:
        self._resolved_overrides = [
            item for item in self._resolved_overrides if item is not marker
        ]
