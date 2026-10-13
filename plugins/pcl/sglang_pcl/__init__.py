"""Company PCL plugin adapted to Edge406; enable with --schedule-policy priority-cost-lpm.

Replaces Scheduler, SchedulePolicy and the msgspec ServerArgs record. Native
admission stays in SGLang; no copied admission method is shipped.
"""

from __future__ import annotations

import logging
import importlib.util

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sglang_pcl.algorithm import PCL_POLICY
from sglang_pcl.policy import PclSchedulePolicy
from sglang_pcl.scheduler import PclScheduler
from sglang_pcl.server_args_ext import check_server_args_pcl, extend_server_args

logger = logging.getLogger(__name__)


def _sglang_already_has_pcl() -> bool:
    """Detect an sglang tree that already carries the in-tree PCL patch.

    If the active sglang installation is one of the PCL-patched trees
    (e.g. the editable install pointing at a modified sglang-0.5.17),
    installing the plugin hooks on top would double-apply PCL. In that
    case the plugin stays inert and logs a warning.
    """
    try:
        from sglang.srt.server_args import ServerArgs

        if "pcl_T_fast" in getattr(ServerArgs, "__dataclass_fields__", {}):
            return True
    except Exception:
        pass
    for mod_name in (
        "sglang.srt.managers.scheduler_pcl_mixin",
        "sglang.srt.managers.scheduler_priority_cost_lpm",
    ):
        if importlib.util.find_spec(mod_name) is not None:
            return True
    try:
        import sglang.srt.managers.scheduler as _sched

        if hasattr(_sched, "SchedulerPclMixin") or hasattr(
            _sched, "PriorityCostLpmScheduler"
        ):
            return True
    except Exception:
        pass
    return False


def plugin_entry():
    if _sglang_already_has_pcl():
        logger.warning(
            "sglang-pcl: the active sglang installation already contains an "
            "in-tree PCL implementation; the plugin stays inert to avoid "
            "double-applying PCL. Use a vanilla sglang tree to activate the "
            "plugin."
        )
        return

    # 1) ServerArgs: msgspec subclass + CLI hook (before the CLI
    #    parser is built; load_plugins() runs before prepare_server_args()).
    extend_server_args()

    # 2) Allow --enable-priority-scheduling with the PCL policy.
    #    (AROUND hook: see server_args_ext.check_server_args_pcl for why
    #    REPLACE would infinitely recurse.)
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.check_server_args",
        check_server_args_pcl,
        HookType.AROUND,
    )

    # 3) PCL-aware SchedulePolicy (accepts "priority-cost-lpm").
    HookRegistry.register(
        "sglang.srt.managers.schedule_policy.SchedulePolicy",
        PclSchedulePolicy,
        HookType.REPLACE,
    )

    # 4) PCL-aware Scheduler.
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler",
        PclScheduler,
        HookType.REPLACE,
    )

    logger.info(
        "sglang-pcl: registered PCL plugin hooks (policy=%s). "
        "Use --schedule-policy %s to enable.",
        PCL_POLICY,
        PCL_POLICY,
    )
