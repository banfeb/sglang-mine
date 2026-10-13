"""PCL-aware SchedulePolicy.

Replaces ``sglang.srt.managers.schedule_policy.SchedulePolicy`` via the plugin
hook framework (HookType.REPLACE).  For every non-PCL policy the class
delegates 1:1 to the vanilla implementation.  For ``priority-cost-lpm`` it
accepts the policy string and dispatches ``calc_priority`` to the
``PriorityCostLpmScheduler``.

This class also owns the BACKGROUND hold-out: after the PCL sort,
requests flagged ``skip_background`` (KV-cache occupancy gate) leave the
waiting queue for the duration of one admission pass and are restored by
``PclScheduler`` afterwards.  This reproduces the in-tree modification's
``continue``-skip in the admission loop of ``_get_new_batch_prefill_raw``
without copying that method.  The hold-out is gated by an *armed* flag that
only ``PclScheduler.get_new_batch_prefill`` sets, so the many other callers
of ``policy.calc_priority`` (disagg decode's ``get_new_prebuilt_batch``,
tests, ...) are unaffected — exactly matching the in-tree reach of the skip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
import torch.distributed as dist

from sglang.srt.managers.schedule_policy import SchedulePolicy, match_prefix_for_req
from sglang.srt.runtime_context import get_disagg

from sglang_pcl.algorithm import PCL_POLICY


def broadcast_decision(value, group):
    if dist.is_initialized() and dist.get_world_size(group) > 1:
        values = [value]
        dist.broadcast_object_list(
            values, src=dist.get_global_rank(group, 0), group=group
        )
        return values[0]
    return value


if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class PclSchedulePolicy(SchedulePolicy):
    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
        pcl_scheduler: Optional[object] = None,
    ):
        self.pcl_scheduler = pcl_scheduler
        # BACKGROUND hold-out state; see module docstring.
        self._held_background: List[Req] = []
        self._holdout_armed: bool = False
        self._priority_ceiling = None
        self.cpu_group = None
        super().__init__(
            policy,
            tree_cache,
            enable_hierarchical_cache,
            enable_priority_scheduling,
            schedule_low_priority_values_first,
        )

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> SchedulePolicy.Policy:
        if policy == PCL_POLICY:
            # Keep the raw string as the policy identity. Vanilla enums do not
            # know this policy; every PCL-aware branch compares against
            # PCL_POLICY instead.
            return PCL_POLICY
        return super()._validate_and_adjust_policy(policy, tree_cache)

    def calc_priority(
        self,
        waiting_queue: List[Req],
        running_batch: Optional[ScheduleBatch] = None,
        processed_tokens: int = 0,
    ) -> None:
        if self.policy == PCL_POLICY:
            # Same prefix-match population vanilla runs for cache-agnostic
            # policies before the policy dispatch (vanilla lines: the
            # `not isinstance(policy, CacheAwarePolicy) and ...` block). The
            # in-tree PCL branch sat right after this block.
            if (
                self.tree_cache.supports_fast_match_prefix()
                and get_disagg().disaggregation_mode != "decode"
            ):
                for r in waiting_queue:
                    match_prefix_for_req(self.tree_cache, r, include_req=True)

            if self.pcl_scheduler is not None:
                for req in waiting_queue:
                    self.pcl_scheduler.ensure_profile(req)
                decision = None
                if not dist.is_initialized() or dist.get_rank(self.cpu_group) == 0:
                    self.pcl_scheduler.calc_priority(waiting_queue, running_batch)
                    decision = [
                        (
                            r.rid,
                            r.req_profile.tier,
                            r.req_profile.aging_promoted,
                            r.skip_background,
                        )
                        for r in waiting_queue
                    ]
                decision = broadcast_decision(decision, self.cpu_group)
                by_rid = {r.rid: r for r in waiting_queue}
                if len(by_rid) != len(waiting_queue) or set(by_rid) != {
                    d[0] for d in decision
                }:
                    raise RuntimeError(
                        "PCL TP queues disagree or contain duplicate request IDs"
                    )
                waiting_queue[:] = [by_rid[rid] for rid, *_ in decision]
                for rid, tier, promoted, skip in decision:
                    req = by_rid[rid]
                    req.req_profile.tier = tier
                    req.req_profile.aging_promoted = promoted
                    req.skip_background = skip
                self._hold_out_background(waiting_queue)
            return
        return super().calc_priority(
            waiting_queue, running_batch, processed_tokens=processed_tokens
        )

    # ------------------------------------------------------------------
    # BACKGROUND hold-out (in-tree equivalent: admission-loop skip)
    # ------------------------------------------------------------------

    def arm_holdout(self, priority_ceiling=None) -> None:
        """Allow the next calc_priority pass to hold flagged BACKGROUND
        requests out of the queue.  Set by PclScheduler around the admission
        pass; disarm/restore always follows in the same scheduler round."""
        self._holdout_armed = True
        self._priority_ceiling = priority_ceiling

    def restore_held_background(self, waiting_queue: List[Req]) -> None:
        """Disarm and put held requests back into the waiting queue.

        Safe to call unconditionally (idempotent).  Held requests are
        re-sorted by the next calc_priority pass, so their restored position
        is irrelevant.
        """
        self._holdout_armed = False
        self._priority_ceiling = None
        if self._held_background:
            waiting_queue.extend(self._held_background)
            self._held_background.clear()

    def _hold_out_background(self, waiting_queue: List[Req]) -> None:
        """Remove flagged BACKGROUND requests from the queue so the admission
        loop never sees them.  No-op unless armed: the disagg-decode path
        (``get_new_prebuilt_batch``) also calls calc_priority but, like the
        in-tree modification, must not skip or lose any request there."""
        if not self._holdout_armed:
            return
        held = [
            r
            for r in waiting_queue
            if getattr(r, "skip_background", False)
            or (
                self._priority_ceiling is not None
                and r.priority >= self._priority_ceiling
            )
        ]
        if not held:
            return
        self._held_background.extend(held)
        held_ids = {id(r) for r in held}
        waiting_queue[:] = [r for r in waiting_queue if id(r) not in held_ids]
