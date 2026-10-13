"""Company PCL algorithm: fixed enqueue profile, tiers, LPM and deadlines.

Edge406 adaptation adds parameter validation; the ranking formula and
waiting-request Fast-Fail semantics are preserved. Distributed decisions
are synchronized by the policy and scheduler adapters.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.managers.schedule_policy import match_prefix_for_req

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

PCL_POLICY = "priority-cost-lpm"


# ======================================================================
# Pure scheduling algorithm
# ======================================================================


class Tier(IntEnum):
    """Scheduling tier within the same priority level.

    Smaller value = scheduled earlier. FAST < NORMAL < BACKGROUND.
    """

    FAST = 0
    NORMAL = 1
    BACKGROUND = 2


@dataclass
class ReqProfile:
    """Static per-request profile, generated once at enqueue time.

    `matched_len` is captured from the local Radix Tree at enqueue and is NOT
    refreshed afterwards (the tier classification is fixed except for aging
    promotion). The native per-round `match_prefix_for_req` still runs each
    scheduling pass to refresh `req.prefix_indices`/`last_node` for admission,
    but the profile's `matched_len`/`recompute_len` stay constant.
    """

    matched_len: int
    context_len: int
    recompute_len: int
    c_miss: float
    priority: int
    slo_deadline: Optional[float]
    tier: Tier = Tier.NORMAL
    aging_promoted: bool = False
    enqueue_time: float = field(default_factory=time.monotonic)

    @classmethod
    def build(
        cls,
        req: "Req",
        tree_cache: "BasePrefixCache",
        c_pre: float,
    ) -> "ReqProfile":
        token_ids = req.origin_input_ids + req.output_ids
        match_result = match_prefix_for_req(
            tree_cache, req, token_ids, include_req=True
        )
        matched_len = min(len(match_result.device_indices), len(token_ids))
        context_len = len(token_ids)
        recompute_len = max(context_len - matched_len, 0)
        priority = req.priority if req.priority is not None else 1
        c_miss = recompute_len * c_pre
        return cls(
            matched_len=matched_len,
            context_len=context_len,
            recompute_len=recompute_len,
            c_miss=c_miss,
            priority=priority,
            slo_deadline=getattr(req, "slo_deadline", None),
        )


@dataclass
class PriorityCostLpmConfig:
    """Tunable parameters for the priority-cost-aware LPM scheduler."""

    T_fast: int = 512
    T_demote: int = 8192
    W_low: float = 0.7
    T_aging: float = 30000.0
    T_failfast: float = 0.0
    c_pre: float = 1.0

    def __post_init__(self):
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in vars(self).values()
        ):
            raise ValueError("PCL parameters must be finite numbers")
        if not 0 <= self.T_fast <= self.T_demote or not 0 <= self.W_low <= 1:
            raise ValueError("PCL requires 0 <= T_fast <= T_demote and 0 <= W_low <= 1")
        if self.T_aging < 0 or self.c_pre < 0:
            raise ValueError("PCL T_aging and c_pre must be nonnegative")

    def classify_tier(self, recompute_len: int) -> Tier:
        if recompute_len <= self.T_fast:
            return Tier.FAST
        if recompute_len > self.T_demote:
            return Tier.BACKGROUND
        return Tier.NORMAL


class PriorityCostLpmScheduler:
    """Core scheduler: priority -> tier -> LPM mixed scheduling.

    Sorting key (ascending): (priority, tier, -matched_len, wait_queue_entry_time).
    BACKGROUND-tier requests naturally sort to the tail of their priority group
    due to tier=2.  They are no longer gated by a congestion condition; instead
    they participate in normal scheduling and are admitted as soon as higher
    tiers drain.  A lightweight KV-occupancy check (W_low) prevents BACKGROUND
    admission when memory is nearly full.  Aging promotes BACKGROUND within its priority group; it does not
    prevent starvation across different priority values.
    """

    def __init__(
        self,
        config: PriorityCostLpmConfig,
        tree_cache: "BasePrefixCache",
        token_to_kv_pool_allocator: "BaseTokenToKVPoolAllocator",
    ):
        self.config = config
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def ensure_profile(self, req: "Req") -> ReqProfile:
        profile = getattr(req, "req_profile", None)
        if profile is None:
            profile = ReqProfile.build(req, self.tree_cache, self.config.c_pre)
            req.req_profile = profile
        return profile

    def kv_occupancy(self) -> float:
        size = self.token_to_kv_pool_allocator.size
        if size <= 0:
            return 0.0
        return 1.0 - (self.token_to_kv_pool_allocator.available_size() / size)

    def apply_aging(self, waiting_queue: List["Req"], now: float) -> None:
        cfg = self.config
        for req in waiting_queue:
            profile = self.ensure_profile(req)
            if profile.tier == Tier.BACKGROUND and not profile.aging_promoted:
                if now - profile.enqueue_time >= (cfg.T_aging / 1000.0):
                    profile.aging_promoted = True
                    profile.tier = Tier.NORMAL

    def check_failfast(self, req: "Req", now: float) -> bool:
        """Return True when the queued request should be fast-failed.

        A waiting-queue request is failed when either:

        * it has waited >= ``T_failfast`` (ms) since enqueue, or
        * its SLO deadline (if provided) has already passed.

        ``T_failfast <= 0`` disables the feature entirely.  ``T_failfast``
        must stay larger than ``T_aging`` (with headroom for one admission +
        prefill), otherwise BACKGROUND requests are failed before aging
        promotion ever gets a chance to schedule them.
        """
        cfg = self.config
        if cfg.T_failfast <= 0:
            return False
        profile = self.ensure_profile(req)
        if now - profile.enqueue_time >= cfg.T_failfast / 1000.0:
            return True
        deadline = profile.slo_deadline
        if deadline is not None and now >= deadline:
            return True
        return False

    def _tier_rank(self, profile: ReqProfile) -> Tier:
        return Tier.NORMAL if profile.aging_promoted else profile.tier

    def calc_priority(
        self,
        waiting_queue: List["Req"],
        running_batch: Optional[object] = None,
    ) -> None:
        now = time.monotonic()
        cfg = self.config

        # Ensure profiles exist and refresh tier classification.
        for req in waiting_queue:
            profile = self.ensure_profile(req)
            if not profile.aging_promoted:
                profile.tier = cfg.classify_tier(profile.recompute_len)

        self.apply_aging(waiting_queue, now)

        if not waiting_queue:
            return

        # Lightweight KV-occupancy gate: skip BACKGROUND-tier requests when KV
        # cache is nearly full to avoid OOM.  This is the only remaining gate
        # for BACKGROUND; the old congestion-based Q_low gate has been removed.
        kv_full = self.kv_occupancy() >= cfg.W_low
        for req in waiting_queue:
            profile = req.req_profile
            tier = self._tier_rank(profile)
            req.skip_background = tier == Tier.BACKGROUND and kv_full

        # Stable multi-key sort: priority asc, tier asc, -matched_len asc,
        # wait_queue_entry_time asc.  BACKGROUND requests with skip_background
        # still sort by tier; the policy adapter temporarily holds them out
        # of the native admission pass and restores them afterwards.
        def sort_key(req: "Req"):
            profile = req.req_profile
            tier = self._tier_rank(profile)
            return (
                profile.priority,
                tier,
                -profile.matched_len,
                req.time_stats.wait_queue_entry_time,
            )

        waiting_queue.sort(key=sort_key)
