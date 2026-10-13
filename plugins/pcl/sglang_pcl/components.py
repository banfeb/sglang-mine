"""Include the retained PCL chunk in native memory checks and load reports."""

from dataclasses import dataclass
from typing import Callable

from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer


@dataclass(kw_only=True)
class PclInvariantChecker(SchedulerInvariantChecker):
    get_pcl_paused_req: Callable

    def _get_total_uncached_sizes(self):
        full, swa = super()._get_total_uncached_sizes()
        req = self.get_pcl_paused_req()
        seen = [self.get_chunked_req()]
        for batch in (self.get_last_batch(), self.get_running_batch()):
            if batch is not None:
                seen.extend(batch.reqs)
        if req is not None and not any(r is req for r in seen) and req.kv.holds_kv:
            allocated = (
                (req.kv.kv_allocated_len + self.page_size - 1) // self.page_size
            ) * self.page_size
            full += allocated - req.kv.cache_protected_len
        return full, swa


@dataclass(kw_only=True, frozen=True)
class PclLoadInquirer(SchedulerLoadInquirer):
    get_pcl_paused_req: Callable

    def _get_num_pending_tokens(self, chunk_deduct=0):
        total = super()._get_num_pending_tokens(chunk_deduct)
        req = self.get_pcl_paused_req()
        if req is not None:
            total += max(0, req.seqlen - len(req.prefix_indices))
        return total

    def get_num_waiting_uncached_tokens(self):
        total = super().get_num_waiting_uncached_tokens()
        req = self.get_pcl_paused_req()
        if req is not None:
            total += max(0, req.seqlen - len(req.prefix_indices))
        return total

    def get_loads(self):
        snapshot = super().get_loads()
        req = self.get_pcl_paused_req()
        if req is not None:
            remaining = max(0, req.seqlen - len(req.prefix_indices))
            snapshot.num_waiting_reqs += 1
            snapshot.num_total_tokens += remaining
            snapshot.num_active_tokens += remaining
            snapshot.queues.paused += 1
        return snapshot
