"""Company PCL scheduler adapted to Edge406 50ec9702d0dc.

Keeps native admission via super() and a single persistent paused chunk.
"""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from dataclasses import fields
import torch.distributed as dist
from typing import TYPE_CHECKING, Optional

from sglang.srt.disaggregation.utils import DisaggregationMode, prepare_abort
from sglang.srt.managers.scheduler import (
    AbortReq,
    Scheduler,
    _make_abort_req,
    retract_all,
)
from sglang.srt.runtime_context import get_parallel, get_disagg
from sglang.srt.environ import envs
from sglang_pcl.policy import broadcast_decision
from sglang_pcl.components import PclInvariantChecker, PclLoadInquirer
from sglang.srt.mem_cache.common import release_kv_cache

from sglang_pcl.algorithm import (
    PCL_POLICY,
    PriorityCostLpmConfig,
    PriorityCostLpmScheduler,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)


class PclScheduler(Scheduler):
    """Vanilla Scheduler with the priority-cost-lpm (PCL) scheduling policy.

    All PCL state is inert unless ``schedule_policy == "priority-cost-lpm"``:
    ``pcl_scheduler`` stays ``None`` and every override below short-circuits
    to a ``super()`` pass-through, so the class behaves exactly like the
    upstream Scheduler.
    """

    # Populated by init_schedule_policy() when the PCL policy is active.
    pcl_scheduler: Optional[PriorityCostLpmScheduler] = None
    # In-flight chunked request paused by a strictly higher-priority waiting
    # request (see _pcl_pause_or_resume_chunked_req).  At most one request
    # may be paused at a time (no nested preemption).
    _paused_chunked_req: Optional[Req] = None
    # monotonic timestamp of when _paused_chunked_req was paused; anchor for
    # the fast-fail of paused requests.
    _paused_chunked_req_since: Optional[float] = None

    # ------------------------------------------------------------------
    # Scheduling-policy setup
    # ------------------------------------------------------------------

    def init_schedule_policy(self):
        super().init_schedule_policy()
        if self.schedule_policy == PCL_POLICY:
            self._init_pcl_scheduler()

    def _init_pcl_scheduler(self):
        """Build the PCL scheduler from the pcl_* server args and attach it
        to the schedule policy, which dispatches calc_priority to it."""
        pcl_cfg = PriorityCostLpmConfig(
            T_fast=self.server_args.pcl_T_fast,
            T_demote=self.server_args.pcl_T_demote,
            W_low=self.server_args.pcl_W_low,
            T_aging=self.server_args.pcl_T_aging,
            T_failfast=self.server_args.pcl_T_failfast,
            c_pre=self.server_args.pcl_c_pre,
        )
        self.pcl_scheduler = PriorityCostLpmScheduler(
            pcl_cfg,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
        )
        self.policy.pcl_scheduler = self.pcl_scheduler
        self.policy.cpu_group = self.tp_cpu_group
        self._paused_chunked_req = None
        self._paused_chunked_req_since = None
        unsupported = []
        if self.enable_overlap:
            unsupported.append("overlap: use --disable-overlap-schedule")
        if get_parallel().pp_size != 1 or self.ps.dp_size != 1:
            unsupported.append("PP/DP: use PP=1 and DP=1")
        if (
            get_parallel().attn_cp_size != 1
            or get_parallel().enable_dp_attention
            or get_parallel().dcp_enabled
        ):
            unsupported.append("CP/DP attention/DCP")
        if get_disagg().disaggregation_mode != "null" or self.enable_pdmux:
            unsupported.append("PD/PDMUX")
        if not self.spec_algorithm.is_none() or self.dllm_config is not None:
            unsupported.append("speculative/diffusion decoding")
        if (
            self.is_hybrid_swa
            or self.is_hybrid_ssm
            or self.enable_hierarchical_cache
            or self.enable_unified_cache_external_linker
            or self.enable_hisparse
            or getattr(self.tree_cache, "disable", False)
            or self.tree_cache.is_chunk_cache()
        ):
            unsupported.append("nonstandard or disabled radix cache")
        if self.is_mixed_chunk:
            unsupported.append("mixed chunk prefill")
        if (
            not self.enable_priority_scheduling
            or not self.schedule_low_priority_values_first
        ):
            unsupported.append("enable priority scheduling and lower-values-first")
        if self.enable_priority_preemption:
            unsupported.append(
                "native decode preemption: use --disable-priority-preemption"
            )
        if unsupported:
            raise ValueError("PCL unsupported configuration: " + "; ".join(unsupported))

        if 0 < pcl_cfg.T_failfast <= pcl_cfg.T_aging:
            logger.warning(
                f"pcl_T_failfast ({pcl_cfg.T_failfast}ms) should be larger than "
                f"pcl_T_aging ({pcl_cfg.T_aging}ms) with headroom for one admission "
                "+ prefill; otherwise BACKGROUND requests are fast-failed before "
                "aging promotion can ever schedule them."
            )

    # ------------------------------------------------------------------
    # Chunked-request pause / resume (priority preemption of prefill)
    # ------------------------------------------------------------------

    @staticmethod
    def _pcl_req_priority(req: Req) -> int:
        return req.priority if req.priority is not None else 1

    def _pcl_top_waiting_priority(self) -> int:
        return min(
            (
                self._pcl_req_priority(r)
                for r in self.waiting_queue
                if not getattr(r, "skip_background", False)
            ),
            default=float("inf"),
        )

    def _pcl_pause_or_resume_chunked_req(self):
        """Pause the in-flight chunked request when a strictly higher-priority
        request is waiting; resume it only once no strictly higher-priority
        request remains — run-to-completion preemption with no round-robin:
        the preempting side is served until the priority inversion drains.

        At most one request may be paused at a time: while a pause is
        outstanding, the in-flight chunked request is never paused again,
        so a preemptor cannot itself be preempted (no nesting).  Indefinite
        pausing is bounded by the fast-fail pass
        (_pcl_failfast_expired_reqs), which aborts a request paused
        longer than pcl_T_failfast or past its SLO deadline.
        """
        if self.pcl_scheduler is None:
            return

        if self.chunked_req is not None:
            if (
                self._paused_chunked_req is None
                and self.waiting_queue
                and self._pcl_top_waiting_priority()
                < self._pcl_req_priority(self.chunked_req)
            ):
                self._paused_chunked_req = self.chunked_req
                self._paused_chunked_req_since = time.monotonic()
                self.chunked_req = None
                logger.info("PCL paused chunk %s", self._paused_chunked_req.rid)
        elif self._paused_chunked_req is not None:
            paused = self._paused_chunked_req
            if paused.finished():
                # Its last chunk completed while paused (possible under
                # overlap scheduling); nothing to resume, just drop the
                # marker so the finished request is not re-admitted.
                self._paused_chunked_req = None
                self._paused_chunked_req_since = None
            elif not self.waiting_queue or (
                self._pcl_top_waiting_priority() >= self._pcl_req_priority(paused)
            ):
                self.chunked_req = paused
                logger.info("PCL resumed chunk %s", paused.rid)
                self._paused_chunked_req = None
                self._paused_chunked_req_since = None

    # ------------------------------------------------------------------
    # Fast-fail of expired waiting requests
    # ------------------------------------------------------------------

    def _pcl_failfast_expired_reqs(self):
        # Evaluate clocks only on the leader; every TP rank retires the same
        # requests before constructing the next batch.
        decision = None
        if not dist.is_initialized() or dist.get_rank(self.tp_cpu_group) == 0:
            now = time.monotonic()
            failed = [
                r.rid
                for r in self.waiting_queue
                if self.pcl_scheduler.check_failfast(r, now)
            ]
            paused = self._paused_chunked_req
            expired = False
            cfg = self.pcl_scheduler.config
            if paused is not None and cfg.T_failfast > 0:
                deadline = self.pcl_scheduler.ensure_profile(paused).slo_deadline
                expired = (
                    now - self._paused_chunked_req_since >= cfg.T_failfast / 1000
                    or (deadline is not None and now >= deadline)
                )
            decision = (failed, expired)
        failed, expired = broadcast_decision(decision, self.tp_cpu_group)
        if expired:
            self._pcl_abort_req(
                self._paused_chunked_req,
                "Fast-failed: paused chunk exceeded pcl_T_failfast or SLO deadline.",
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"reason": "FastFail"},
            )
        failed_ids = set(failed)
        victims = [r for r in self.waiting_queue if r.rid in failed_ids]
        self.waiting_queue[:] = [
            r for r in self.waiting_queue if r.rid not in failed_ids
        ]
        for req in victims:
            self._pcl_abort_req(
                req,
                "Fast-failed: queued longer than pcl_T_failfast or SLO deadline passed.",
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"reason": "FastFail"},
            )

    def _pcl_abort_req(self, req, message="Aborted", status_code=None, abort_info=None):
        # Synchronous execution only: previous GPU work has completed at this
        # point. Finish the cache handle and release the retained slot and KV.
        prepare_abort(req, message, status_code=status_code)
        req.time_stats.trace_ctx.abort(abort_info=abort_info or {"reason": "Aborted"})
        self._release_aborted_request(req)
        self.beam_coordinator.retire_group(req)
        if req is self._paused_chunked_req:
            req.to_finish = None
            release_kv_cache(req, self.tree_cache, is_insert=False)
            self._paused_chunked_req = None
            self._paused_chunked_req_since = None
            if self._pending_chunked_abort_req is req:
                self._pending_chunked_abort_req = None
        self.ipc_channels.send_to_tokenizer.send_output(
            _make_abort_req(req, finished_reason=req.finished_reason.to_json()), req
        )
        logger.info("PCL retired request %s: %s", req.rid, message)

    # ------------------------------------------------------------------
    # Admission-pass prework
    # ------------------------------------------------------------------

    def _pcl_prefill_admission_prework(self) -> None:
        """Retire expired requests, refresh gates, then decide chunk ownership.

        Edge406 currently supports synchronous ordinary TP. Refresh gates
        before preemption so gated work cannot strand the paused request.
        """
        if (
            self.pcl_scheduler is not None
            and self.disaggregation_mode == DisaggregationMode.NULL
            and not self.enable_pdmux
        ):
            self._pcl_failfast_expired_reqs()
            self.policy.calc_priority(self.waiting_queue)
            self._pcl_pause_or_resume_chunked_req()

    # ------------------------------------------------------------------
    # Thin vanilla-method overrides
    # ------------------------------------------------------------------

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        super()._add_request_to_queue(req, is_retracted)
        # PCL: build the per-request cost profile at enqueue time.  The tail
        # check fires only when the vanilla method actually appended the
        # request to the waiting queue (NULL disaggregation branch, after the
        # priority-validation and queued-limit early returns); in the
        # PREFILL/DECODE branches the request never reaches the waiting
        # queue, matching the in-tree hook placement.
        if (
            self.pcl_scheduler is not None
            and self.waiting_queue
            and self.waiting_queue[-1] is req
        ):
            self.pcl_scheduler.ensure_profile(req)

    def process_pending_chunked_abort(self) -> None:
        # PCL: a paused chunked request is not self.chunked_req; abort it
        # here instead of falling through to the running-batch path.
        req = self._pending_chunked_abort_req
        if (
            req is not None
            and self.pcl_scheduler is not None
            and self._paused_chunked_req is req
        ):
            self._pcl_abort_req(req)
            return
        super().process_pending_chunked_abort()

    def get_new_batch_prefill(self, running_batch):
        if self.pcl_scheduler is None:
            return super().get_new_batch_prefill(running_batch)
        # Include newly-ready grammars in the preemption decision.
        if self.grammar_manager.has_waiting_grammars():
            for req in self.grammar_manager.get_ready_grammar_requests():
                self._add_request_to_queue(req)
        self._pcl_prefill_admission_prework()
        paused = self._paused_chunked_req
        self.policy.arm_holdout(
            self._pcl_req_priority(paused) if paused is not None else None
        )
        try:
            plan = super().get_new_batch_prefill(running_batch)
        finally:
            self.policy.restore_held_background(self.waiting_queue)
        if (
            plan.batch_to_run is None
            and self.chunked_req is None
            and paused is not None
        ):
            # Higher-priority work could not be admitted (e.g. no free slot).
            # Do not strand A behind an impossible preemption attempt.
            self.chunked_req = paused
            self._paused_chunked_req = None
            self._paused_chunked_req_since = None
            self.policy.arm_holdout()
            try:
                return super().get_new_batch_prefill(plan.running_batch)
            finally:
                self.policy.restore_held_background(self.waiting_queue)
        return plan

    def _set_or_validate_priority(self, req):
        if self.pcl_scheduler is not None and req.priority is None:
            req.priority = 1
        return super()._set_or_validate_priority(req)

    def abort_request(self, recv_req):
        paused = self._paused_chunked_req
        if paused is not None and (
            recv_req.abort_all or paused.rid.startswith(recv_req.rid)
        ):
            reason = recv_req.finished_reason or {}
            self._pcl_abort_req(
                paused,
                reason.get("message") or recv_req.abort_message or "Aborted",
                reason.get("status_code"),
            )
        return super().abort_request(recv_req)

    def _poll_timeout_aborts(self):
        aborts = super()._poll_timeout_aborts()
        paused = self._paused_chunked_req
        timeout = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
        if paused is not None and timeout > 0:
            entry = paused.time_stats.forward_entry_time
            if 0 < entry < time.perf_counter() - timeout and not any(
                a.rid == paused.rid for a in aborts
            ):
                aborts.append(
                    AbortReq(
                        rid=paused.rid,
                        finished_reason={
                            "type": "abort",
                            "status_code": 503,
                            "message": "Request running timeout reached.",
                        },
                    )
                )
        return aborts

    def is_fully_idle(self, for_health_check=False):
        if self._paused_chunked_req is not None:
            return False
        return super().is_fully_idle(for_health_check)

    def collect_inflight_reqs(self):
        reqs = super().collect_inflight_reqs()
        if self._paused_chunked_req is not None:
            reqs.add(self._paused_chunked_req)
        return reqs

    def pause_generation(self, recv_req):
        paused = self._paused_chunked_req
        if paused is not None and recv_req.mode == "retract":
            if paused not in super().collect_inflight_reqs():
                retract_all(
                    reqs=[paused],
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    tree_cache=self.tree_cache,
                    hisparse_coordinator=self.hisparse_coordinator,
                    offload_kv=False,
                )
                self._add_request_to_queue(paused)
            self._paused_chunked_req = None
            self._paused_chunked_req_since = None
        return super().pause_generation(recv_req)

    def init_invariant_checker(self):
        super().init_invariant_checker()
        if self.pcl_scheduler is not None:
            self.invariant_checker = PclInvariantChecker(
                **{
                    f.name: getattr(self.invariant_checker, f.name)
                    for f in fields(self.invariant_checker)
                },
                get_pcl_paused_req=lambda: self._paused_chunked_req,
            )

    def init_load_inquirer(self):
        super().init_load_inquirer()
        if self.pcl_scheduler is not None:
            self.load_inquirer = PclLoadInquirer(
                **{
                    f.name: getattr(self.load_inquirer, f.name)
                    for f in fields(self.load_inquirer)
                },
                get_pcl_paused_req=lambda: self._paused_chunked_req,
            )
