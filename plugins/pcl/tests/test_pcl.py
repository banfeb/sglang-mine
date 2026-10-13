import argparse
import base64
import pickle
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.srt.plugins import load_plugins

load_plugins()

from sglang.srt.server_args import ServerArgs
from sglang_pcl.algorithm import (
    PriorityCostLpmConfig,
    PriorityCostLpmScheduler,
    ReqProfile,
    Tier,
)
from sglang_pcl.algorithm import PCL_POLICY
from sglang_pcl.policy import PclSchedulePolicy
from sglang_pcl.scheduler import PclScheduler
from sglang_pcl.server_args_ext import check_server_args_pcl


def request(rid, priority=1, recompute=10, matched=0, age=0):
    return NS(
        rid=rid,
        priority=priority,
        skip_background=False,
        req_profile=ReqProfile(
            matched,
            matched + recompute,
            recompute,
            recompute,
            priority,
            None,
            enqueue_time=time.monotonic() - age,
        ),
        time_stats=NS(wait_queue_entry_time=1, trace_ctx=NS(abort=Mock())),
        finished=lambda: False,
        finished_reason=NS(
            to_json=lambda: {
                "type": "abort",
                "status_code": 503,
                "message": "Fast-failed",
            }
        ),
    )


class ConfigTests(unittest.TestCase):
    def test_fresh_process_unpickles_before_plugin_loading(self):
        args = ServerArgs(model_path="dummy", pcl_T_fast=77)
        payload = base64.b64encode(pickle.dumps(args)).decode()
        code = "import base64,pickle,sys; a=pickle.loads(base64.b64decode(sys.argv[1])); assert a.pcl_T_fast == 77"
        subprocess.run([sys.executable, "-c", code, payload], check=True, timeout=90)

    def test_cli_fields_and_pickle(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--schedule-policy",
                PCL_POLICY,
                "--pcl-T-fast",
                "123",
                "--pcl-T-failfast",
                "5000",
            ]
        )
        args = ServerArgs.from_cli_args(ns)
        self.assertEqual(args.pcl_T_fast, 123)
        defaults = ServerArgs(model_path="dummy")
        for name, value in vars(PriorityCostLpmConfig()).items():
            self.assertEqual(getattr(defaults, "pcl_" + name), value)
        copy = pickle.loads(pickle.dumps(args))
        self.assertEqual(copy.pcl_T_failfast, 5000)
        self.assertEqual(copy.schedule_policy, PCL_POLICY)

    def test_validator_restores_policy_even_on_error(self):
        from sglang.srt.arg_groups.overrides import resolving_view

        args = ServerArgs(model_path="dummy", schedule_policy=PCL_POLICY)

        def validate(record):
            self.assertEqual(resolving_view(record).schedule_policy, "fcfs")
            raise ValueError("validation failed")

        with self.assertRaises(ValueError):
            check_server_args_pcl(validate, args)
        self.assertEqual(resolving_view(args).schedule_policy, PCL_POLICY)
        self.assertEqual(args._resolved_overrides, [])

    def test_native_policy_unchanged(self):
        args = ServerArgs(model_path="dummy", schedule_policy="lpm")
        validate = Mock()
        check_server_args_pcl(validate, args)
        validate.assert_called_once_with(args)
        self.assertEqual(args.schedule_policy, "lpm")


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.core = PriorityCostLpmScheduler(
            PriorityCostLpmConfig(), None, NS(size=100, available_size=lambda: 30)
        )

    def test_sort_gate_aging(self):
        a, b, c = (
            request("a", recompute=9000),
            request("b", matched=50),
            request("c", priority=0, recompute=9000, age=31),
        )
        queue = [a, b, c]
        self.core.calc_priority(queue)
        self.assertEqual([r.rid for r in queue], ["c", "b", "a"])
        self.assertTrue(a.skip_background)
        self.assertFalse(c.skip_background)
        self.assertEqual(c.req_profile.tier, Tier.NORMAL)

    def test_failfast_disabled_and_deadline(self):
        req = request("a", age=100)
        req.req_profile.slo_deadline = 0
        self.assertFalse(self.core.check_failfast(req, time.monotonic()))
        self.core.config.T_failfast = 60000
        self.assertTrue(self.core.check_failfast(req, time.monotonic()))

    def test_holdout_restores_after_native_replaces_queue(self):
        policy = object.__new__(PclSchedulePolicy)
        policy._held_background = []
        blocked, low, high = (
            request("b"),
            request("l", priority=5),
            request("h", priority=0),
        )
        blocked.skip_background = True
        queue = [blocked, low, high]
        policy.arm_holdout(priority_ceiling=2)
        policy._hold_out_background(queue)
        self.assertEqual(queue, [high])
        remaining = []  # native admission assigns a new waiting_queue
        policy.restore_held_background(remaining)
        self.assertEqual(remaining, [blocked, low])
        policy.restore_held_background(remaining)
        self.assertEqual(len(remaining), 2)

    def test_non_pcl_processed_tokens_forwarded(self):
        policy = object.__new__(PclSchedulePolicy)
        policy.policy = "fcfs"
        with patch.object(PclSchedulePolicy.__bases__[0], "calc_priority") as native:
            policy.calc_priority([], processed_tokens=99)
            native.assert_called_once_with([], None, processed_tokens=99)


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.s = object.__new__(PclScheduler)
        self.s.pcl_scheduler = PriorityCostLpmScheduler(
            PriorityCostLpmConfig(), None, NS()
        )
        self.s.tp_cpu_group = None
        self.a, self.b = request("a", priority=2), request("b", priority=0)
        self.s.chunked_req = self.a
        self.s._paused_chunked_req = None
        self.s._paused_chunked_req_since = None
        self.s._pending_chunked_abort_req = None
        self.s.waiting_queue = [self.b]

    def test_long_preemptor_and_resume(self):
        self.s._pcl_pause_or_resume_chunked_req()
        self.assertIs(self.s._paused_chunked_req, self.a)
        self.s.chunked_req = self.b
        self.s.waiting_queue = [request("c", priority=-1)]
        self.s._pcl_pause_or_resume_chunked_req()
        self.assertIs(self.s.chunked_req, self.b)  # no nested preemption
        self.assertIs(self.s._paused_chunked_req, self.a)
        self.s.chunked_req = None
        self.s.waiting_queue = []
        self.s._pcl_pause_or_resume_chunked_req()
        self.assertIs(self.s.chunked_req, self.a)

    def test_gate_blocked_high_priority_does_not_pause(self):
        self.b.skip_background = True
        self.s._pcl_pause_or_resume_chunked_req()
        self.assertIs(self.s.chunked_req, self.a)

    def test_failed_admission_immediately_resumes_original_chunk(self):
        running = NS()
        self.s.chunked_req = None
        self.s._paused_chunked_req = self.a
        self.s.grammar_manager = NS(has_waiting_grammars=lambda: False)
        self.s._pcl_prefill_admission_prework = Mock()
        self.s.policy = NS(arm_holdout=Mock(), restore_held_background=Mock())
        seen = []

        def native(batch):
            seen.append(self.s.chunked_req)
            return NS(batch_to_run=None if len(seen) == 1 else "A", running_batch=batch)

        with patch.object(
            PclScheduler.__bases__[0], "get_new_batch_prefill", side_effect=native
        ):
            plan = self.s.get_new_batch_prefill(running)
        self.assertEqual(seen, [None, self.a])
        self.assertEqual(plan.batch_to_run, "A")
        self.assertIsNone(self.s._paused_chunked_req)

    def test_holdout_restored_on_admission_exception(self):
        self.s.grammar_manager = NS(has_waiting_grammars=lambda: False)
        self.s._pcl_prefill_admission_prework = Mock()
        self.s.policy = NS(arm_holdout=Mock(), restore_held_background=Mock())
        with patch.object(
            PclScheduler.__bases__[0],
            "get_new_batch_prefill",
            side_effect=RuntimeError("admission"),
        ):
            with self.assertRaises(RuntimeError):
                self.s.get_new_batch_prefill(NS())
        self.s.policy.restore_held_background.assert_called_once_with(
            self.s.waiting_queue
        )

    def test_gate_blocked_high_priority_does_not_block_resume(self):
        self.s._paused_chunked_req = self.a
        self.s.chunked_req = None
        self.b.skip_background = True
        self.s._pcl_pause_or_resume_chunked_req()
        self.assertIs(self.s.chunked_req, self.a)

    def test_cancel_paused_is_routed_without_pending_marker(self):
        self.s._paused_chunked_req = self.a
        self.s._pcl_abort_req = Mock()
        recv = NS(
            abort_all=False, rid="a", finished_reason=None, abort_message="cancel"
        )
        with patch.object(PclScheduler.__bases__[0], "abort_request") as native:
            self.s.abort_request(recv)
            self.s._pcl_abort_req.assert_called_once_with(self.a, "cancel", None)
            native.assert_called_once_with(recv)

    def test_abort_all_routes_paused_and_active(self):
        self.s._paused_chunked_req = self.a
        self.s._pcl_abort_req = Mock()
        recv = NS(abort_all=True, rid="", finished_reason=None, abort_message=None)
        with patch.object(PclScheduler.__bases__[0], "abort_request") as native:
            self.s.abort_request(recv)
            self.s._pcl_abort_req.assert_called_once()
            native.assert_called_once_with(recv)

    def test_paused_prevents_idle(self):
        self.s._paused_chunked_req = self.a
        self.assertFalse(self.s.is_fully_idle())

    def test_paused_fastfail_uses_pause_start(self):
        self.s.pcl_scheduler.config.T_failfast = 1000
        self.s._paused_chunked_req = self.a
        self.a.req_profile.enqueue_time = time.monotonic() - 100
        self.s._paused_chunked_req_since = time.monotonic()
        self.s._pcl_abort_req = Mock()
        self.s.waiting_queue = []
        self.s._pcl_failfast_expired_reqs()
        self.s._pcl_abort_req.assert_not_called()
        self.s._paused_chunked_req_since -= 2
        self.s._pcl_failfast_expired_reqs()
        self.s._pcl_abort_req.assert_called_once()

    def test_tp_follower_uses_broadcast_failures(self):
        self.s._pcl_abort_req = Mock()
        with patch(
            "sglang_pcl.scheduler.dist.is_initialized", return_value=True
        ), patch("sglang_pcl.scheduler.dist.get_rank", return_value=1), patch(
            "sglang_pcl.scheduler.broadcast_decision", return_value=(["b"], False)
        ):
            self.s._pcl_failfast_expired_reqs()
        self.assertEqual(self.s.waiting_queue, [])
        self.s._pcl_abort_req.assert_called_once_with(
            self.b,
            "Fast-failed: queued longer than pcl_T_failfast or SLO deadline passed.",
            503,
            {"reason": "FastFail"},
        )

    def test_paused_cleanup_preserves_active_abort_marker(self):
        self.s._paused_chunked_req = self.a
        self.s._pending_chunked_abort_req = self.b
        self.s._release_aborted_request = Mock()
        self.s.beam_coordinator = NS(retire_group=Mock())
        self.s.tree_cache = object()
        sender = Mock()
        self.s.ipc_channels = NS(send_to_tokenizer=NS(send_output=sender))
        with patch("sglang_pcl.scheduler.prepare_abort"), patch(
            "sglang_pcl.scheduler.release_kv_cache"
        ) as release, patch(
            "sglang_pcl.scheduler._make_abort_req", return_value="response"
        ):
            self.s._pcl_abort_req(self.a)
        release.assert_called_once_with(self.a, self.s.tree_cache, is_insert=False)
        self.assertIs(self.s._pending_chunked_abort_req, self.b)
        self.assertIsNone(self.s._paused_chunked_req)

    def test_waiting_fastfail_preserves_client_status(self):
        self.s._release_aborted_request = Mock()
        self.s.beam_coordinator = NS(retire_group=Mock())
        sender = Mock()
        self.s.ipc_channels = NS(send_to_tokenizer=NS(send_output=sender))
        with patch("sglang_pcl.scheduler.prepare_abort") as prepare, patch(
            "sglang_pcl.scheduler._make_abort_req", return_value="output"
        ) as make:
            self.s._pcl_abort_req(self.b, "Fast-failed", 503, {"reason": "FastFail"})
        prepare.assert_called_once_with(self.b, "Fast-failed", status_code=503)
        make.assert_called_once_with(
            self.b, finished_reason=self.b.finished_reason.to_json()
        )
        self.s._release_aborted_request.assert_called_once_with(self.b)
        sender.assert_called_once_with("output", self.b)


class AccountingTests(unittest.TestCase):
    def test_paused_uncached_tokens_counted_once(self):
        from sglang_pcl.components import PclInvariantChecker

        checker = object.__new__(PclInvariantChecker)
        req = NS(kv=NS(holds_kv=True, kv_allocated_len=19, cache_protected_len=16))
        checker.get_pcl_paused_req = lambda: req
        checker.get_chunked_req = lambda: None
        checker.get_last_batch = lambda: NS(reqs=[])
        checker.get_running_batch = lambda: NS(reqs=[])
        checker.page_size = 4
        with patch.object(
            PclInvariantChecker.__bases__[0],
            "_get_total_uncached_sizes",
            return_value=(0, 0),
        ):
            self.assertEqual(checker._get_total_uncached_sizes(), (4, 0))
            checker.get_last_batch = lambda: NS(reqs=[req])
            self.assertEqual(checker._get_total_uncached_sizes(), (0, 0))

    def test_paused_load_metrics(self):
        from sglang_pcl.components import PclLoadInquirer

        checker = object.__new__(PclLoadInquirer)
        req = NS(seqlen=100, prefix_indices=list(range(60)))
        object.__setattr__(checker, "get_pcl_paused_req", lambda: req)
        snapshot = NS(
            num_waiting_reqs=2,
            num_total_tokens=200,
            num_active_tokens=200,
            queues=NS(paused=0),
        )
        with patch.object(
            PclLoadInquirer.__bases__[0], "get_loads", return_value=snapshot
        ):
            result = checker.get_loads()
        self.assertEqual(result.num_waiting_reqs, 3)
        self.assertEqual(result.num_total_tokens, 240)
        self.assertEqual(result.queues.paused, 1)
        with patch.object(
            PclLoadInquirer.__bases__[0], "_get_num_pending_tokens", return_value=10
        ):
            self.assertEqual(checker._get_num_pending_tokens(), 50)


if __name__ == "__main__":
    unittest.main()
