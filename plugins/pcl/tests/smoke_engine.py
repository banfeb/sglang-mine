"""GPU regressions for the company PCL adapter; use a local OPT-125M model."""

import argparse
import asyncio
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--fastfail", action="store_true")
    parser.add_argument("--paused-fastfail", action="store_true")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    os.environ["SGLANG_PLUGINS"] = "pcl"
    os.environ["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY"] = "1"
    from fastapi import HTTPException
    from sglang import Engine

    extra = {}
    if args.fastfail:
        extra = dict(
            pcl_T_fast=0,
            pcl_T_demote=1,
            pcl_W_low=0,
            pcl_T_aging=30000,
            pcl_T_failfast=50,
        )
    if args.paused_fastfail:
        extra = dict(pcl_T_failfast=70)
    engine = Engine(
        model_path=args.model_path,
        tp_size=args.tp,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        schedule_policy="fcfs" if args.native else "priority-cost-lpm",
        enable_priority_scheduling=True,
        schedule_low_priority_values_first=True,
        disable_priority_preemption=True,
        default_priority_value=1,
        chunked_prefill_size=64,
        max_total_tokens=8192,
        max_running_requests=8,
        mem_fraction_static=0.15,
        attention_backend="triton",
        log_level="info",
        **extra,
    )
    sampling = {"max_new_tokens": 8, "temperature": 0, "ignore_eos": True}
    try:
        if args.fastfail:
            try:
                engine.generate(input_ids=[42] * 100, sampling_params=sampling)
            except HTTPException as error:
                assert error.status_code == 503 and "Fast-failed" in str(error), str(
                    error
                )
                print("PASS: waiting Fast-Fail returned HTTP 503", flush=True)
            else:
                raise AssertionError("Fast-Fail did not reject the gated request")
        elif args.native:
            output = engine.generate(
                "The capital of France is", sampling_params=sampling
            )
            assert output["meta_info"]["completion_tokens"] == 8, output
            print("PASS: native FCFS with plugin installed", flush=True)
        else:
            # Reference outputs + warmed kernels; clear prefixes so the next
            # requests genuinely span many chunks.
            inputs = [[42] * 1800, [43] * 1500]
            reference = [
                engine.generate(input_ids=ids, sampling_params=sampling)
                for ids in inputs
            ]
            engine.flush_cache()

            async def preempt(cancel=False, expect_fastfail=False):
                low = asyncio.create_task(
                    engine.async_generate(
                        input_ids=inputs[0],
                        sampling_params=sampling,
                        priority=2,
                        rid="pcl-low",
                    )
                )
                await asyncio.sleep(0.025)
                high = asyncio.create_task(
                    engine.async_generate(
                        input_ids=inputs[1],
                        sampling_params=sampling,
                        priority=0,
                        rid="pcl-high",
                    )
                )
                if cancel:
                    await asyncio.sleep(0.04)
                    engine.tokenizer_manager.abort_request("pcl-low")
                outputs = await asyncio.gather(
                    low, high, return_exceptions=cancel or expect_fastfail
                )
                if expect_fastfail:
                    assert isinstance(outputs[0], HTTPException), outputs[0]
                    assert outputs[0].status_code == 503 and "paused chunk" in str(
                        outputs[0]
                    ), outputs[0]
                    assert not isinstance(outputs[1], Exception), outputs[1]
                    assert outputs[1]["text"] == reference[1]["text"], outputs[1]
                elif cancel:
                    assert (
                        isinstance(outputs[0], HTTPException)
                        or outputs[0]["meta_info"]["finish_reason"]["type"] == "abort"
                    ), outputs[0]
                    assert not isinstance(outputs[1], Exception), outputs[1]
                    assert outputs[1]["text"] == reference[1]["text"], outputs[1]
                else:
                    for result, expected in zip(outputs, reference):
                        assert result["text"] == expected["text"], (result, expected)
                        assert result["meta_info"]["completion_tokens"] == 8, result

            if args.paused_fastfail:
                engine.loop.run_until_complete(
                    asyncio.wait_for(preempt(expect_fastfail=True), 60)
                )
                print(
                    "PASS: paused A Fast-Fail returned 503 while long B completed",
                    flush=True,
                )
            else:
                engine.loop.run_until_complete(asyncio.wait_for(preempt(), 60))
                print(
                    "PASS: long B preempts long A; both match sequential outputs",
                    flush=True,
                )
                engine.flush_cache()
                engine.loop.run_until_complete(
                    asyncio.wait_for(preempt(cancel=True), 60)
                )
                print("PASS: cancel paused A while long B completes", flush=True)
        result = engine.flush_cache()
        assert result, result
        print(
            "PASS: cache flush after all requests retired; TP=" + str(args.tp),
            flush=True,
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
