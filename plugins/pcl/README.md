# PCL 插件：Edge406 当前版本

更新日期：2026-10-13。基于用户提供的 `my_pcl_plugin`，适配 SGLang 提交
`50ec9702d0dc680d114c311a4df83d96a15062a0`，安装包为 `sglang-pcl==0.2.0`。

## 使用

远端插件目录为 `/data0/wxtang/sglang/plugins/pcl`，已 editable 安装到
`/data0/wxtang/envs/sglang`。在 Linux/NVIDIA CUDA 环境中使用下列命令，
将 `MODEL_PATH` 替换为本地模型目录或环境能访问的 Hugging Face 模型标识。

```bash
export SGLANG_PLUGINS=pcl
export CUDA_HOME=/usr/local/cuda-12.6

/data0/wxtang/envs/sglang/bin/python -m sglang.launch_server \
  --model-path MODEL_PATH \
  --schedule-policy priority-cost-lpm \
  --enable-priority-scheduling \
  --schedule-low-priority-values-first \
  --default-priority-value 1 \
  --disable-priority-preemption \
  --disable-overlap-schedule \
  --tp-size 1 \
  --chunked-prefill-size 4096
```

普通 TP 可以增加到多卡；DP=1、PP=1。当前明确拒绝 overlap、CP/DCP、DP attention、
PD/PDMUX、speculative/diffusion decoding、mixed chunk，以及非标准/关闭的 radix cache。
LoRA、多模态、beam 组合尚未做完整验证。

启用条件是 `--schedule-policy priority-cost-lpm`。旧版的 `SGLANG_PCL_ENABLED`
和 `SGLANG_PCL_CONFIG` 不再用于配置此版本。选择 `fcfs` 等原生策略并重启，
即可使用原生调度行为；已测试插件安装后的 FCFS 路径。

需要重新安装时：

```bash
cd /data0/wxtang/sglang
/home/wxtang/.local/bin/uv pip install \
  --python /data0/wxtang/envs/sglang/bin/python \
  --no-deps --no-build-isolation -e plugins/pcl
```

## 插件接入

entry point 为 `sglang.srt.plugins` 组中的 `pcl = sglang_pcl:plugin_entry`。
插件使用 REPLACE 替换 Scheduler、SchedulePolicy、msgspec ServerArgs 及 Engine 保存的
配置类引用；使用 AROUND 扩展 CLI choices 和原生参数校验。

`PclScheduler` 通过 `super()` 调用原生 prefill 准入方法，不维护复制的准入函数。
policy 将被 KV 门控的 BACKGROUND 请求临时移出 waiting queue，调用结束后在 `finally`
中恢复；暂停期间也暂存优先级不高于原请求的候选。原生准入阶段看到的是过滤后的队列，
因此队列长度等观察值与逐项 `continue` 的实现可能不同。

ServerArgs 扩展使用真正的 msgspec 子类，支持 CLI、Engine 关键字参数以及子进程序列化。
原生校验中的 policy whitelist 通过临时解析声明适配，结束或异常时都会移除临时声明。
升级 SGLang 后仍需检查接口并重跑测试。

## 调度语义

请求在成功入队时建立一次 profile，固定 `matched_len` 和 `recompute_len`。
排序键为 `(priority, tier, -matched_len, wait_queue_entry_time)`，数字小的 priority 优先。
实时 prefix metadata 每轮刷新，但不改变固定 profile。重新排队仍复用原 profile。

FAST 为 `recompute_len <= T_fast`；超过 `T_demote` 为 BACKGROUND；其余为 NORMAL。
allocator 占用率达到 `W_low` 时门控 BACKGROUND。超过 `T_aging` 后永久提升为 NORMAL。
aging 不改变 priority，不能保证跨优先级不饥饿；KV 门控也不是 OOM 保证，原生容量检查仍生效。

## 长请求抢占

最多保留一个活动 chunk 和一个暂停 chunk，按 chunk 边界切换：

```text
A 分块 prefill → 更高优先级 B 到达 → 暂停 A → B 可执行多个 chunk
→ B 不再占用 chunk 槽位且更高优先级候选排空 → 恢复 A
```

不允许嵌套抢占。被 KV 门控的候选不会触发暂停，也不会阻止恢复。
若更高优先级候选无法准入，则同一轮回退继续原 chunk，避免因请求槽位不足等原因停滞。
暂停请求继续持有 KV 和请求槽位；取消、原生运行超时、资源检查、负载统计、权重版本记录、
引擎 retract 都已包含暂停请求。默认 Fast-Fail 关闭时，持续高优先级流量仍可能长期推迟 A。

## Fast-Fail

`T_failfast > 0` 时启用：

- waiting queue：从 profile 入队时间计时，超过阈值或内部 `slo_deadline` 到期即失败。
- 暂停 chunk：从本次暂停开始计时，超过阈值或内部 deadline 到期即失败。

`T_failfast <= 0` 会关闭两种检查，包括 deadline 分支。内部 deadline 使用服务端单调时钟域，
当前没有新增对应 HTTP/Engine 输入参数。

TP 组内 rank 0 决定老化、门控和 Fast-Fail，再广播决策；各 rank 清理相同请求。
waiting 和 paused 请求共用终止处理，paused 额外释放保留的 KV/槽位并清空暂停状态。
发送 AbortReq 时显式携带 finish reason，Fast-Fail 返回 HTTP 503。
普通用户取消沿用原生 RID 前缀匹配；Fast-Fail 对 waiting 请求按完整 RID 删除。

检查在 prefill 调度阶段发生，不保证在长 GPU step 中按毫秒精确触发。
如希望先 aging 再争取执行，设置 `T_failfast > T_aging` 并保留准入余量。
较小阈值允许用于快速丢弃实验，但可能在 aging 前失败。

| CLI 参数 | 默认值 |
| --- | --- |
| `--pcl-T-fast` | 512 token |
| `--pcl-T-demote` | 8192 token |
| `--pcl-W-low` | 0.7 |
| `--pcl-T-aging` | 30000 ms |
| `--pcl-T-failfast` | 0 ms |
| `--pcl-c-pre` | 1.0 |

默认值统一来自 `PriorityCostLpmConfig`。配置要求数值有限、
`0 <= T_fast <= T_demote`、`0 <= W_low <= 1`、`T_aging >= 0`、`c_pre >= 0`。
`c_pre` 只用于 profile 的 `c_miss`，不直接参与排序。

## 验证

```bash
cd /data0/wxtang/sglang/plugins/pcl
PYTHON=/data0/wxtang/envs/sglang/bin/python
SGLANG_PLUGINS=pcl $PYTHON -m unittest discover -s tests -v
$PYTHON tests/check_plugin.py
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda-12.6 \
  $PYTHON tests/smoke_engine.py --model-path MODEL_PATH
CUDA_VISIBLE_DEVICES=1,2 CUDA_HOME=/usr/local/cuda-12.6 \
  $PYTHON tests/smoke_engine.py --model-path MODEL_PATH --tp 2
CUDA_VISIBLE_DEVICES=1,2 CUDA_HOME=/usr/local/cuda-12.6 \
  $PYTHON tests/smoke_engine.py --model-path MODEL_PATH --tp 2 --paused-fastfail
CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda-12.6 \
  $PYTHON tests/smoke_engine.py --model-path MODEL_PATH --fastfail
```

测试使用现有 OPT-125M 快照，没有下载新模型。GPU 测试结束后关闭 Engine。
Fast-Fail 测试故意使用很小的超时，不能作为正常服务配置。
验证结果与清理记录见 `VALIDATION.md`，不代表吞吐量或尾延迟收益。

## 文件与回退

`algorithm.py` 保存策略标识、配置和纯算法；`policy.py` 管理排序同步与队列暂存；
`scheduler.py` 管理抢占和请求生命周期；`components.py` 补充统计和资源检查；
`server_args_ext.py` 接入参数；`__init__.py` 注册 hooks。

旧版复制的 `admission.py` 已移除，微小的 `constants.py` 已合并到算法模块。
本地一次性迁移脚本也已移除。用户提供的原始 `my_pcl_plugin` 保留作参考。
回退归档在 `/data0/wxtang/sglang-pcl-backups/pcl-before-local-replacement-20261012.tar.gz`，
不参与插件加载。恢复源码后需重装 editable 包，以恢复旧版 entry point。
