# PCL 清理与验证记录

日期：2026-10-13。远端：Edge406。SGLang 基线：
`50ec9702d0dc680d114c311a4df83d96a15062a0`。
代码部署于 `/data0/wxtang/sglang/plugins/pcl`，包版本 `sglang-pcl==0.2.0`。

## 本次清理

- 删除远端旧版 `sglang_pcl/admission.py`，及对应缓存。
- 将 `constants.py` 的策略标识合并到 `algorithm.py`，删除独立模块及缓存。
- 合并 waiting Fast-Fail、paused Fast-Fail 和暂停取消的公共终止处理。
  paused 请求仍额外释放 KV/请求槽位；发送响应时保留显式 finish reason。
- CLI 参数默认值统一从 `PriorityCostLpmConfig` 读取。
- 删除本地一次性 `adapt.py`、`adapt_scheduler.py` 迁移脚本。
- 本地旧 `pcl_plugin` 目录归档后删除，当前本地适配代码为 `pcl_replacement`。
- 更新远端 README，替换旧版环境变量和单轮暂停的说明。

用户提供的 `my_pcl_plugin` 原始代码保留作参考；SGLang 核心文件没有改动。

## 验证结果

| 检查 | 结果与证据 |
| --- | --- |
| 补完 Fast-Fail 修复后的回归 | 22 项通过，`/tmp/pcl-v2-unit-final.log` |
| 补完 waiting Fast-Fail | HTTP 503、缓存清理通过，`/tmp/pcl-v2-waiting-fastfail-final.log` |
| 补完 TP=2 paused Fast-Fail | 两个 rank 同轮清理，B 完成，客户端 503，`/tmp/pcl-v2-paused-fastfail-final.log` |
| 简化后的回归 | 22 项通过，`/tmp/pcl-clean-unit.log` |
| 简化后的实际 entry point | 类替换、原生准入复用、重复加载通过，`tests/check_plugin.py` |
| 简化后的 TP=2 长请求抢占 | A=1800 token、B=1500 token，B 抢占 A；日志确认两个 rank 暂停/恢复，输出均与顺序执行一致，`/tmp/pcl-clean-tp2.log` |
| 简化后的暂停取消 | TP=2 下取消暂停 A，B 仍完成，随后缓存清理成功，同上日志 |
| 简化后的 waiting Fast-Fail | 客户端 HTTP 503，缓存清理成功，`/tmp/pcl-clean-waiting.log` |
| 简化后的 paused Fast-Fail | TP=1 下 A 返回 503、B 完成、缓存清理成功，`/tmp/pcl-clean-paused.log` |
| Python 编译、Black | 6 个运行模块与 3 个测试文件通过 |
| 仓库检查 | `git diff --check` 通过，仅插件目录为未跟踪新增文件，无核心源码改动 |
| 资源回收 | 测试 Engine 全部关闭，各 GPU 显存回到测试前的 4 MiB |

模型使用现有本地 `facebook/opt-125m` 快照
`27dcfa74d334bc871f3234de431e71c6eeba5dd6`，NVIDIA RTX A6000、CUDA 12.6、
`attention_backend=triton`、`chunked_prefill_size=64`、关闭 overlap/CUDA graph。
GPU 测试启用了 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=1`，上述最终日志未发现
Traceback、ERROR 或 memory leak 报告。没有下载新模型。

Fast-Fail 测试的 50/70 ms 阈值是为触发分支设置的，不是正常服务推荐配置。
验证覆盖功能与资源生命周期，不代表吞吐量、尾延迟或所有模型组合均已验证。

## 回退归档

- 旧服务器版：`/data0/wxtang/sglang-pcl-backups/pcl-before-local-replacement-20261012.tar.gz`。
- 本次清理前：`/data0/wxtang/sglang-pcl-backups/pcl-before-cleanup-20261013.tar.gz`。
- 本地旧版：`F:\F_Disk\codex-session\sglang\backups\pcl_plugin-before-cleanup-20261013.zip`。

归档均不参与插件加载。回退到旧服务器版后需重新安装 editable 包以恢复对应 entry point。
