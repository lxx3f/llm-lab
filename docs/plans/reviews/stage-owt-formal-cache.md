# OWT 正式 token cache 阶段审查

审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）

## 完成范围

- 使用 `scripts/encode_token_cache.py` 生成正式范围 OWT cache；
- train byte limit：512 MiB；
- validation byte limit：64 MiB；
- newline-aligned UTF-8 流式编码；
- little-endian `uint16` token 文件；
- source/tokenizer/cache metadata hash 校验；
- memory-mapped cache 加载；
- batch sampler smoke；
- 正式范围 cache 接入 Dense training 配置；
- 使用正式 cache 完成 100-step Dense smoke。

## 产物统计

```text
train:
  requested_max_bytes: 536870912
  encoded_bytes: 536870901
  token_count: 143918122
  token_file_sha256: c8de1424ded43c48e6d7ab570da88e55fda452fc3840be70691b47aa7341276b

validation:
  requested_max_bytes: 67108864
  encoded_bytes: 67105041
  token_count: 17999093
  token_file_sha256: f7ca22ba6ee5c000fc11377a5d3a2f9e5bf8b53602e573df40ae5b75afa99069
```

## 验证结果

正式 cache 的 source、tokenizer、metadata 和 mmap batch smoke 均通过：

```text
train batches_per_epoch: 562180
validation batches_per_epoch: 70308
sample shape: [4, 64]
```

正式 cache Dense smoke：

```text
step 100 epoch: 0
step 100 validation_loss: 22.880113
step 100 last_train_loss: 20.823641
checkpoint: artifacts/checkpoints/dense-owt-formal-cache-smoke.pt
```

## 结果边界

- 当前仍为 100-step、d_model=64 的训练闭环 smoke；
- 不代表完整 OWT 训练质量、收敛结果或模型规模结论；
- tokenizer artifact 仍是 64 MiB train 前缀训练得到的 8192 vocab artifact；
- 尚未进行 Dense/MoE 公平训练对比；
- cache 和 checkpoint 均不提交 Git。

## 审查状态

审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）

本阶段已由 `reviewer` 完成只读审查。Critical：无；Warnings：已处理结果追踪、协议边界和已完成项归类；审查结论：通过。
