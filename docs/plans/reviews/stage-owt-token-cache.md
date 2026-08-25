# OWT token cache 阶段审查

审查模型：minimax-cn/MiniMax-M3
审查 agent：reviewer（subagent dispatch 名称）
实际 provider/model：PI_PROVIDER=minimax-cn, PI_MODEL=MiniMax-M3
可选运行标识：PI_AGENT_NAME（本次 harness 未提供）

## 完成范围

- 新增 `architecture_lab/data/token_cache.py`；
- 新增 `scripts/encode_token_cache.py`；
- 新增 token cache 公共导出；
- 按 UTF-8 行流式读取源文件；
- 按完整行进行 newline-aligned byte limit 截断；
- 使用 little-endian `uint16` 保存 token IDs；
- metadata 绑定 source SHA-256、tokenizer artifact SHA-256、split、byte limit、token count 和 cache SHA-256；
- 新增 cache loader/validator；
- 新增单元测试和 CLI 测试；
- 新增 `docs/protocols/owt-token-cache.md`；
- 使用真实 OWT train/validation 各 1 MiB 完成流式编码和 metadata 校验；
- metadata validator 补充校验 metadata type/version、source encoding、source size、newline alignment 和 tokenizer special token IDs；
- 增加 CRLF UTF-8 fixture，验证换行字节保留和 round-trip；
- 修复 metadata 写入失败时 token、metadata 和临时文件的清理。

## 验证结果

```text
Token cache unit tests: 6 passed
Token cache CLI tests: 3 passed
BPE tokenizer tests: 8 passed
Tokenizer CLI tests: 2 passed
Dense + MoE tests: 13 passed
Stage 0 schema examples: 5 passed
```

真实 OWT 试跑：

```text
- train encoded_bytes: 1045941（受 newline-aligned 截断影响，低于 1 MiB 上限）
- train token_count: 276038
- validation encoded_bytes: 1048241（受 newline-aligned 截断影响，低于 1 MiB 上限）
- validation token_count: 276038
- cache validation: PASS
```

## 未完成范围

- 尚未生成正式 512 MiB train / 64 MiB validation cache；
- 尚未实现 batch sampler；
- 尚未实现 Dense training loop；
- 尚未实现 validation、checkpoint、resume 和 generation；
- 当前 metadata 的 data version 由 CLI 参数提供，尚未从 tokenizer/source registry 自动读取。

## 风险和边界

- `uint16` 要求 tokenizer vocab size 不超过 65536；
- cache 绑定完整 source hash，但 byte limit 只绑定到 newline-aligned 实际编码范围；
- Windows 文本文件的换行字节语义已通过 UTF-8 bytes fixture 验证；
- cache 和 OWT 原始数据均被 `.gitignore` 排除。

## 下一步

```text
固定 Dense batch sampler 协议
→ 生成正式范围 token cache
→ Dense training loop
```

## 审查结论

本阶段由 `reviewer` subagent dispatch 使用 `minimax-cn/MiniMax-M3` 完成只读审查。

- 上轮指出的 metadata 清理、协议字段校验、CRLF fixture 和测试数字问题已修复；
- Critical：无；
- Warnings：非阻塞，记录为后续优化；
- 审查结论：通过。
