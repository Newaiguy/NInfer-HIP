## 0.1.0-pre-r9 — KV 记忆与 MTP 同时可用，长上下文推到 192K

这一版把两件之前不能并存的能力合到一起，并把长上下文的上限从 98K 推到 192K。

### 这一版做了什么

1. **KV 记忆 + MTP 同时开**。pre-r8 里 `--resident-tokens` 与 `--spec mtp` 互斥（一开就报
   `request reservation exceeds Engine shared KV capacity`）。这一版把窗口基址的纪律推进到投机路径：
   - verify 的注意力 envelope 改用紧凑坐标（下界夹 1，0 号槽是内核哨兵）；
   - MTP 配额按窗口 `resident_entitlement_tokens` 计算，而不是按整段上下文；
   - 投机缓存的 `slide_prefix_pages` 与主 KV 同步滑页；
   - verify 的 RoPE 位置补回窗口基址（窗口只改"注意力扫哪些槽位"，不改旋转相位）。

   实测：16,814 token、窗口 8192、召回开、MTP2 → **decode 52.9 t/s**，输出与 GT 逐 token 相同。

2. **修掉一个致命 bug**。`ensure_sequence_kv_mapped()` 里解引用空的 `sequence.kv->backend`，
   拿到垃圾句柄后抛 `KV address-space handle is stale` —— 这曾让**所有**窗口用例在第一个 chunk
   之前就失败。加上 `has_value()` 判断后恢复。

3. **长上下文实测**（这台 16GB 卡上，128K 及以上的文本**只能**走窗口，全上下文预留会直接被拒）：
   | prompt | 窗口 | prefill | decode | 结果 |
   |---|---|---|---|---|
   | 98,182 | 32768 | 556.0 | 35.4 | 逐 token 正确 |
   | 130,700 | 32768 | 544.3 | 34.4 | 逐 token 正确 |
   | 196,427 | 32768 | 535.7 | 33.8 | 逐 token 正确 |

4. **档位结论**：长上下文窗口 **>= 32768 起步**。同一份 192K 语料开到 16384 时，模型会先复读噪声、
   再把答案补在后半段，ID 串不可用。

5. **与 llama.cpp kvmem 的同机同模型对比**（两边都开 MTP2）：
   | prompt | 上游 prefill / decode | 本引擎 prefill / decode |
   |---|---|---|
   | 16K | 762.9 / 34.56 | 799.7 / 52.9 |
   | 32K | 633.3 / 35.69 | 806.9 / 37.5 |
   | 64K | 454.0 / 35.16 | 693.8 / 30.6 |
   | 98K | 380.4 / 13.98 | 556.0 / 35.4 |
   | 192K | 336.8 / 22.27 | 535.7 / 33.8 |

### 已知问题（P3，暂不修）

召回真正生效的档位，最后一个 token 可能是 `248044` 而不是逐 token 基线里的 `248046`。
两者同属停止符集合（`finish=3` 即 `StopToken`），正文、语义与速度都没有影响；成因是召回后
重放最后一个 chunk、末位 logits 被重算。详见 `docs/KVMEM.md` §10。

### 升级建议

- 只是短上下文聊天：不用动任何参数，行为与 pre-r8 一致。
- 长文（>= 48K）：`--kv rk4v4-e8 --ctx 262144 --resident-tokens 32768 --archive-mib 3072 --kv-retrieval 1`
- 想同时要 MTP 加速：再加 `--spec mtp --draft-tokens 2`，本期起可以共存。

**完整更新日志见 [docs/CHANGELOG.md](docs/CHANGELOG.md)；实现细节见 [docs/KVMEM.md](docs/KVMEM.md)。**
