# R2-E3：A0 单步吞吐的 kernel 级剖析（t17）

**判决：配置层没有空间（≤1.05×）；≥2× 只能重写引擎内核（融合 + 更快的 GEMM），实测天花板 3.2–3.8×、现实 2.0–2.5×。**

口径：引擎与门 0 同参（`--dense --mixed_precision fp16 --num_k 200 --chunk 10`，padded `264/91`，图 0 = 240 节点，GPU1 = RTX 5090，fp16 稠密峰值 209.5 TFLOPS），只测量、不改权重/宽度/K/decode_every；捕获臂 20 block = 200 步（每 block 一 `sync`，A0 语义），eager 臂 50 步；同刻 GPU1 有他人 17 GB 常驻但利用率 0%（不占算力）；两次独立跑 0.8264 / 0.8245 ms/步，分组读数差 <0.3% ⇒ 下列数字不是噪声。成本 ≈0.04 GPU·h；数据 `outputs/v11/r2/e3_profile/kernels.json`，脚本 `models/mis_v13_profile.py`（新增，未碰冻结文件）。

## (A) 0.826 ms/步 去了哪里

| 口径 | ms/步 | 读法 |
| --- | --- | --- |
| wall（捕获臂） | **0.825** | 与门 0 的 0.867 同量级 |
| device busy | 0.782 | 占 wall **94.8%** |
| gap（launch+sync+空隙） | 0.043 | **5.2%** ⇒ 不是 launch-bound |
| eager（同算术不捕获） | 5.97 / 4.59 | 捕获本身已值 **5.6–7.2×** |

每步 device 时间分组（两次跑）：**pointwise 425 µs（54%）**、GEMM 345 µs（44%）、reduce 8.3、sample 1.6、other 3.2 ⇒ **95 个 kernel/步，其中 71 次 pointwise**。前四个 kernel：

| 占比 | 次/步 | µs/次 | kernel |
| --- | --- | --- | --- |
| 19.3% | 5.0 | 30.2 | `cutlass_80_tensorop_f16_s16816gemm_64x64`（adj@support） |
| 12.6% | 6.0 | 16.5 | `unrolled_elementwise_kernel`（sigmoid/where/mul） |
| 12.1% | 5.0 | 19.0 | `cutlass_80_wmma_f16_s161616gemm_32x32`（x@W） |
| 9.6% | 28.0 | 2.7 | `vectorized_elementwise_kernel<4, float16_…>`（archive 等小块） |

GEMM：**17.84 GFLOP/步 ÷ 329 µs = 54.2 TFLOP/s = 峰值 25.9%**；同形状裸 `matmul`（264×264×264×52800）= 93–118 TFLOP/s（44–56%）⇒ GEMM 侧还有 1.6–2.2×。两条旧猜测被反证：`cusparse::csrmm_alg2` 只 6.4 µs（**0.8%**）⇒「CSR 是瓶颈」不成立；gap 5.2% ⇒「去 host 同步能大幅提速」不成立。**一半时间在 71 个 pointwise kernel 上**，每个只算 (264,200) = 10.6 万元素、2.7–21 µs ⇒ 延迟/占用受限，不是带宽受限。

## (B) 三类提速

| 类别 | 理论倍率 | 要改什么 | 改数值？ |
| --- | --- | --- | --- |
| 1 减少 launch / 去同步 | **≤1.05×** | 去掉每 block 的 `sync` | 否 |
| 2a 融合 pointwise（71→~5 kernel） | pointwise **×10**（425→30–40 µs）⇒ 整步 **≈1.9×** | 手写 fused kernel（sigmoid+rand+lt+where+mul+archive） | **是** |
| 2b 更快 GEMM / dtype（54→93+ TFLOP/s） | 整步 **≈1.3×**；micro bf16 111、fp32 46 TFLOP/s | batched GEMM、5 层并一次、bf16（fp8 未测） | **是** |
| 3 跨图复用捕获图 | **1.00×** | 无需改：捕获 1.6 s 摊到 1000 图 = 0.0016 s/图 | 否 |

## (C) 1 s 档还有没有 ≥2.0× 步数空间

- 下界（各项取已实测最好值）：GEMM 17.84 GFLOP ÷ 93–118 TFLOP/s = **151–191 µs** + pointwise（实测最小 kernel 延迟 × 95）= 64 µs + other 3 µs ⇒ **0.218–0.259 ms/步、天花板 3.2–3.8×**。
- 达成条件：(i) 五个 GEMM 全跑到裸 GEMM 速率（+58–118%），(ii) 95 个 kernel 融到 ~5 个（pointwise ×10）⇒ **引擎重写（Triton/CUDA 融合内核，或更少的图节点），不是调参**；现实预期 **2.0–2.5×**（0.33–0.41 ms/步）⇒ 1 s 档 810 → **1600–1950 步**，按步长曲线约 +0.03。
- **结论：配置层 1.05×，≥2× 不可达；重写可达 2.0–2.5×，需 1–2 天引擎工作。** 重写后必须重跑门 0 的 999/1000 复现检查（融合改变 fp16 累加顺序：允许 ±1 图级差异，不允许均值漂移 >0.01）。
