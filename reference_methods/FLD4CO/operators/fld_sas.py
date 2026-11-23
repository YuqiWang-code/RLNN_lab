"""Fractional Langevin Dynamics (FLD) 的核心算子重写 —— 对称 alpha-stable 噪声 + 截断。

论文：Fractional Langevin Dynamics for Combinatorial Optimization via Polynomial-Time
Escape, NeurIPS 2025, doi 10.52202/085713-3542（本地 PDF:
docs/参考文献/采样与退火/FLD-NeurIPS25.pdf）。

作者仓库 https://github.com/Thinklab-SJTU/FLD4CO 在 commit 083101d 只有 README 与
LICENSE，论文代码未公开，因此本文件不是作者实现的复现，而是按论文文本重写的算子。
对应位置：

* §4.1（p6）FLD 更新式：x <- x - tau * grad U(x) + (2 tau)^{1/alpha} * xi，
  xi ~ S-alpha-S（对称 alpha-stable），alpha = 2 退化为 Langevin dynamics。
* §4.1 / 附录 B：噪声按 |xi| <= d_noise 截断（论文 Table 6 用 d_noise = 9）。
* §4.3（p7）+ Table 6（p27）：MIS RB-[200-300] 的取值 alpha = 1.2、tau0 = 0.01、
  lambda = 1.02、K = 200 条并行链、T = 300 步。
* §1（p2）理论主张：高斯噪声下逃逸时间随势垒**高度**指数增长，alpha-stable 噪声下只随
  势垒**宽度**多项式增长 —— 这就是本算子要复用的唯一机制。

采样用 Chambers-Mallows-Stuck 表示（对称情形 beta = 0，故 V0 = 0 且无需偏移）：
  X = sin(alpha*V) / cos(V)^{1/alpha} * (cos((1-alpha)*V) / W)^{(1-alpha)/alpha}
其中 V ~ U(-pi/2, pi/2)、W ~ Exp(1)；alpha = 1 时退化为标准 Cauchy tan(V)，
alpha = 2 时化简为 2*sin(V)*sqrt(W) ~ N(0, 2)。因为 |1-alpha| <= 1，
(1-alpha)*V 始终落在 (-pi/2, pi/2)，cos 恒正，不会出现负底的分数幂。

用法：见文件末尾 __main__ 的自检（不需要 torch）。
"""

from __future__ import annotations

import math

import numpy as np

# 论文 Table 6（p27）MIS RB-[200-300] 上的默认超参
DEFAULTS = dict(alpha=1.2, tau0=0.01, lam=1.02, chains=200, steps=300, d_noise=9.0)


def sas_noise(shape, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """标准对称 alpha-stable 噪声（scale = 1），Chambers-Mallows-Stuck 采样。

    alpha == 2 时分布退化为 N(0, 2)（与论文里 (2 tau)^{1/alpha} 的系数一致，
    所以这里不额外归一化方差）；alpha == 1 时用 CMS 的极限形式。
    """
    if alpha <= 0 or alpha > 2:
        raise ValueError("alpha must be in (0, 2]")
    u = rng.uniform(-math.pi / 2.0, math.pi / 2.0, size=shape)
    if alpha == 1.0:  # beta = 0 时是标准 Cauchy
        return np.tan(u)
    w = rng.exponential(1.0, size=shape)
    return np.sin(alpha * u) / np.power(np.cos(u), 1.0 / alpha) * np.power(
        np.cos((1.0 - alpha) * u) / w, (1.0 - alpha) / alpha
    )


def truncated_sas_noise(shape, alpha: float, rng: np.random.Generator, d_noise: float) -> np.ndarray:
    """按论文 §4.1/附录 B 的做法把噪声截断到 |xi| <= d_noise（超参同名）。

    FLD 的 S-alpha-S 分布有重尾，单次采样可能出现巨大跳步；论文用对称截断保住
    “每次移动有界”，同时保留幂律尾。这里用拒绝采样实现同一语义。
    """
    out = np.empty(shape, dtype=np.float64)
    flat = out.reshape(-1)
    for i in range(flat.size):
        for _ in range(1000):
            x = sas_noise((), alpha, rng)
            if abs(x) <= d_noise:
                flat[i] = x
                break
        else:  # 极端 alpha 下拒绝率过高时的兜底（截断在边界）
            flat[i] = math.copysign(d_noise, flat[i] if flat[i] else 1.0)
    return out


def fld_step(x, grad, alpha: float, tau: float, rng: np.random.Generator, d_noise: float | None = None):
    """一步 FLD 更新（论文 §4.1 的式子的逐元素实现）。

    x     : 当前点（任意形状的 float 数组，对应连续松弛的节点打分/概率）
    grad  : grad U(x)，与 x 同形状（MIS 松弛里由打分网络或能量函数给出）
    tau   : 步长（论文用循环/退火表 tau_t = tau0 * lam^t）
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    xi = sas_noise(x.shape, alpha, rng) if d_noise is None else truncated_sas_noise(
        x.shape, alpha, rng, d_noise
    )
    return x - tau * g + math.pow(2.0 * tau, 1.0 / alpha) * xi


def anneal_schedule(steps: int, tau0: float = DEFAULTS["tau0"], lam: float = DEFAULTS["lam"]):
    """论文 Table 6 的 tau_t = tau0 * lam^t（lam = 1.02，MIS 配置）。"""
    return [tau0 * (lam ** t) for t in range(steps)]


def _selftest() -> None:
    rng = np.random.default_rng(0)
    # 1) alpha = 2 应给出对称、方差 ~2 的分布
    z = sas_noise((200000,), 2.0, rng)
    print(f"alpha=2.0  mean={z.mean():+.4f}  var={z.var():.3f}  (期望 mean~0, var~2)")
    # 2) alpha = 1.2 的尾部应显著重于高斯（用 99.9 分位对比）
    z2 = sas_noise((200000,), 1.2, rng)
    print(f"alpha=1.2  q99.9={np.quantile(z2, 0.999):.2f}  q99.9(|N(0,2)|)="
          f"{np.quantile(np.abs(rng.normal(0, math.sqrt(2), 200000)), 0.999):.2f}")
    # 3) 双井势 U(x) = (x^2 - 1)^2，比较不同 (alpha, tau) 的跨井逃逸。
    #    按论文设置做截断（d_noise = 9）并把位置反射在 [-2.5, 2.5] 内，否则 alpha<2 的
    #    无界尾会让 x 溢出（这正是论文要截断的原因）。
    #    注意：噪声幅度是 (2 tau)^{1/alpha}，所以“同 tau”并不等于“同步长”。论文每个
    #    数据集各自调 tau0，这里额外给一组“同步长”（tau_alpha 使幅度 = 0.1414）作对照。
    for alpha, tau in ((2.0, 0.01), (1.2, 0.01), (1.2, 0.1414 ** 1.2 / 2.0)):
        first, jumps = [], 0
        for _ in range(200):
            y = -1.0
            for t in range(400):
                xi = truncated_sas_noise((), alpha, rng, 9.0)
                y = y - tau * 4.0 * (y ** 3 - y) + math.pow(2.0 * tau, 1.0 / alpha) * xi
                if abs(y) > 2.5:  # 反射边界，只为让试验有界
                    y = math.copysign(2.5, y)
                if y > 0:  # 第一次落到另一个井
                    first.append(t + 1)
                    jumps += int(abs(xi) > 0.5)  # 单步大跳（高斯 3 sigma 约 0.42）
                    break
        med = sorted(first)[len(first) // 2] if first else None
        print(
            f"alpha={alpha} tau={tau:.4f}: 400 步内跨井 {len(first)}/200, 中位首次跨井步 = {med}, "
            f"单步大跳贡献 {jumps}/{len(first)}"
        )


if __name__ == "__main__":
    _selftest()
