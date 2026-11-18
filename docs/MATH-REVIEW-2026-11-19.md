# 第 0–5 章数学复核记录

2026-11-19 · Codex 对初稿进行逐章复核。范围覆盖 47 节的主要推导、适用条件、典型算例及论文记号。改变结论或条件的修订均写入网页正文和 LaTeX，速览同步更新。引用的反演、最优传输和叠加定理沿用各自的定理假设。外部数学审阅尚未进行。

## 分章核对

| 章节 | 核对内容 | 本轮修订 |
| --- | --- | --- |
| [0 数学基础](chapter-0.md) | 条件投影、Gaussian Bayes、ELBO、CNF 散度符号、反向生成元、Fokker–Planck、动能与路径熵、两种时钟 | 在 [0.5.5](chapter-0.md#sec-0-5-5) 明确指数密度可由规范状态路径重建，补充 Wasserstein 二次运输的矩条件，以及 [0.6.2](chapter-0.md#sec-0-6-2) 的 DDPM 与 DDIM 系数对照。 |
| [1 模型框架](chapter-1.md) | 八类模型的训练标签、采样方向、端点分布和条件耦合 | 在 [1.3.2](chapter-1.md#sec-1-3-2) 区分干净初值上的 VE 增量方差与含初始平滑的总扰动方差，在 [1.5](chapter-1.md#sec-1-5) 给出原 FM 的条件 OT 路径换算。 |
| [2 训练推导](chapter-2.md) | 后验完平方、ELBO 权重、DSM/CFM 投影、Reflow 两项方差、Doob 变换、IPF 与 DSB 监督、随机插值目标 | 补充 [DDIM 正权重证明](chapter-2.md#sec-2-2-3) 和 [时间加权条件](chapter-2.md#sec-2-3-5)。在 [2.7.5](chapter-2.md#sec-2-7-5) 区分标签平方可积时的 MSE 与去掉标签平方项的二次泛函。
| [3 生成推导](chapter-3.md) | DDPM 反向核矩、DDIM 跳步方差和 Euler 极限、时间反演、桥漂移、随机插值、数值阶与误差方向 | 统一桥和插值的概率律、密度及端点表示。在 [3.9.2](chapter-3.md#sec-3-9-2) 写明路径 KL 的两个参数及积分期望所用的精确律。 |
| [4 向量场](chapter-4.md) | Stein 与 Hyvärinen 分部积分、Tweedie 后验矩、扩散预测旋转、速度可逆条件、状态依赖扩散、桥势和 Hutchinson 迹 | 明确 [4.2.1](chapter-4.md#sec-4-2-1) 中 Score 识别密度所需的共同连通支持与归一化条件，并在 [4.6.5](chapter-4.md#sec-4-6-5) 补充线性 SDE 的时间可积性和子区间初值。 |
| [5 统一理论](chapter-5.md) | 共享边缘、叠加原理、Lipschitz 稳定性、路径 KL、矩阵扩散、受约束 SDE、后验采样和 guidance | 明确 [Euler 路径 KL 的冻结误差条件](chapter-5.md#sec-5-5-3) 和 [后验似然的条件独立性](chapter-5.md#sec-5-8-2)。直接计算 [CFG 倾斜密度的前向方程残差](chapter-5.md#sec-5-8-3)，并将 [仿射约束的辅助噪声](chapter-5.md#sec-5-8-4) 放在核空间中。 |

## 原论文记号与推导位置

| 原论文 | 对照结果 |
| --- | --- |
| [Ho 等，DDPM，式 (4)、(7)、(11)、(12)、(14) 与算法 2](https://arxiv.org/pdf/2006.11239v2) | 单步 $`\alpha_k`$ 对应 $`r_k`$，累乘 $`\bar\alpha_k`$ 对应 $`A_k^2`$。后验方差、固定反向方差的变分权重和简化噪声 MSE 分别对应第 2 章的后验、加权目标和简化目标。最后一步的解码器与直接输出均值的实现各有定义。 |
| [Song、Meng 与 Ermon，DDIM，式 (6)、(12)、(16)、定理 1](https://arxiv.org/pdf/2010.02502v4) | 原文用 $`\alpha_k`$ 表示累乘量，用 $`\sigma_k`$ 表示反向步标准差。书中相邻步用 $`\delta_k`$，跨步用 $`\delta_{k,j}`$；第 1 章用 $`\omega_{k,j}`$ 表示同一类反向标准差。正方差中间步可用高斯 KL 推导，零方差采样可复用边缘回归网络。 |
| [Song 等，Score SDE，式 (6)、(13)、(16)、(17)、(29)–(31)](https://arxiv.org/html/2011.13456v2) | 原论文的反向 SDE 用递减原时间。改为递增生成时间后漂移整体取负，扩散协方差按时间重标记。矩阵散度在概率流和反向漂移里分别带半系数和全系数。sub-VP 的条件方差是条件标准差的平方。 |
| [Chen 等，Neural ODE，定理 1](https://arxiv.org/pdf/1806.07366v1) · [FFJORD，第 2–3 节](https://arxiv.org/abs/1810.01367) | 瞬时对数密度导数为负散度。由数据向先验积分时，用先验处对数密度加散度积分。固定轨迹上的随机迹估计使用二阶矩恒等式。 |
| [Lipman 等，FM，定理 1–3、式 (20)–(22)](https://arxiv.org/pdf/2210.02747v2) · [Tong 等，OT-CFM，第 3.2.3 节](https://arxiv.org/pdf/2302.00482v4) | 单端点条件路径的后验平均给出其边缘速度，原作者的生成时间对应本书的 $`\tau`$。条件高斯上的 OT 路径与两端点 OT 耦合的优化对象不同。 |
| [Liu 等，Rectified Flow，定理 3.3、3.5、3.7](https://arxiv.org/pdf/2209.03003v1) | 生成方向以噪声为源、数据为目标。准确输运边缘后，两个 Jensen 不等式给出凸代价下降；Reflow 的平方位移恒等式将下降量分成条件位移方差与流上的速度偏差。 |
| [Léonard，Schrödinger 问题，第 2–3 节](https://arxiv.org/pdf/1308.0215) · [De Bortoli 等，DSB，第 3 节及算法 1](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html) | 端点 KL 加条件桥 KL，桥势乘积以参考路径律为基准。DSB 的理想 IPF 半步分别重置终点和起点。书中用步长 $`h_k`$ 对应原文 $`\gamma_k`$，核协方差为 $`2h_kI_d`$。均值回归公式对应离散 Euler 近似。 |
| [Albergo 等，Stochastic Interpolants，定义 1、定理 6–8、推论 10](https://jmlr.org/papers/v26/23-1605.html) | 原文 $`b`$ 是输运速度，对应 $`u_t`$。原文递减时钟的 $`b_B=b-\epsilon s`$ 对应生成漂移 $`-u_{1-\tau}+\kappa(1-\tau)s_{1-\tau}`$。论文要求平滑的量之一是 $`\gamma^2`$；其端点允许出现 $`\dot\gamma`$ 奇异，需检查具体平方风险。 |
| [Hyvärinen，Score Matching，第 2 节](https://www.jmlr.org/papers/v6/hyvarinen05a.html) · [Song 等，Sliced Score Matching，第 2–3 节](https://proceedings.mlr.press/v115/song20a.html) · [Salimans 与 Ho，第 4 节](https://arxiv.org/pdf/2202.00512v2) | 分部积分需要边界通量条件。投影方向二阶矩为单位阵时，投影风险等于完整 Score 风险。扩散 $`v`$ 标签是信号和噪声的正交旋转，乘路径角速度且条件平均后才给出输运速度。 |
| [Trevisan，定理 2.5](https://arxiv.org/pdf/1507.01357v2) · [Chung 等，DPS，第 3 节](https://arxiv.org/pdf/2209.14687v4) · [Ho 与 Salimans，CFG，式 (6)](https://arxiv.org/pdf/2207.12598) | 叠加定理把弱 Fokker–Planck 解提升为鞅问题的路径律。DPS 以去噪均值代入似然近似后验积分。CFG 的原文外推权重加一，得到本书以无条件分支为基准的权重。 |

## 交付检查

- 网页与 LaTeX 逐一匹配 171 个编号公式的标签、编号和数学内容。Markdown 数学解析及本地链接检查覆盖六章、速览和大纲。
- 复算模型关键恒等式和代表性算例。覆盖离散后验和反向方差、Gauss/OU、状态依赖矩阵扩散、路径方向、回归、桥与条件生成。
- 分别从 `main.tex` 与 `quick-guide.tex` 编译完整讲义（101 页）和速览（6 页）。检查引文、交叉引用、字符、页流和版面。
