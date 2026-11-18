# 生成模型的数学理论

一套中文生成模型讲义，从概率路径出发，逐步推导模型的训练目标、生成动力学和适用条件。主线涵盖 DDPM、DDIM、Score SDE、CNF、Flow Matching / CFM、Rectified Flow、Schrödinger Bridge 与 Stochastic Interpolants。

[章节目录](#contents) · [已完成](#done) · [待办](#todo) · [核心推导](#derivations) · [完整学习大纲](docs/study-plan.md) · [PDF 讲义](output/pdf/model-study.pdf)

[**直接在 GitHub 阅读第 0 章 →**](docs/chapter-0.md)

<a name="contents"></a>

## 章节目录

| 章 | 主题 | 学习入口 | 进度 |
| --- | --- | --- | --- |
| 0 | 统一数学基础与符号体系 | [网页正文](docs/chapter-0.md) · [TeX](chapters/00-foundations.tex) | 初稿完成 |
| 1 | 各生成模型的训练—生成整体框架 | [详细大纲](docs/study-plan.md#part-1) · [TeX](chapters/01-model-overview.tex) | 目录已完成 |
| 2 | 训练阶段的理论与数学推导 | [详细大纲](docs/study-plan.md#part-2) · [TeX](chapters/02-training.tex) | 目录已完成 |
| 3 | 生成阶段的理论与数学推导 | [详细大纲](docs/study-plan.md#part-3) · [TeX](chapters/03-generation.tex) | 目录已完成 |
| 4 | Score、Velocity、Drift 的计算、估计与相互转换 | [详细大纲](docs/study-plan.md#part-4) · [TeX](chapters/04-score-velocity-drift.tex) | 目录已完成 |
| 5 | 模型统一理论、数学可解性与研究拓展 | [详细大纲](docs/study-plan.md#part-5) · [TeX](chapters/05-unified-theory.tex) | 目录已完成 |

**第 0 章按节阅读**

- [0.1 概率论与条件概率](docs/chapter-0.md#sec-0-1)
- [0.2 确定性动力系统与概率输运](docs/chapter-0.md#sec-0-2)
- [0.3 随机过程与随机微分方程](docs/chapter-0.md#sec-0-3)
- [0.4 Fokker–Planck 方程](docs/chapter-0.md#sec-0-4)
- [0.5 最优传输与变分方法](docs/chapter-0.md#sec-0-5)
- [0.6 统一时间与符号约定](docs/chapter-0.md#sec-0-6)

<a name="time"></a>

## 全书的时间约定

| 坐标 | 起点 | 终点 |
| --- | --- | --- |
| 加噪 $t$ | $t=0$，原始数据 | $t=1$，噪声端点 |
| 生成 $\tau=1-t$ | $\tau=0$，噪声先验 | $\tau=1$，生成样本 |

加噪速度记为 $u_t$，生成速度为 $v_\tau=-u_{1-\tau}$。Flow Matching 原论文的时间对应这里的 $\tau$。有限步 DDPM 和有限时间 VP 扩散的实际终止分布可能只近似指定先验。详见 [0.6.1 端点分布](docs/chapter-0.md#sec-0-6-1) 和 [0.6.2 原论文时间方向](docs/chapter-0.md#sec-0-6-2)。

<a name="done"></a>

## 已完成

- [x] 建立第 0–5 章三级目录，共 47 节、243 个小节。
- [x] 完成第 0 章 6 节、30 个小节的正文初稿，包括推导、例题、边注和参考文献。
- [x] 统一数据端、噪声端、生成时间及 Score、Drift 和 Velocity 的记号。
- [x] 配置中文 LaTeX 排版和边注，默认使用 TeX Live 字体。
- [x] 完成第 0 章 XeLaTeX 构建、版面及交叉引用检查，并校验 16 项代表性公式。
- [x] 发布第 0 章的 GitHub 网页阅读版，保留公式编号和交叉引用。

<a name="todo"></a>

## 待办

- [ ] **第 1 章**　整理八条模型主线的训练目标、网络预测量和生成流程。
- [ ] **第 2 章**　推导 DDPM 后验与 ELBO、Score Matching、CFM 和桥接优化目标。
- [ ] **第 3 章**　推导 DDPM、DDIM 的采样递推，梳理反向 SDE、概率流 ODE 与离散误差。
- [ ] **第 4 章**　推导 Score、噪声预测、速度和漂移的转换，明确成立条件。
- [ ] **第 5 章**　统一比较模型，补充可解性、误差分析、各向异性扩散和条件生成。
- [ ] 逐章复核公式假设与论文记号，同步更新网页正文和 PDF。

<a name="derivations"></a>

## 核心推导

第 0 章给出数学基础。A–F 的模型推导安排和专题边界见完整大纲。

| 推导 | 第 0 章阅读入口 | 后续专题 |
| --- | --- | --- |
| A　DDPM 任意时间步的闭式加噪 | [离散方差递推](docs/chapter-0.md#sec-0-6-2) | [推导 A](docs/study-plan.md#derivation-a) |
| B　从噪声回归到边缘 Score | [高斯加噪与去噪](docs/chapter-0.md#sec-0-1-4) | [推导 B](docs/study-plan.md#derivation-b) |
| C　从 Fokker–Planck 到概率流 ODE | [概率流 ODE 的推导](docs/chapter-0.md#sec-0-4-5) | [推导 C](docs/study-plan.md#derivation-c) |
| D　从条件速度回归到边缘速度 | [线性 Flow Matching 的例子](docs/chapter-0.md#sec-0-6-4) | [推导 D](docs/study-plan.md#derivation-d) |
| E　各向异性扩散下的概率流速度 | [状态依赖扩散矩阵](docs/chapter-0.md#sec-0-4-4) | [推导 E](docs/study-plan.md#derivation-e) |
| F　Schrödinger Bridge 的控制能量 | [从路径 KL 到二次控制能量](docs/chapter-0.md#sec-0-5-5) | [推导 F](docs/study-plan.md#derivation-f) |

<a name="build"></a>

## 本地编译

在装有 XeLaTeX 和 Latexmk 的 TeX Live 环境中，从项目根目录运行。

```sh
latexmk -xelatex main.tex
```

[main.tex](main.tex) 是讲义入口。[references.tex](references.tex) 收录正文引用，[SOURCES.md](SOURCES.md) 记录 Lumenbook 模板来源。
