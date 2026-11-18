# 生成模型的数学理论

一套中文生成模型讲义，从概率路径出发，逐步推导模型的训练目标、生成动力学和适用条件。主线涵盖 DDPM、DDIM、Score SDE、CNF、Flow Matching / CFM、Rectified Flow、Schrödinger Bridge 与 Stochastic Interpolants。

[章节目录](#contents) · [已完成](#done) · [待办](#todo) · [核心推导](#derivations) · [完整学习大纲](docs/study-plan.md) · [PDF 讲义](output/pdf/model-study.pdf)

**统一框架快速版**　[网页速览](docs/quick-guide.md) · [六页 PDF](output/pdf/model-study-quick.pdf) · [LaTeX 源码](quick-guide.tex)。按概率路径、条件回归、场转换、生成动力学和误差条件串联第 0–5 章。

**网页正文**　[第 0 章 数学基础](docs/chapter-0.md) · [第 1 章 训练与生成框架](docs/chapter-1.md) · [第 2 章 训练理论](docs/chapter-2.md) · [第 3 章 生成理论](docs/chapter-3.md) · [第 4 章 向量场的计算与转换](docs/chapter-4.md) · [第 5 章 统一理论与研究拓展](docs/chapter-5.md)

**网页公式维护**　[检查结果与编写规范](docs/MARKDOWN-REVIEW.md)。公式编号置于数学块外，保留原有编号和跳转链接，以兼容 GitHub 的浏览器数学排版。

<a name="contents"></a>

## 章节目录

| 章 | 主题 | 学习入口 | 进度 |
| --- | --- | --- | --- |
| 0 | 统一数学基础与符号体系 | [网页正文](docs/chapter-0.md) · [TeX](chapters/00-foundations.tex) | 初稿完成 |
| 1 | 各生成模型的训练—生成整体框架 | [网页正文](docs/chapter-1.md) · [详细大纲](docs/study-plan.md#part-1) · [TeX](chapters/01-model-overview.tex) | 初稿完成 |
| 2 | 训练阶段的理论与数学推导 | [网页正文](docs/chapter-2.md) · [详细大纲](docs/study-plan.md#part-2) · [TeX](chapters/02-training.tex) | 初稿完成 |
| 3 | 生成阶段的理论与数学推导 | [网页正文](docs/chapter-3.md) · [详细大纲](docs/study-plan.md#part-3) · [TeX](chapters/03-generation.tex) | 初稿完成 |
| 4 | Score、Velocity、Drift 的计算、估计与相互转换 | [网页正文](docs/chapter-4.md) · [详细大纲](docs/study-plan.md#part-4) · [TeX](chapters/04-score-velocity-drift.tex) | 初稿完成 |
| 5 | 模型统一理论、数学可解性与研究拓展 | [网页正文](docs/chapter-5.md) · [详细大纲](docs/study-plan.md#part-5) · [TeX](chapters/05-unified-theory.tex) | 初稿完成 |

**第 0 章按节阅读**

- [0.1 概率论与条件概率](docs/chapter-0.md#sec-0-1)
- [0.2 确定性动力系统与概率输运](docs/chapter-0.md#sec-0-2)
- [0.3 随机过程与随机微分方程](docs/chapter-0.md#sec-0-3)
- [0.4 Fokker–Planck 方程](docs/chapter-0.md#sec-0-4)
- [0.5 最优传输与变分方法](docs/chapter-0.md#sec-0-5)
- [0.6 统一时间与符号约定](docs/chapter-0.md#sec-0-6)

**第 1 章按模型阅读**

[DDPM](docs/chapter-1.md#sec-1-1) · [DDIM](docs/chapter-1.md#sec-1-2) · [Score SDE](docs/chapter-1.md#sec-1-3) · [CNF](docs/chapter-1.md#sec-1-4) · [FM / CFM](docs/chapter-1.md#sec-1-5) · [Rectified Flow / OT-CFM](docs/chapter-1.md#sec-1-6) · [Schrödinger Bridge](docs/chapter-1.md#sec-1-7) · [Stochastic Interpolants](docs/chapter-1.md#sec-1-8)

**第 2 章按节阅读**

[DDPM](docs/chapter-2.md#sec-2-1) · [DDIM](docs/chapter-2.md#sec-2-2) · [Score Matching](docs/chapter-2.md#sec-2-3) · [FM / CFM](docs/chapter-2.md#sec-2-4) · [Rectified Flow / OT-CFM](docs/chapter-2.md#sec-2-5) · [Schrödinger Bridge](docs/chapter-2.md#sec-2-6) · [Stochastic Interpolants](docs/chapter-2.md#sec-2-7) · [训练目标对照](docs/chapter-2.md#sec-2-8)

**第 3 章按节阅读**

[DDPM](docs/chapter-3.md#sec-3-1) · [DDIM](docs/chapter-3.md#sec-3-2) · [反向 SDE 与概率流 ODE](docs/chapter-3.md#sec-3-3) · [CNF / FM](docs/chapter-3.md#sec-3-4) · [Rectified Flow](docs/chapter-3.md#sec-3-5) · [Schrödinger Bridge](docs/chapter-3.md#sec-3-6) · [Stochastic Interpolants](docs/chapter-3.md#sec-3-7) · [数值分析](docs/chapter-3.md#sec-3-8) · [质量与复杂度](docs/chapter-3.md#sec-3-9)

**第 4 章按节阅读**

[Score 基础](docs/chapter-4.md#sec-4-1) · [Score Matching](docs/chapter-4.md#sec-4-2) · [噪声预测与 Tweedie 公式](docs/chapter-4.md#sec-4-3) · [Velocity](docs/chapter-4.md#sec-4-4) · [Drift 与扩散修正](docs/chapter-4.md#sec-4-5) · [FM 与 Score 的转换](docs/chapter-4.md#sec-4-6) · [桥过程](docs/chapter-4.md#sec-4-7) · [计算实现与验证](docs/chapter-4.md#sec-4-8)

**第 5 章按节阅读**

[输运与扩散统一](docs/chapter-5.md#sec-5-1) · [模型比较](docs/chapter-5.md#sec-5-2) · [随机插值](docs/chapter-5.md#sec-5-3) · [可解性与稳定性](docs/chapter-5.md#sec-5-4) · [误差与收敛](docs/chapter-5.md#sec-5-5) · [矩阵扩散](docs/chapter-5.md#sec-5-6) · [耦合与物理约束](docs/chapter-5.md#sec-5-7) · [条件生成与研究应用](docs/chapter-5.md#sec-5-8)

<a name="time"></a>

## 全书的时间约定

| 坐标 | 起点 | 终点 |
| --- | --- | --- |
| 加噪 $`t`$ | $`t=0`$，原始数据 | $`t=1`$，噪声端点 |
| 生成 $`\tau=1-t`$ | $`\tau=0`$，噪声先验 | $`\tau=1`$，生成样本 |

加噪速度记为 $`u_t`$，生成速度为 $`v_\tau=-u_{1-\tau}`$。Flow Matching 原论文的时间对应这里的 $`\tau`$。有限步 DDPM 和有限时间 VP 扩散的实际终止分布可能只近似指定先验。详见 [0.6.1 端点分布](docs/chapter-0.md#sec-0-6-1) 和 [0.6.2 原论文时间方向](docs/chapter-0.md#sec-0-6-2)。

<a name="done"></a>

## 已完成

- [x] 建立第 0–5 章三级目录，共 47 节、243 个小节。
- [x] 完成第 0 章 6 节、30 个小节的正文初稿，包括推导、例题、边注和参考文献。
- [x] 完成第 1 章 8 节、40 个小节的正文初稿，以及八套训练与生成流程和模型对照。
- [x] 完成第 2 章 8 节、44 个小节的正文初稿，包括变分训练目标、条件回归证明、桥接优化与章末练习。
- [x] 完成第 3 章 9 节、48 个小节的正文初稿，包括采样递推、时间反演、概率流、数值分析及 8 道核对练习。
- [x] 完成第 4 章 8 节、41 个小节的正文初稿，包括向量场转换、8 个算例、10 道核对练习及公式速查表。
- [x] 完成第 5 章 8 节、40 个小节的正文初稿，包括可解性、误差分析、9 个算例、12 道核对练习及模型条件对照表。
- [x] 统一数据端、噪声端、生成时间及 Score、Drift 和 Velocity 的记号。
- [x] 配置中文 LaTeX 排版和边注，默认使用 TeX Live 字体。
- [x] 完成第 0 章 XeLaTeX 构建、版面及交叉引用检查，并校验 16 项代表性公式。
- [x] 整理第 0–5 章的网页阅读版，保留公式编号和交叉引用。
- [x] 完成第 3 章的 18 项公式与数值自检、章节结构和本地链接检查，以及 PDF 新增页面的版面检查。
- [x] 完成第 4 章的 35 项公式与数值自检、章节结构和本地链接检查，以及本章 16 页正文的 PDF 版面检查。
- [x] 完成第 5 章的 36 项公式与数值自检、章节结构和本地链接检查，以及本章 16 页正文的 PDF 版面检查。
- [x] 整理第 5 部分的统一框架快速版，提供六页 PDF、网页速览及完整章节定位表。

第 3–5 章当前为正文初稿，独立数学审阅待完成。数值与符号自检覆盖代表性算例，通用结论的假设写在对应小节中。第 5 章的低正则叠加原理作为引用定理使用，物理应用按有限维模型展开。

<a name="todo"></a>

## 待办

- [x] **第 3 章**　推导 DDPM、DDIM 的采样递推，梳理反向 SDE、概率流 ODE 与离散误差。
- [x] **第 4 章**　推导 Score、噪声预测、速度和漂移的转换，明确成立条件。
- [x] **第 5 章**　统一比较模型，补充可解性、误差分析、各向异性扩散和条件生成。
- [ ] 逐章复核公式假设与论文记号，同步更新网页正文和 PDF。

<a name="derivations"></a>

## 核心推导

第 0 章给出数学基础。A–F 的模型推导安排和专题边界见完整大纲。

| 推导 | 第 0 章阅读入口 | 后续专题 |
| --- | --- | --- |
| A　DDPM 任意时间步的闭式加噪 | [离散方差递推](docs/chapter-0.md#sec-0-6-2) | [第 2.1.2 小节](docs/chapter-2.md#sec-2-1-2) |
| B　从噪声回归到边缘 Score | [高斯加噪与去噪](docs/chapter-0.md#sec-0-1-4) | [第 2.3.3 小节](docs/chapter-2.md#sec-2-3-3) |
| C　从 Fokker–Planck 到概率流 ODE | [概率流 ODE 的推导](docs/chapter-0.md#sec-0-4-5) | [第 3.3.3 小节](docs/chapter-3.md#sec-3-3-3) |
| D　从条件速度回归到边缘速度 | [线性 Flow Matching 的例子](docs/chapter-0.md#sec-0-6-4) | [第 2.4.5 小节](docs/chapter-2.md#sec-2-4-5) |
| E　各向异性扩散下的概率流速度 | [状态依赖扩散矩阵](docs/chapter-0.md#sec-0-4-4) | [第 4.5.4 小节](docs/chapter-4.md#sec-4-5-4) |
| F　Schrödinger Bridge 的控制能量 | [从路径 KL 到二次控制能量](docs/chapter-0.md#sec-0-5-5) | [第 2.6.3 小节](docs/chapter-2.md#sec-2-6-3) |

<a name="build"></a>

## 本地编译

在装有 XeLaTeX 和 Latexmk 的 TeX Live 环境中，从项目根目录运行。

```sh
latexmk -xelatex main.tex
```

[main.tex](main.tex) 是讲义入口。[references.tex](references.tex) 收录正文引用，[SOURCES.md](SOURCES.md) 记录 Lumenbook 模板来源。

修改网页正文后，运行 Markdown 数学检查。

```sh
python scripts/check-markdown-math.py
```
