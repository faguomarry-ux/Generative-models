# 学习大纲与阅读路线

[返回项目目录](../README.md#contents) · [阅读第 0 章](chapter-0.md) · [核心推导计划](#derivations)

[Part 0 数学基础](#part-0) · [Part 1 模型框架](#part-1) · [Part 2 训练推导](#part-2) · [Part 3 生成推导](#part-3) · [Part 4 场的计算与转换](#part-4) · [Part 5 统一理论](#part-5)

结合你目前希望研究的 Flow Matching、Fokker–Planck 方程、Score 表示以及随机动力系统，我建议将这次学习组织成一套**以数学理论为主线、以模型为研究对象、以推导为核心、以模型之间的统一性为最终目标**的知识体系。

你原本提出的四个部分是合理的，但还不足以完整理解这些生成模型背后的理论联系。

我的建议是：

- **模型范围**：保留你提出的模型，补充 Score-based SDE、Conditional Flow Matching、Rectified Flow 和 Stochastic Interpolants 等关键框架。
- **知识结构**：由原来的四部分扩展为六部分，其中前四部分保留你的核心思路，增加数学基础和理论统一性两部分。
- **学习深度**：采用一级目录—二级目录—三级目录，明确每一节需要理解的概念和必须独立完成的数学推导。

## **一、现有模型是否充分？还应该补充哪些模型？**

你目前提出：

1. Flow Matching
2. DDPM
3. DDIM
4. Schrödinger Bridge（薛定谔桥）
5. Diffusion Models

这五个名称涵盖了生成建模中的几个重要方向，但它们**不处于完全相同的理论层级**。

例如，Diffusion Models 是一个较宽泛的模型家族，DDPM 属于其中的典型模型，而 DDIM 与 DDPM 共享常用训练目标，但采用不同的生成过程。Flow Matching 则主要是一种学习连续时间速度场的训练方法。

### 1.1 推荐的模型体系

**概率分布演化与生成建模**

Probability Distribution Evolution

按照理论机制分类（并非互斥分类）

**Diffusion / Score**

扩散与分数估计

DDPM

DDIM

Score-based SDE

EDM

**Flow / Transport**

确定性输运与速度场

CNF

Flow Matching / CFM

Rectified Flow

OT-CFM

**Bridge / Control**

端点约束与随机控制

Schrödinger Bridge

Diffusion SB

IPF / DSBM

**Unifying Frameworks**

统一概率路径与动力学

Stochastic Interpolants

Probability Flow ODE

ODE / SDE 统一

注意：这些分类存在交叉。例如 Probability Flow ODE 是将某类扩散过程对应的边缘分布演化表示成确定性 ODE 的方法，而不是一个独立于扩散模型的模型家族。

### 1.2 应该增加的核心模型与框架

| 模型或框架                                | 优先级 | 为什么需要学习                                          |
| ----------------------------------------- | ------ | ------------------------------------------------------- |
| Score-based Generative Models / Score SDE | 必学   | 理解 Score、反向 SDE、Fokker–Planck 和概率流 ODE 的关键 |
| Continuous Normalizing Flow（CNF）        | 必学   | 理解 Flow Matching 为什么可以通过学习速度场构建生成模型 |
| Conditional Flow Matching（CFM）          | 必学   | 解释为何不需要知道边缘概率密度，就能通过条件样本训练    |
| Rectified Flow                            | 必学   | 理解直线路径、速度场回归、轨迹整直和快速生成            |
| Stochastic Interpolants                   | 必学   | 将 Flow、Diffusion、Score 放进共同的概率路径理论        |
| Optimal Transport / OT-CFM                | 重点   | 理解样本耦合、运输成本、路径设计与生成效率              |
| EDM（扩散设计空间）                       | 建议   | 将噪声参数化、预条件化和数值采样策略分开理解            |
| Consistency Models                        | 进阶   | 理解一步或少步生成与自洽性约束                          |
| Diffusion Schrödinger Bridge              | 重点   | 把传统薛定谔桥与基于 Score 的生成模型联系起来           |

其中 Stochastic Interpolants 尤其值得加入。Albergo 等人的工作从随机插值出发，同时构造满足输运方程与 Fokker–Planck 方程的概率演化，为理解确定性流与随机扩散之间的关系提供了直接的理论工具。

原始的 Flow Matching 论文则强调：通过对预先设计的条件概率路径上的向量场进行回归，可以无需训练期间模拟完整生成动力学而训练 CNF。

**建议将必学主线确定为 8 个理论对象：**

DDPM、DDIM、Score-based SDE、CNF、Flow Matching/CFM、Rectified Flow、Schrödinger Bridge、Stochastic Interpolants。

OT-CFM、EDM、Consistency Models 和相关快速采样算法可作为这些主线的拓展。

## **二、你的四个部分是否完整？**

你提出的四部分是：

1. 训练—生成的整体框架
2. 训练的理论知识
3. 生成的理论知识
4. Score 的计算与推导

我建议保留这些内容，但调整为六个一级模块。

**Part 0**

**统一数学基础与符号规范**

新增

概率论、随机过程、ODE/PDE、SDE、Fokker–Planck、时间方向

**Part 1**

**训练—生成的整体框架**

原有

模型目标、数据与噪声分布、概率路径、训练目标、采样流程

**Part 2**

**训练阶段的理论与数学推导**

原有

ELBO、Score Matching、Flow Matching、Schrödinger Bridge 的优化目标

**Part 3**

**生成阶段的理论与数学推导**

原有

反向马尔可夫链、反向 SDE、Probability Flow ODE、数值求解器

**Part 4**

**Score、Velocity、Drift 的计算与联系**

原有·扩展

条件与边缘 Score、Tweedie 公式、速度场、漂移场、网络参数化转换

**Part 5**

**模型的统一理论、可解性与研究拓展**

新增

连续性方程、Fokker–Planck、ODE/SDE 等价条件、误差与收敛、条件生成

增加 Part 0 和 Part 5 很重要。

Part 0 解决“公式从哪里来”的问题。Part 5 解决“不同模型之间到底有什么联系、哪些联系可以严格证明”的问题。

尤其对你所关注的理论研究而言，仅掌握某个模型的损失函数与生成算法，还不能充分判断一个新的概率路径、各向异性扩散系数或耦合漂移场是否合理。

## **三、完整的三级目录设计**

下面这套目录可以直接作为你的长期学习大纲。每个 Part 是一级模块，使用 `0.1` 表示二级目录，`0.1.1` 表示三级目录。

为了避免只学习概念而忽略数学推导，我会在重要章节标注 **★ 必须推导**。

<a name="part-0"></a>

### Part 0. 统一数学基础与符号体系

这一部分不单纯是准备知识，而是为了能够从基本数学方程独立推导生成模型。

#### 0.1 概率论与条件概率

- 0.1.1 随机变量、概率密度、联合分布与边缘分布
- 0.1.2 条件概率密度、Bayes 公式与全概率公式
- 0.1.3 数学期望、条件期望与条件期望的投影性质
- 0.1.4 高斯分布、多元高斯分布与条件高斯分布
- 0.1.5 KL 散度、相对熵、交叉熵与变分下界

**★ 必须推导**：高斯分布的条件分布、KL 散度的计算、条件期望为什么是均方误差意义下的最优预测器。

#### 0.2 确定性动力系统与概率输运

- 0.2.1 常微分方程与时间依赖向量场
- 0.2.2 流映射、轨迹、初值问题与终值问题
- 0.2.3 概率密度的 Pushforward 与变量变换
- 0.2.4 连续性方程与 Liouville 方程
- 0.2.5 Jacobian 行列式与瞬时变量变换公式

**★ 必须推导**：

```math
\frac{\partial p_t}{\partial t} +\nabla\cdot(p_t u_t)=0
```

以及

```math
\frac{d}{dt}\log p_t(X_t) =-\nabla\cdot u_t(X_t).
```

#### 0.3 随机过程与随机微分方程

- 0.3.1 Brownian Motion、Wiener 过程与马尔可夫过程
- 0.3.2 Itô 积分与 Itô 公式
- 0.3.3 随机微分方程中 Drift 与 Diffusion 的含义
- 0.3.4 SDE 的生成元与 Kolmogorov 方程
- 0.3.5 时间反演与反向随机过程

#### 0.4 Fokker–Planck 方程

- 0.4.1 从 Itô 公式推导 Fokker–Planck 方程 ★
- 0.4.2 概率流密度与守恒律
- 0.4.3 各向同性与各向异性扩散
- 0.4.4 常系数与状态依赖扩散矩阵
- 0.4.5 Fokker–Planck 方程与连续性方程之间的联系 ★

对于一般 SDE：

```math
dX_t=f(X_t,t)\,dt+G(X_t,t)\,dW_t,
```

定义 $`a=GG^\top`$，其概率密度在适当条件下满足

```math
\boxed{ \partial_t p_t =-\nabla\cdot(fp_t) +\frac12\sum_{i,j} \partial_{x_i}\partial_{x_j}(a_{ij}p_t) }
```

这将是后续研究 Score-based SDE、Schrödinger Bridge 和各向异性扩散的重要起点。

#### 0.5 最优传输与变分方法

- 0.5.1 Wasserstein 距离与运输映射
- 0.5.2 Kantorovich 最优耦合问题
- 0.5.3 Benamou–Brenier 动态最优传输
- 0.5.4 熵正则化最优传输与 Sinkhorn 算法
- 0.5.5 相对熵最小化与随机控制

#### 0.6 统一时间与符号约定

- 0.6.1 数据分布、先验分布与中间边缘分布
- 0.6.2 加噪时间与生成时间的方向
- 0.6.3 $`t,\tau,T`$ 与 $`t\leftrightarrow T-t`$ 的变换
- 0.6.4 条件标签、路径端点与样本耦合
- 0.6.5 Score、Drift、Velocity、Noise 的符号区分

加噪从 $t=0$ 的原始数据走向 $t=1$ 的噪声端点。生成用 $\tau=1-t$，从 $\tau=0$ 的先验走向 $\tau=1$ 的生成样本。

加噪时间下的速度、漂移和 Score 分别记为 $u_t(x)$、$f_t(x)$ 和 $s_t(x)$。生成速度满足 $v_\tau(x)=-u_{1-\tau}(x)$。先验与实际终止分布的区别，以及各原始论文的记号对照，见 [第 0.6 节](chapter-0.md#sec-0-6)。

<a name="part-1"></a>

### Part 1. 各生成模型的训练—生成整体框架

[阅读第 1 章正文](chapter-1.md)

本部分暂时不深入推导复杂公式，目标是完整回答：**模型要学习什么、通过什么损失训练、生成时究竟求解什么方程？**

#### 1.1 DDPM：离散扩散概率模型

- 1.1.1 原始数据、噪声先验与离散时间链
- 1.1.2 前向加噪马尔可夫过程
- 1.1.3 噪声预测网络与训练损失
- 1.1.4 反向去噪马尔可夫链
- 1.1.5 DDPM 的完整训练与生成算法

#### 1.2 DDIM：隐式扩散生成模型

- 1.2.1 DDIM 与 DDPM 的共同训练目标
- 1.2.2 非马尔可夫前向过程
- 1.2.3 确定性与随机采样参数
- 1.2.4 时间步跳跃与采样加速
- 1.2.5 DDPM 与 DDIM 的核心差异

#### 1.3 Score-based Diffusion / Score SDE

- 1.3.1 Score-based 生成模型的基本思想
- 1.3.2 VE、VP、sub-VP 等正向 SDE
- 1.3.3 Score 网络的训练目标
- 1.3.4 Reverse-time SDE 生成
- 1.3.5 Probability Flow ODE 生成

#### 1.4 Continuous Normalizing Flow

- 1.4.1 Normalizing Flow 与可逆映射
- 1.4.2 连续时间流映射
- 1.4.3 速度场参数化
- 1.4.4 CNF 的似然训练
- 1.4.5 CNF 的 ODE 采样

#### 1.5 Flow Matching 与 Conditional Flow Matching

- 1.5.1 概率路径设计
- 1.5.2 条件概率路径与条件速度场
- 1.5.3 FM 与 CFM 的损失函数
- 1.5.4 条件训练与边缘速度场之间的关系
- 1.5.5 训练阶段与生成阶段的区别

#### 1.6 Rectified Flow 与 OT-CFM

- 1.6.1 线性随机插值
- 1.6.2 Rectified Flow 的速度回归
- 1.6.3 Reflow 与轨迹整直
- 1.6.4 独立耦合与 OT 耦合
- 1.6.5 路径几何与生成步数的关系

#### 1.7 Schrödinger Bridge

- 1.7.1 给定初始分布与终端分布的桥接问题
- 1.7.2 参考随机过程与路径空间 KL 最小化
- 1.7.3 Schrödinger Potentials
- 1.7.4 正向与反向动力学
- 1.7.5 Diffusion Schrödinger Bridge 的训练与生成流程

#### 1.8 Stochastic Interpolants

- 1.8.1 两个端点随机变量的联合耦合
- 1.8.2 随机插值路径的定义
- 1.8.3 速度场与 Score 的学习
- 1.8.4 对应的 ODE 与 SDE 生成
- 1.8.5 与 Flow Matching 和 Diffusion Models 的关系

**本部分的最终成果**：为八个主线模型分别绘制训练—生成流程图，并统一列出：

```math
\boxed{ \text{初始分布} \rightarrow \text{概率路径} \rightarrow \text{训练目标} \rightarrow \text{学习的场} \rightarrow \text{生成动力学} }
```

<a name="part-2"></a>

### Part 2. 训练阶段的理论与数学推导

[阅读第 2 章正文](chapter-2.md)

这一部分是生成模型学习的核心之一。

重点不仅是知道损失函数的表达式，而是理解损失函数如何从概率模型、条件期望或变分原理中得到。

#### 2.1 DDPM 的训练理论

- 2.1.1 前向马尔可夫链的联合概率分解
- 2.1.2 $`q(x_k\mid x_0)`$ 的闭式推导 ★
- 2.1.3 $`q(x_{k-1}\mid x_k,x_0)`$ 的后验推导 ★
- 2.1.4 ELBO 的构造、KL 项分解与化简 ★
- 2.1.5 从 ELBO 到噪声预测 MSE 损失 ★
- 2.1.6 $`\epsilon`$-prediction、$`x_0`$-prediction 与 $`v`$-prediction

#### 2.2 DDIM 的训练理论

- 2.2.1 共享 DDPM 边缘加噪分布的条件
- 2.2.2 非马尔可夫前向联合分布的构造
- 2.2.3 DDIM 训练目标与 DDPM 简化损失的联系
- 2.2.4 不同路径与相同边缘分布的区别
- 2.2.5 采样方式改变是否要求重新训练

#### 2.3 Score-based Models 的训练理论

- 2.3.1 Fisher Divergence 与 Score Matching
- 2.3.2 Hyvärinen Score Matching 的推导 ★
- 2.3.3 Denoising Score Matching 的推导 ★
- 2.3.4 多噪声尺度与连续时间损失
- 2.3.5 时间权重与网络优化目标

#### 2.4 Flow Matching 的训练理论

- 2.4.1 连续性方程约束下的速度场学习
- 2.4.2 概率路径 $`p_t`$ 的指定
- 2.4.3 边缘速度场与条件速度场
- 2.4.4 Flow Matching 理想损失的推导
- 2.4.5 Conditional Flow Matching 损失的推导 ★
- 2.4.6 CFM 与 FM 梯度等价的证明 ★

#### 2.5 Rectified Flow 与 OT-CFM 的训练理论

- 2.5.1 线性插值的条件速度目标
- 2.5.2 直线插值与真实 ODE 轨迹的区别
- 2.5.3 Rectified Flow 的条件期望表达式 ★
- 2.5.4 轨迹整直的理论与 Reflow
- 2.5.5 最优耦合与训练目标之间的关系

#### 2.6 Schrödinger Bridge 的训练理论

- 2.6.1 路径空间相对熵的定义
- 2.6.2 Schrödinger Bridge 约束优化问题
- 2.6.3 Girsanov 定理与漂移控制代价 ★
- 2.6.4 Schrödinger 系统及其势函数
- 2.6.5 Iterative Proportional Fitting（IPF）
- 2.6.6 Diffusion Schrödinger Bridge 的迭代训练

#### 2.7 Stochastic Interpolants 的训练理论

- 2.7.1 随机插值函数及端点条件
- 2.7.2 插值变量的时间导数
- 2.7.3 速度场的条件期望表示 ★
- 2.7.4 Score 的条件回归表示
- 2.7.5 二次回归损失与其最优解 ★
- 2.7.6 不同耦合与随机插值系数的影响

#### 2.8 训练目标之间的等价性与差异

- 2.8.1 噪声预测与 Score Matching
- 2.8.2 Score Matching 与 Denoising Score Matching
- 2.8.3 Flow Matching 与 Conditional Flow Matching
- 2.8.4 Drift Matching、Velocity Matching 与 Score Matching
- 2.8.5 各类训练目标的适用条件与不等价情形

<a name="part-3"></a>

### Part 3. 生成阶段的理论与数学推导

[阅读第 3 章正文](chapter-3.md)

训练解决的是如何估计一个函数或向量场；生成解决的是如何使用这个函数或向量场，将简单分布演化为目标分布。

这一部分需要将概率论、ODE、SDE 和数值计算统一起来。

#### 3.1 DDPM 的生成理论

- 3.1.1 反向条件概率的形式
- 3.1.2 反向均值的推导 ★
- 3.1.3 反向方差与随机噪声
- 3.1.4 完整去噪递推公式 ★
- 3.1.5 DDPM 生成过程与 Langevin Dynamics 的联系

#### 3.2 DDIM 的生成理论

- 3.2.1 DDIM 采样更新公式的推导 ★
- 3.2.2 随机参数 $`\eta`$ 的作用
- 3.2.3 $`\eta=0`$ 时的确定性生成
- 3.2.4 时间步子序列与跳步采样
- 3.2.5 DDIM 与 Probability Flow ODE 的连续时间联系

#### 3.3 Score-based SDE 的生成理论

- 3.3.1 反向时间扩散过程的数学条件
- 3.3.2 Reverse-time SDE 的推导 ★
- 3.3.3 Probability Flow ODE 的推导 ★
- 3.3.4 SDE 与 ODE 共享边缘分布的证明 ★
- 3.3.5 Predictor–Corrector 采样
- 3.3.6 Langevin Corrector 的原理

#### 3.4 CNF 与 Flow Matching 的生成理论

- 3.4.1 从学习的速度场构造初值 ODE
- 3.4.2 流映射与概率密度演化
- 3.4.3 生成轨迹与训练条件插值路径的区别
- 3.4.4 密度变化及对数似然计算
- 3.4.5 流的可逆性、存在性与唯一性

#### 3.5 Rectified Flow 的生成理论

- 3.5.1 速度场定义下的 ODE 生成
- 3.5.2 直线插值与非直线真实轨迹
- 3.5.3 轨迹曲率与数值积分误差
- 3.5.4 Reflow 后的采样
- 3.5.5 少步生成与一步 Euler 近似的局限

#### 3.6 Schrödinger Bridge 的生成理论

- 3.6.1 最优控制漂移的构造
- 3.6.2 Doob $`h`$-transform 的作用 ★
- 3.6.3 正向与反向桥过程
- 3.6.4 Schrödinger Bridge 的 Fokker–Planck 方程
- 3.6.5 桥过程与参考扩散过程的区别
- 3.6.6 端点分布约束与生成误差

#### 3.7 Stochastic Interpolants 的生成理论

- 3.7.1 从插值边缘分布构造连续性方程
- 3.7.2 与相同边缘分布对应的 ODE
- 3.7.3 可调扩散强度下的 SDE 构造
- 3.7.4 正向与反向随机生成过程
- 3.7.5 确定性生成与随机生成的统一

#### 3.8 生成算法的数值分析

- 3.8.1 Euler、Heun、Runge–Kutta 方法
- 3.8.2 Euler–Maruyama 与随机数值方法
- 3.8.3 局部截断误差与全局误差
- 3.8.4 SDE 的强收敛与弱收敛
- 3.8.5 求解器误差与模型估计误差
- 3.8.6 时间步选择与自适应求解

#### 3.9 生成质量及计算复杂度

- 3.9.1 采样步数、NFE 与计算成本
- 3.9.2 训练目标误差对生成分布的影响
- 3.9.3 KL、Wasserstein、MMD 等分布误差
- 3.9.4 确定性与随机生成的质量比较
- 3.9.5 生成稳定性、似然与样本多样性

<a name="part-4"></a>

### Part 4. Score、Velocity、Drift 的计算、估计与相互转换

[阅读第 4 章正文](chapter-4.md)

我建议将你原来的第四部分提升为独立核心模块，而不只是 Score 的计算。

原因是不同模型实际学习的对象不同：

- DDPM 经常学习噪声预测器；
- Score-based Diffusion 学习 Score 场；
- Flow Matching 学习速度场；
- Schrödinger Bridge 涉及受控 Drift、Score 与势函数。

它们在特定条件下能够相互联系，但**不能直接把 Score、Velocity 和 Drift 当成同一个对象**。

#### 4.1 Score 函数的基本理论

- 4.1.1 Score 的定义及几何意义
- 4.1.2 高斯分布 Score 的显式计算 ★
- 4.1.3 条件 Score 与边缘 Score
- 4.1.4 Score 的积分恒等式
- 4.1.5 Score 在概率密度高维估计中的作用

#### 4.2 Score Matching 的数学推导

- 4.2.1 Fisher Divergence 展开
- 4.2.2 分部积分消除未知真实 Score ★
- 4.2.3 Denoising Score Matching 的条件期望推导 ★
- 4.2.4 Sliced Score Matching
- 4.2.5 隐式与显式 Score Matching 的区别

#### 4.3 Score 与噪声预测之间的关系

- 4.3.1 高斯扰动核的条件 Score
- 4.3.2 噪声预测网络的最优目标
- 4.3.3 Tweedie 公式的推导 ★
- 4.3.4 Score、Noise 与 Clean Data 之间的转换
- 4.3.5 DDPM 中的 $`v`$-prediction 与 Flow Velocity 的区别

#### 4.4 Velocity Field 的数学推导

- 4.4.1 速度场的定义与连续性方程
- 4.4.2 条件速度与边缘速度的关系
- 4.4.3 高斯条件路径的速度目标 ★
- 4.4.4 Flow Matching 速度场的条件期望表示
- 4.4.5 速度场的非唯一性与规范选择

#### 4.5 Drift Field 与 Score 的关系

- 4.5.1 SDE 中 Drift 的定义
- 4.5.2 Drift、Diffusion 与概率流
- 4.5.3 各向同性时 Drift 与 Velocity 的转换 ★
- 4.5.4 状态依赖各向异性扩散中的修正项 ★
- 4.5.5 反向 SDE 中的 Score 修正

#### 4.6 Flow Matching 与 Score Matching 的联系

- 4.6.1 高斯插值路径的 Score
- 4.6.2 高斯插值路径的 Velocity
- 4.6.3 在非退化条件下由 Score 表示 Velocity ★
- 4.6.4 由 Velocity 恢复 Score 的条件
- 4.6.5 Flow Matching 与 Diffusion 共享概率路径的构造

#### 4.7 Schrödinger Bridge 中的 Score 与漂移

- 4.7.1 Schrödinger Potentials 的对数梯度
- 4.7.2 最优控制 Drift 的表示
- 4.7.3 正向与反向漂移的关系
- 4.7.4 桥过程 Score 与边缘密度
- 4.7.5 Score 估计误差对桥接过程的影响

#### 4.8 Score 及向量场的计算实现

- 4.8.1 已知解析密度时的自动微分
- 4.8.2 未知密度时的神经网络估计
- 4.8.3 条件样本回归与 Monte Carlo 估计
- 4.8.4 向量场的 Jacobian、散度与 Hessian
- 4.8.5 高维散度估计与 Hutchinson 方法
- 4.8.6 Score 与 Velocity 的数值验证

<a name="part-5"></a>

### Part 5. 模型统一理论、数学可解性与研究拓展

[阅读第 5 章正文](chapter-5.md) · [统一框架快速版](quick-guide.md)

这一部分建议放在整套知识体系的最后，作为从“学习现有模型”转向“研究新模型”的过渡。

#### 5.1 确定性输运与随机扩散的统一

- 5.1.1 连续性方程和 Fokker–Planck 方程
- 5.1.2 概率流、概率流速度与守恒形式
- 5.1.3 SDE 对应的 Probability Flow ODE ★
- 5.1.4 相同边缘分布与不同路径测度
- 5.1.5 ODE、SDE 与随机插值的统一表达

#### 5.2 Flow Matching、Diffusion 与 Schrödinger Bridge 的比较

- 5.2.1 固定概率路径与学习概率路径
- 5.2.2 端点约束与中间边缘约束
- 5.2.3 样本耦合、最优运输和熵正则化
- 5.2.4 确定性与随机动力学的关系
- 5.2.5 不同模型之间成立等价关系的条件

#### 5.3 随机插值与统一生成建模

- 5.3.1 一般随机插值函数的构造
- 5.3.2 插值边缘分布的正则性
- 5.3.3 从插值导出连续性方程
- 5.3.4 从插值导出一族扩散方程
- 5.3.5 Flow、Score 与扩散系数的联合设计

#### 5.4 理论可解性与稳定性

- 5.4.1 ODE 流映射的存在与唯一性
- 5.4.2 SDE 解的存在与路径唯一性
- 5.4.3 Fokker–Planck 方程的弱解与概率解
- 5.4.4 Score 与速度场的正则性条件
- 5.4.5 端点极限、奇异性与有限时间演化

#### 5.5 误差传播与收敛性

- 5.5.1 训练回归误差
- 5.5.2 速度场或 Score 的估计误差
- 5.5.3 ODE/SDE 数值离散误差
- 5.5.4 初始分布与终端分布匹配误差
- 5.5.5 Wasserstein 与 KL 误差上界

#### 5.6 各向异性及状态依赖扩散

- 5.6.1 标量扩散系数向矩阵扩散系数推广
- 5.6.2 $`G(x,t)G(x,t)^\top`$ 与扩散矩阵
- 5.6.3 非常系数扩散的 Fokker–Planck 方程
- 5.6.4 空间依赖扩散中的概率流速度
- 5.6.5 各向异性情况下的逆时间动力学

#### 5.7 耦合动力学与物理约束

- 5.7.1 耦合 Drift 与状态变量
- 5.7.2 耦合随机过程的联合分布演化
- 5.7.3 物理约束下的速度场与 Score
- 5.7.4 守恒律、边界条件与生成动力学
- 5.7.5 生成模型与经典 PDE 之间的联系

#### 5.8 条件生成、逆问题与研究应用

- 5.8.1 Conditional Score 与 Conditional Flow Matching
- 5.8.2 Posterior Sampling 与逆问题
- 5.8.3 Classifier Guidance 与 Classifier-Free Guidance
- 5.8.4 物理条件下的概率路径设计
- 5.8.5 面向流场、涡度场和随机动力系统的生成建模

<a name="derivations"></a>

## **四、真正需要掌握的核心数学推导**

上面的目录比较完整，但不意味着每一个三级标题都需要投入同样多的时间。

我建议优先完成以下六组推导。它们实际上能够把你学习的多个模型联系起来。

<a name="derivation-a"></a>

### 推导 A · DDPM 为什么能够直接从任意时间步采样？

从

```math
q(x_t\mid x_{t-1}) =\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I)
```

出发，推导：

```math
\boxed{ x_t=\sqrt{\bar\alpha_t}x_0+ \sqrt{1-\bar\alpha_t}\epsilon }
```

这里需要理解高斯随机变量的线性组合，以及独立高斯噪声方差的累积规律。

随后才能严格推导 DDPM 的后验分布和去噪损失。DDPM 的原始论文给出了这条理论路线。

<a name="derivation-b"></a>

### 推导 B · 为什么训练噪声预测器等价于估计 Score？

设

```math
X_t=\alpha_t X_0+\sigma_t\epsilon, \qquad \epsilon\sim\mathcal N(0,I).
```

条件概率密度满足

```math
\nabla_x\log p_t(x\mid x_0) =-\frac{x-\alpha_t x_0}{\sigma_t^2}.
```

通过边缘化可得

```math
\boxed{ \nabla_x\log p_t(x) =-\frac{\mathbb E[\epsilon\mid X_t=x]}{\sigma_t} }
```

因此，当神经网络的均方误差训练达到理想最优时：

```math
\boxed{ s_\theta(x,t)\approx -\frac{\epsilon_\theta(x,t)}{\sigma_t} }
```

这一公式是理解 DDPM、Score Matching 和连续时间扩散模型联系的关键。

<a name="derivation-c"></a>

### 推导 C · 从 Fokker–Planck 方程到 Probability Flow ODE

考虑各向同性、扩散系数只依赖时间的 SDE：

```math
dX_t=f(X_t,t)\,dt+g(t)\,dW_t.
```

其 Fokker–Planck 方程为：

```math
\partial_t p_t =-\nabla\cdot(fp_t) +\frac12g(t)^2\Delta p_t.
```

利用

```math
\nabla p_t=p_t\nabla\log p_t,
```

可以将其写成连续性方程：

```math
\partial_t p_t=-\nabla\cdot(u_t p_t),
```

其中

```math
\boxed{ u_t(x)=f(x,t) -\frac12g(t)^2\nabla_x\log p_t(x) }
```

于是构造确定性 ODE：

```math
\boxed{ \frac{dX_t}{dt}=u_t(X_t) }
```

在适当的正则性、初值及解存在唯一性条件下，该 ODE 与原 SDE 具有相同的单时刻边缘分布，但不具有相同的随机路径分布。

这正是 Score-based SDE 理论中的关键结果。

<a name="derivation-d"></a>

### 推导 D · Flow Matching 为什么可以用条件速度训练？

对于线性插值

```math
X_t=(1-t)X_0+tX_1,
```

可以直接计算：

```math
\frac{dX_t}{dt}=X_1-X_0.
```

但这只是给定端点的插值速度，并不意味着生成时的真实 ODE 轨迹一定也是直线。

最优边缘速度场由条件期望给出：

```math
\boxed{ u^*(x,t) =\mathbb E[X_1-X_0\mid X_t=x] }
```

所以可以通过最小化

```math
\mathcal L_{\mathrm{CFM}} =\mathbb E\left[ \left\|u_\theta(X_t,t)-(X_1-X_0)\right\|^2 \right]
```

来学习边缘速度场。

这也回答了你之前关心的一个问题：

**训练中人为设计的条件插值轨迹，并不等同于生成过程中由学得的边缘速度场积分得到的真实轨迹。**

<a name="derivation-e"></a>

### 推导 E · 各向异性扩散中如何由 Drift 和 Score 构造 Velocity？

这是你后续理论研究尤其需要掌握的内容。

对于

```math
dX_t=f(X_t,t)\,dt+G(X_t,t)\,dW_t,
```

令

```math
a(x,t)=G(x,t)G(x,t)^\top.
```

相应的概率流速度可表示为

```math
\boxed{ u_i(x,t) =f_i(x,t) -\frac12\sum_j\partial_{x_j}a_{ij}(x,t) -\frac12\sum_j a_{ij}(x,t)\partial_{x_j}\log p_t(x) }
```

这里出现了一个很重要的区别：

当扩散矩阵依赖空间变量时，除了 $`a\nabla\log p_t`$ 之外，还必须考虑扩散矩阵的散度项。

这意味着不能简单地把标量扩散情况下的公式直接替换成矩阵形式，而忽略空间导数。

<a name="derivation-f"></a>

### 推导 F · Schrödinger Bridge 为什么是一个随机控制问题？

给定参考路径测度 $`R`$，Schrödinger Bridge 的核心变分问题是：

```math
\boxed{ \min_{P:\,P_0=\mu_0,\;P_T=\mu_T} D_{\mathrm{KL}}(P\|R) }
```

这里的 $`P`$ 是整个随机过程的路径测度，而不只是某个时刻的概率密度。

对于满足适当条件、具有相同扩散系数的受控过程，可以通过 Girsanov 定理，把路径空间相对熵与漂移控制的二次能量联系起来。

因此，需要重点研究：

- 路径空间 KL 与端点分布约束。
- 受控漂移的构造及其最优性。
- Schrödinger Potentials 和 Doob $`h`$-transform。
- 最优桥过程对应的 Fokker–Planck 方程。

这一理论也是 Diffusion Schrödinger Bridge 的重要基础。

## **五、八个核心模型的统一对照**

这一张表适合放在全部学习笔记的开头，并在每学完一个模型之后不断补充。

| 模型                    | 核心学习对象       | 典型训练方式             | 典型生成方式         |
| ----------------------- | ------------------ | ------------------------ | -------------------- |
| DDPM                    | 噪声/Score         | ELBO、去噪 MSE           | 反向随机链           |
| DDIM                    | 去噪预测器         | 通常沿用 DDPM 损失       | 确定性或随机隐式采样 |
| Score-based SDE         | Score              | Denoising Score Matching | 反向 SDE / PF-ODE    |
| CNF                     | 连续速度场         | 似然最大化               | ODE                  |
| FM / CFM                | 速度场             | 条件速度回归             | ODE                  |
| Rectified Flow          | 运输速度场         | 线性插值速度回归         | ODE                  |
| Schrödinger Bridge      | 最优受控路径及漂移 | 路径 KL、迭代桥估计等    | 受控 SDE             |
| Stochastic Interpolants | Velocity 与 Score  | 条件回归                 | ODE / SDE            |

需要特别注意：**同一边缘概率分布路径，不代表相同的样本轨迹；相同的训练损失，也不代表相同的生成算法。**

例如 DDIM 论文的一个重要贡献，就是构造与 DDPM 常用训练目标兼容、但采样方式不同的生成过程。

## **六、推荐的学习顺序**

这里不建议按论文发表时间逐篇阅读，而应该按照理论依赖关系学习。

推荐学习路径

**01**

概率论 + ODE + 连续性方程

建立密度随流动演化的基本认识

**02**

SDE + Itô 公式 + Fokker–Planck

理解随机动力学与概率密度之间的联系

**03**

DDPM → DDIM

学习离散扩散与去噪生成

**04**

Score Matching → Score SDE

完成从离散模型到连续时间扩散的过渡

**05**

CNF → FM → CFM → Rectified Flow

建立确定性概率输运的理论体系

**06**

Optimal Transport → Schrödinger Bridge

学习路径最优化、端点约束及随机控制

**07**

Stochastic Interpolants

将速度、Score、ODE 和 SDE 放进统一框架

**08**

各向异性、耦合动力学及误差分析

由模型理解进入理论研究

## **七、建议配套阅读的原始论文**

下面不是全面的文献综述，而是与你这套学习目录最直接相关的核心论文。

| 学习主题           | 推荐论文                                                     |
| ------------------ | ------------------------------------------------------------ |
| DDPM               | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) |
| DDIM               | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) |
| Score SDE          | [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) |
| Flow Matching      | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) |
| FM 理论与实现      | [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) |
| Rectified Flow     | [Flow Straight and Fast](https://arxiv.org/abs/2209.03003) |
| Schrödinger Bridge | [Diffusion Schrödinger Bridge](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html) |
| 统一理论           | [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://jmlr.org/papers/v26/23-1605.html) |
| 采样与参数化       | [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) |
| 快速生成           | [Consistency Models](https://arxiv.org/abs/2303.01469) |

其中，2025 年发表在 JMLR 的 Stochastic Interpolants 工作尤其适合放在整个学习体系的后半部分，用于连接概率路径、确定性输运与随机扩散。

## **八、针对你的理论研究方向，哪些章节最值得深入？**

考虑到你已经在接触 Fokker–Planck 方程、Flow Matching 及其与随机动力学之间的联系，我建议采用以下重点分配。

数学基础与 Fokker–Planck

能从一般 Itô SDE 推导概率密度演化

Flow Matching / CFM

能证明条件速度回归与边缘速度场的联系

Score SDE 与概率流 ODE

能推导反向 SDE 及 ODE/SDE 边缘等价关系

Schrödinger Bridge

理解路径 KL、端点约束及最优受控漂移

Stochastic Interpolants

建立 Flow 与 Diffusion 的统一数学表示

DDPM / DDIM

完整掌握离散加噪、训练损失和采样递推

采样算法和实验实现

能独立验证模型与数值误差

各向异性与耦合扩展

形成理论推广问题与可证明的数学命题

星级表示针对你当前研究目标的学习优先级，不代表模型本身的重要性排名。

对你而言，最终最值得追求的能力，是面对新的随机动力系统：

```math
dX_t=f(X_t,t)\,dt+G(X_t,t)\,dW_t
```

能够独立完成以下理论闭环：

```math
\boxed{ \text{随机动力系统} \longrightarrow \text{Fokker–Planck 方程} \longrightarrow \text{概率流与 Score} \longrightarrow \text{训练目标} \longrightarrow \text{生成 ODE/SDE} }
```

并进一步研究这个闭环在非线性耦合、各向异性扩散与端点分布约束下需要什么条件才能成立。

**最终建议：把这套体系作为一门完整的研究生级生成模型理论课程，而不只是五篇模型论文的读书笔记。**

你的原有四部分应当保留，但应以 Part 0—Part 5 的六模块结构组织。学习过程中，对每一个模型都使用相同的四个问题进行检验：它指定了什么概率路径？训练时估计什么数学对象？生成时求解什么动力系统？这些构造在哪些数学假设下成立？

这样完成学习后，你不仅能够解释 DDPM、DDIM、Flow Matching 与薛定谔桥如何工作，还能严谨分析它们之间什么时候等价、什么时候不同，以及哪些理论能够推广到你的课题。
