# 第 5 章　模型统一理论、数学可解性与研究拓展

[返回项目目录](../README.md#contents) · [统一框架快速版](quick-guide.md) · [上一章](chapter-4.md) · [PDF 版](../output/pdf/model-study.pdf) · [LaTeX 源码](../chapters/05-unified-theory.tex)

[5.1 确定性输运与随机扩散的统一](#sec-5-1) · [5.2 Flow Matching、Diffusion 与 Schrödinger Bridge 的比较](#sec-5-2) · [5.3 随机插值与统一生成建模](#sec-5-3) · [5.4 理论可解性与稳定性](#sec-5-4) · [5.5 误差传播与收敛性](#sec-5-5) · [5.6 各向异性及状态依赖扩散](#sec-5-6) · [5.7 耦合动力学与物理约束](#sec-5-7) · [5.8 条件生成、逆问题与研究应用](#sec-5-8)

前四章分别建立了概率路径、训练目标、生成算法及场的转换。
本章把这些环节连接起来，讨论一个模型何时定义了适定的概率演化，
回归误差如何变为生成分布误差，以及怎样加入条件信息、耦合变量和物理结构。
仍令 $`t=0`$ 为数据端、$`t=1`$ 为噪声端，$`\tau=1-t`$ 为生成时间。
加噪边缘为 $`p_t`$，生成边缘为 $`\rho_\tau=p_{1-\tau}`$，
$`s_t=\nabla\log p_t`$、$`v_\tau=-u_{1-\tau}`$。
除边界专题外，先在 $`\mathbb R^d`$ 的内部正则时段工作。
密度导数、分部积分和时间反演分别使用其所需的光滑性、可积性及边界条件。

本章给出经典正则情形的直接推导。低正则叠加原理等结果作为有来源的定理使用，
相应的研究应用按明确的有限维建模问题展开。
物理时间记为 $`r`$，用来区分物理系统的演化与生成坐标 $`t,\tau`$。

### 统一框架的快速读法

先把每个模型写成同一条推理链，再按需展开具体推导。

```math
\begin{gathered}
 \text{目标与概率路径}\ \longrightarrow\ \text{可采样监督}\ \longrightarrow\ \text{边缘场},\\
 \text{边缘场}\ \longrightarrow\ \text{生成动力学}\ \longrightarrow\ \text{分布与物理验证}.
 \end{gathered}
```

目标包括数据、先验与条件信息；路径包括端点耦合、扰动核或随机插值。
条件回归连接可采样监督与边缘场，连续性方程和 Fokker–Planck 方程连接场与生成分布。
同一高斯路径上的网络输出可以在非退化处换算，同边缘的不同动力学仍需各自检查适定性与离散误差。
快速比较模型时，依次核对路径选择、学习对象、生成方程和误差条件；详细公式见后续各节。

**学习目标**

- 用概率流统一连续性方程与 Fokker–Planck 方程，并明确时刻边缘与路径律的关系。
- 比较固定路径回归、耦合优化与桥路径优化，核对模型等价所需条件。
- 建立存在唯一性、端点极限和误差传播的可验证条件。
- 推导矩阵扩散、联合状态、物理约束及后验条件生成的具体方程。

<a name="sec-5-1"></a>

## 5.1 确定性输运与随机扩散的统一

<a name="sec-5-1-1"></a>

### 5.1.1 连续性方程和 Fokker–Planck 方程

确定性 ODE $`\dot X_t=u_t(X_t)`$ 的测试函数恒等式为
$`\frac{\mathrm{d} }{\mathrm{d}  t}\mathbb E\varphi(X_t)=\mathbb E[u_t\cdot\nabla\varphi(X_t)]`$，
从而密度满足 $`\partial_t p=-\nabla\cdot(pu)`$。
对 Itô SDE $`\mathrm{d}  X_t=f_t(X_t)\mathrm{d}  t+G_t(X_t)\mathrm{d}  W_t`$，
记 $`a_t=G_tG_t^\top`$。Itô 公式在期望下消去随机积分，给出

<a name="eq-unified-fp"></a>

```math
\begin{aligned}
 \frac{\mathrm{d} }{\mathrm{d}  t}\int\varphi p_t
 &=\int\left(f_t\cdot\nabla\varphi+
             \tfrac12\sum_{ij}a_{ij,t}\partial_{ij}\varphi\right)p_t,\\
 \partial_t p_t&=-\nabla\cdot(f_tp_t)+\tfrac12\sum_{ij}\partial_{ij}(a_{ij,t}p_t).
 \end{aligned}
\tag{5.1}
```

前一个恒等式是弱形式，后一个是分部积分后的密度形式。
当边缘没有密度时，可把 $`p_t(x)\mathrm{d}  x`$ 改成概率测度 $`\mu_t(\mathrm{d}  x)`$
保留弱形式。两种动力学都通过测试函数刻画分布演化，SDE 的二阶项来自二次变差。

<a name="sec-5-1-2"></a>

### 5.1.2 概率流、概率流速度与守恒形式

逐坐标定义概率流密度

```math
j_{i,t}=f_{i,t}p_t-\tfrac12\sum_j\partial_j(a_{ij,t}p_t).
```

则 $`\partial_t p_t+\nabla\cdot j_t=0`$。在 $`p_t>0`$ 处定义 $`u_t=j_t/p_t`$，有

<a name="eq-unified-current"></a>

```math
j_t=p_tu_t,\qquad
 u_t=f_t-\tfrac12\operatorname{div}a_t-\tfrac12a_ts_t,
 \quad (\operatorname{div}a_t)_i=\sum_j\partial_j a_{ij,t}.
\tag{5.2}
```

若区域有边界，总质量满足
$`\frac{\mathrm{d} }{\mathrm{d}  t}\int_\Omega p_t=-\int_{\partial\Omega}j_t\cdot n`$。
因此守恒还需要无通量、周期条件或足够快的无穷远衰减。
概率流速度选定了给定扩散的一种输运表示。若加入满足
$`\nabla\cdot(p_tw_t)=0`$ 的场，时刻边缘仍可相同。

<a name="sec-5-1-3"></a>

### 5.1.3 SDE 对应的 Probability Flow ODE ★

对 $`a_{ij,t}p_t`$ 使用乘积法则，得到
$`\sum_j\partial_j(a_{ij,t}p_t)
=p_t(\operatorname{div}a_t)_i+p_t(a_ts_t)_i`$。
代回式 [(5.1)](#eq-unified-fp) 恰得到
$`\partial_t p_t=-\nabla\cdot(p_tu_t)`$。
若 $`u_t`$ 生成的 ODE 与该连续性方程都适定，并从同一初始律出发，
ODE 的推前律和给定 SDE 的边缘满足同一初值问题，故相等。
生成时间下对应

<a name="eq-unified-probability-flow"></a>

```math
\dot Z_\tau=v_\tau(Z_\tau),\qquad
 v_\tau=-f_t+\tfrac12\operatorname{div}a_t+\tfrac12a_ts_t,
 \quad t=1-\tau.
\tag{5.3}
```

这一步同时使用密度方程的一致性和解的唯一性。
在正则内部时段可以先证明共享边缘，再研究两端的极限。
Score SDE 的构造及一般扩散形式见 〔[3](#ref-song2021sde)〕。

<a name="sec-5-1-4"></a>

### 5.1.4 相同边缘分布与不同路径测度

路径律记录 $`(X_{t_1},\ldots,X_{t_m})`$ 的全部联合分布，
单时刻边缘只给出其中 $`m=1`$ 的信息。
若扩散非退化，SDE 轨迹的二次变差为 $`\int a_t\mathrm{d}  t`$，
正则 ODE 轨迹的二次变差为零。因此共享边缘可以对应不同的路径统计。
训练时固定端点的插值曲线也有自己的路径律，其边缘速度回归只保证相应的输运方程。

**例 5.1 · 共享标准高斯边缘的三种过程**

令 $`Z_0\sim\mathcal N(0,I_2)`$。常值过程 $`Z_t=Z_0`$、
旋转过程 $`Z_t=e^{tJ}Z_0`$（$`J^\top=-J`$），
以及平稳 OU 过程 $`\mathrm{d}  X_t=-X_t\mathrm{d}  t+\sqrt2\mathrm{d}  W_t`$
都保持标准高斯边缘。
常值过程的两时刻协方差为 $`I_2`$，旋转过程为
$`\mathbb E[Z_tZ_s^\top]=e^{(t-s)J}`$，OU 过程为 $`e^{-|t-s|}I_2`$。
计算边缘指标会得到相同答案，计算时间相关性则能区分这些路径律。

<a name="sec-5-1-5"></a>

### 5.1.5 ODE、SDE 与随机插值的统一表达

假设已知 $`p_t,u_t,s_t`$ 且 $`\partial_t p_t=-\nabla\cdot(p_tu_t)`$。
对选定的半正定协方差 $`a_t(x)`$，定义

<a name="eq-unified-diffusion-family"></a>

```math
f_t^{(a)}=u_t+\tfrac12\operatorname{div}a_t+\tfrac12a_ts_t,
 \qquad
 \mathrm{d}  X_t=f_t^{(a)}(X_t)\mathrm{d}  t+G_t(X_t)\mathrm{d}  W_t,
 \quad GG^\top=a.
\tag{5.4}
```

代回概率流公式便得到原来的 $`u_t`$。当该 SDE 与密度方程适定且初始律匹配时，
它实现指定边缘。$`a=0`$ 对应确定性输运；$`a=2\kappa(t)I_d`$
对应漂移 $`u+\kappa s`$。随机插值提供可采样的 $`p_t`$ 及条件回归标签，
由此可以构造这一族动力学 〔[8](#ref-albergo2025si)〕。
选择扩散会改变路径律和误差传播，实际近似场下的输出分布还要分别验证。

<a name="sec-5-2"></a>

## 5.2 Flow Matching、Diffusion 与 Schrödinger Bridge 的比较

<a name="sec-5-2-1"></a>

### 5.2.1 固定概率路径与学习概率路径

标准 FM/CFM 先选端点耦合与条件插值，再回归边缘速度；
标准扩散先选前向漂移与噪声表，再学习这些边缘上的 Score。
两者的训练目标都依赖预先指定的概率路径。
Schrödinger Bridge 给定参考路径律 $`R`$ 和两端边缘，求解

<a name="eq-unified-sb-objective"></a>

```math
P^\star\in\arg\min_{P:\,P_0=\mu_0,\ P_1=\mu_1}
                                      D_{\mathrm{KL}}(P\|R).
\tag{5.5}
```

中间边缘由优化后的路径律决定。
也可以在随机插值族中优化耦合、系数或控制，使路径本身成为学习变量。
这时需要记录外层路径优化与内层场回归各自的目标和误差，
并检查该参数化族能否表达所需最优解。

<a name="sec-5-2-2"></a>

### 5.2.2 端点约束与中间边缘约束

两端分布给定后，中间路径仍有许多选择。
例如标准高斯两端可以用恒定边缘，也可用先缩小再放大的高斯方差路径。
连续性方程或 Fokker–Planck 方程约束了路径与场的相容性，
附加最小动能或相对熵目标才进一步选择其中的演化。
指定所有 $`p_t`$ 后，速度仍可相差一个 $`p_t`$-加权无散场。
因此依次需要明确端点、边缘路径、场的选择和路径空间目标。
桥的两端约束按测度满足，有限迭代的近似模型则应检查相应边缘残差。

<a name="sec-5-2-3"></a>

### 5.2.3 样本耦合、最优运输和熵正则化

给定端点耦合 $`\pi\in\Pi(\mu_0,\mu_1)`$，
线性插值与条件回归由 $`\pi`$ 共同决定。
二次 OT 最小化 $`\int\lVert x-y\rVert^2\pi(\mathrm{d}  x,\mathrm{d}  y)`$，
Benamou--Brenier 形式在相应条件下最小化
$`\int_0^1\int\lVert u_t\rVert^2p_t\mathrm{d}  x\mathrm{d}  t`$，
约束是连续性方程及端点 〔[11](#ref-benamou2000ot)〕。
若参考过程是方差参数 $`\epsilon`$ 的 Brownian 扩散，其端点核含
$`\exp[-\lVert x-y\rVert^2/(2\epsilon)]`$。
对固定参考初始律 $`R_0`$ 和目标边缘，KL 链式分解给出

```math
D_{\mathrm{KL}}(\pi\|R_{01})
 =D_{\mathrm{KL}}(\pi\|R_0\otimes\mu_1)
       +\frac1{2\epsilon}\int\lVert x-y\rVert^2\mathrm{d} \pi+C,
```

其中常数在密度与熵可积条件下与 $`\pi`$ 无关。
相应的熵正则权重依赖参考核的尺度。

给定端点后，桥优化保留参考条件桥律，最小路径 KL 归结为端点耦合优化。
小噪声极限与 OT 的联系需要紧性、代价收敛和极限辨识等条件
〔[13](#ref-leonard2014schrodinger)〕。
minibatch OT-CFM 中的有限批次匹配定义了实际训练耦合，
其与总体 OT 的关系还涉及抽样和批次大小 〔[7](#ref-tong2024otcfm)〕。

<a name="sec-5-2-4"></a>

### 5.2.4 确定性与随机动力学的关系

对可微流，生成端点由初始点通过映射确定，因此端点联合律集中在映射图像上。
随机动力学还会在给定起点后产生条件波动。
两者均可输运同一对分布，但实现的端点耦合与路径代价可能不同。
对同边缘族使用式 [(5.4)](#eq-unified-diffusion-family)，
可直接比较不同扩散强度下的路径性质。
确定性流的可逆性要求相应流映射存在且唯一；
随机桥的相对熵最优性由指定参考律上的变分问题定义。
将桥的边缘换成 ODE 表示时，应重新说明要研究的路径泛函。

<a name="sec-5-2-5"></a>

### 5.2.5 不同模型之间成立等价关系的条件

本书使用的几种等价关系各有明确的对象。
DSM 与边缘 Score 回归、CFM 与边缘速度回归，在同一训练分布上满足加常数风险恒等式。
高斯路径上的噪声、干净数据与 Score 在非退化处可代数换算，
损失的时间权重和二次度量也需相应换算。
概率流 ODE 与扩散 SDE 的共享边缘，还要求精确场、匹配初始律和演化唯一性。
DDPM 与 DDIM 可共享单时刻训练边缘，有限步生成核依所选算法确定。

判断某个“模型统一”结论时，可以逐项核对
所用的联合律、时间坐标、边缘路径、网络输出、误差度量、
生成初始律以及离散求解器。
若要声称桥最优性，还需验证式 [(5.5)](#eq-unified-sb-objective) 的参考律和变分条件。
这些核对把公式转换落实为可检验的数学命题。

<a name="sec-5-3"></a>

## 5.3 随机插值与统一生成建模

<a name="sec-5-3-1"></a>

### 5.3.1 一般随机插值函数的构造

取耦合端点 $`(D,N)\sim\pi`$，再取独立辅助噪声
$`\zeta\sim\mathcal N(0,I_d)`$。定义

<a name="eq-unified-interpolant"></a>

```math
\begin{gathered}
 I_t=\mathcal I(t,D,N)+\gamma(t)\zeta,\\
 \mathcal I(0,D,N)=D,\quad\mathcal I(1,D,N)=N,\quad
 \gamma(0)=\gamma(1)=0.
 \end{gathered}
\tag{5.6}
```

端点由上述恒等式直接保证。
$`\mathcal I`$ 可取线性或非线性插值，$`\pi`$ 可指定独立、OT 或教师生成耦合。
辅助噪声用于内部平滑，其随机变量与端点噪声 $`N`$ 各有自己的角色。
构造速度标签时，沿单条路径固定 $`(D,N,\zeta)`$，
使用 $`\dot I_t=\partial_t\mathcal I+\dot\gamma(t)\zeta`$。
训练风险要求该标签在实际时间抽样律下具有所需的二次矩。

<a name="sec-5-3-2"></a>

### 5.3.2 插值边缘分布的正则性

在固定内部时刻 $`\gamma(t)>0`$ 时，$`p_t`$ 是某个概率测度与非退化高斯核的卷积，
因此空间上正且光滑。固定噪声尺度下，高斯核的任意阶导数为有界的
多项式乘高斯，允许把空间导数移入概率积分。
时间正则性还依赖 $`\mathcal I,\gamma`$ 的时间导数及其可积控制。
Score 的存在只使用局部正性，全球 Lipschitz 界则要求更强的后验矩或尾部控制。

对 $`X=Y+\gamma\zeta`$，二阶 Tweedie 公式给出

<a name="eq-unified-score-hessian"></a>

```math
\nabla_xs(x)=-\frac1{\gamma^2}I_d
                 +\frac1{\gamma^4}\operatorname{Cov}(Y\mid X=x).
\tag{5.7}
```

因此 Hessian 有下界 $`-I_d/\gamma^2`$。
若要得到双侧有界的 Jacobian，尚需控制后验协方差的最大特征值。
当 $`\gamma\downarrow0`$ 时，这些界会显式依赖噪声尺度。

<a name="sec-5-3-3"></a>

### 5.3.3 从插值导出连续性方程

设 $`\mathbb E\int_0^1\lVert \dot I_t\rVert\mathrm{d}  t<\infty`$。
对紧支撑光滑测试函数 $`\varphi`$，链式法则与 Fubini 给出

```math
\mathbb E\varphi(I_t)-\mathbb E\varphi(I_0)
 =\int_0^t\mathbb E[\nabla\varphi(I_r)\cdot\dot I_r]\mathrm{d}  r.
```

定义 $`u_t(x)=\mathbb E[\dot I_t\mid I_t=x]`$，即可把右端写成
$`\int_0^t\int\nabla\varphi(x)\cdot u_r(x)p_r(x)\mathrm{d}  x\mathrm{d}  r`$。
这就是连续性方程的积分弱形式。
若标签平方可积，条件投影还给出
$`\int\lVert u_t\rVert^2p_t\le\mathbb E\lVert \dot I_t\rVert^2`$。
因此从插值到可回归速度的推导同时给出一个能量上界。

<a name="sec-5-3-4"></a>

### 5.3.4 从插值导出一族扩散方程

在 $`p_t>0`$ 的内部时段，设 $`\kappa(t)\ge0`$ 只依赖时间。
令 $`f_t=u_t+\kappa(t)s_t`$、$`G_t=\sqrt{2\kappa(t)}I_d`$。
则密度方程中
$`-\kappa\nabla\cdot(s_tp_t)+\kappa\Delta p_t=0`$，
所以仍得到原插值的边缘。
生成时间的对应 SDE 为

<a name="eq-unified-interpolant-sde"></a>

```math
\mathrm{d}  Y_\tau=[v_\tau(Y_\tau)+\kappa(1-\tau)s_{1-\tau}(Y_\tau)]\mathrm{d} \tau
                  +\sqrt{2\kappa(1-\tau)}\mathrm{d} \overline W_\tau.
\tag{5.8}
```

从 $`p_1`$ 出发、场精确且演化唯一时，它实现 $`p_{1-\tau}`$。
如果使用状态依赖的 $`\kappa(x,t)`$，协方差为 $`2\kappa I_d`$，
前向漂移还要加入 $`\nabla_x\kappa`$，才能保持相同概率流。

<a name="sec-5-3-5"></a>

### 5.3.5 Flow、Score 与扩散系数的联合设计

路径设计决定训练样本、回归标签与边缘正则性；扩散选择决定采样路径及误差的权重。
例如 $`\gamma(t)=\sqrt{t(1-t)}`$ 使内部平滑，但
$`\dot\gamma(t)^2=(1-2t)^2/[4t(1-t)]`$ 在两端不可积。
对于含独立辅助噪声的导数标签，这会使均匀时间抽样的完整平方标签风险发散。
可使用端点导数更温和的 $`\gamma(t)=c\,t(1-t)`$，
或在明确目标权重的前提下改变时间抽样与截断。
相应的 Score 标签尺度又受 $`1/\gamma(t)`$ 影响。

**例 5.2 · 三种场共用一条高斯路径**

设 $`D,N,\zeta`$ 独立标准高斯，$`I_t=(1-t)D+tN+\gamma(t)\zeta`$，
记 $`H_t=(1-t)^2+t^2+\gamma(t)^2`$。则

```math
p_t=\mathcal N(0,H_tI_d),\qquad
 u_t(x)=H_t'x/(2H_t),\qquad s_t(x)=-x/H_t.
```

在前向 SDE 中，单坐标方差 $`C_t`$ 满足
$`C_t'=2[H_t'/(2H_t)-\kappa/H_t]C_t+2\kappa`$。
代入 $`C_t=H_t`$ 得 $`C_t'=H_t'`$。
这同时验证了插值、输运 ODE 与所构造扩散的边缘一致性。

<a name="sec-5-4"></a>

## 5.4 理论可解性与稳定性

<a name="sec-5-4-1"></a>

### 5.4.1 ODE 流映射的存在与唯一性

在生成时间写 $`\dot Z_\tau=v_\tau(Z_\tau)`$。
一套经典充分条件是 $`v`$ 关于时间连续，关于空间全局 Lipschitz，
$`\lVert v(x,\tau)-v(y,\tau)\rVert\le L(\tau)\lVert x-y\rVert`$，
且 $`L`$ 与 $`\lVert v(0,\tau)\rVert`$ 在有限区间可积。
积分方程 $`Z_\tau=z+\int_0^\tau v_r(Z_r)\mathrm{d}  r`$
可在短区间通过 Picard 迭代构造，分段延拓到全区间。
增长界 $`\lVert v(x,\tau)\rVert\le\lVert v(0,\tau)\rVert+L(\tau)\lVert x\rVert`$
和 Gronwall 控制轨迹范数。对两个初值，有

<a name="eq-unified-flow-stability"></a>

```math
\lVert \Phi_{\tau\leftarrow0}(x)-\Phi_{\tau\leftarrow0}(y)\rVert
 \le\exp\left(\int_0^\tau L(r)\mathrm{d}  r\right)\lVert x-y\rVert.
\tag{5.9}
```

反向时间的同类方程也适定时，流映射可逆。
局部 Lipschitz 给出爆破前的唯一解，还需增长或 Lyapunov 控制保证全时间存在。

**例 5.3 · 唯一性与爆破各自需要条件**

方程 $`\dot x=2\sqrt{|x|}`$、$`x(0)=0`$ 的解可以先保持零到任意 $`c\ge0`$，
再取 $`x(t)=(t-c)^2`$。这展示了连续但非 Lipschitz 的场允许非唯一解。
方程 $`\dot x=x^2`$、$`x(0)=x_0>0`$ 的解为
$`x(t)=x_0/(1-x_0t)`$，在 $`t=1/x_0`$ 爆破。
其漂移局部光滑，仍需要额外条件来保证指定区间上的全局解。

<a name="sec-5-4-2"></a>

### 5.4.2 SDE 解的存在与路径唯一性

对 $`\mathrm{d}  X_t=f_t(X_t)\mathrm{d}  t+G_t(X_t)\mathrm{d}  W_t`$，
若 $`f,G`$ 满足全局空间 Lipschitz、线性增长及适当时间可测性，
且初始变量平方可积并与未来 Brownian 增量独立，
则可在给定概率空间与给定 Brownian 运动上构造唯一强解。
Picard 迭代同时更新漂移积分与随机积分。
Itô 等距、Doob 不等式与线性增长给出有限时段二阶矩控制，
迭代差的积分估计产生阶乘衰减，从而在 $`L^2`$ 意义下收敛。

对同一 Brownian 驱动与同一初始条件的两个解，Itô 公式给出

```math
\mathbb E\lVert X_t-\widetilde X_t\rVert^2
 \le (2L_f+L_G^2)\int_0^t\mathbb E\lVert X_r-\widetilde X_r\rVert^2\mathrm{d}  r.
```

Gronwall 得到路径唯一性，再由连续轨迹推出不可区分性。
这里 $`L_G`$ 对 Frobenius 范数取 Lipschitz 界。
局部 Lipschitz 情形可先停时局部化，再用 Lyapunov 函数排除爆破。
例如若 $`V(x)\to\infty`$ 且生成元满足 $`LV\le C(1+V)`$，
可由停时矩估计控制离开大球的概率。
弱解允许同时选择概率空间，路径唯一性则针对相同噪声上的解。

<a name="sec-5-4-3"></a>

### 5.4.3 Fokker–Planck 方程的弱解与概率解

设 $`\mu_t`$ 为窄连续概率测度族。称其满足弱 Fokker–Planck 方程，是指
对任意 $`\varphi\in C_c^\infty`$，

<a name="eq-unified-weak-fp"></a>

```math
\int\varphi\mathrm{d} \mu_t-\int\varphi\mathrm{d} \mu_0
 =\int_0^t\int\left[f_r\cdot\nabla\varphi+
                     \tfrac12 a_r:\nabla^2\varphi\right]\mathrm{d} \mu_r\mathrm{d}  r.
\tag{5.10}
```

SDE 解在适当可积条件下由 Itô 公式产生这样的概率解。
反方向需要把时刻测度提升为路径空间上的过程。
可使用叠加原理，在 Borel 系数、对称半正定 $`a`$、
窄连续概率解及例如
$`\int_0^1\int(\lVert f\rVert+\lVert a\rVert)\mathrm{d} \mu_t\mathrm{d}  t<\infty`$
这样的充分可积条件下，构造具有这些边缘并解对应鞅问题的路径律。
低正则情形的准确陈述与证明引用 Trevisan 〔[23](#ref-trevisan2016wellposed)〕。

该定理提供概率路径表示。要推出路径律唯一性或确定性流的唯一性，
还需验证相应的鞅问题或特征线问题的唯一性条件。
在经典光滑、增长受控的系数下，可结合前一小节的 SDE 理论完成这些步骤。
弱密度方程允许奇异测度，适合讨论端点极限。

<a name="sec-5-4-4"></a>

### 5.4.4 Score 与速度场的正则性条件

密度正且光滑保证 Score 局部光滑。
但 $`s=\nabla p/p`$ 的分母在尾部可很小，其全球增长和 Lipschitz 常数需单独估计。
式 [(5.7)](#eq-unified-score-hessian) 将 Jacobian 约束联系到后验协方差，
可用于具体高斯卷积路径的正则性分析。
高斯分布的 Score 是线性的；混合高斯的局部导数还受分量分离与后验权重影响。
从条件期望定义的速度同样需要分母 $`p_t(x)`$ 的控制。

如果路径中出现 $`\dot\alpha/\alpha`$、$`\dot\sigma/\sigma`$ 或 $`1/\gamma`$，
应检查这些系数在求解区间是否有界或可积。
对每个固定 $`\epsilon>0`$，内部区间 $`[\epsilon,1-\epsilon]`$
可能具有良好估计，但常数可在 $`\epsilon\downarrow0`$ 时发散。
这会同时影响流的稳定性、数值步长及误差上界。
神经网络的结构可提供部分 Lipschitz 控制，训练风险本身还需与这些结构条件结合。

<a name="sec-5-4-5"></a>

### 5.4.5 端点极限、奇异性与有限时间演化

若先验有全维密度，正则可逆的有限时间流会保持绝对连续性。
要精确达到低维或离散端点，需研究退化、奇异系数或适当极限。
常见做法是在正噪声时刻停止，并把平滑目标与原始目标之间的距离计入误差。

**例 5.4 · 到达点质量的奇异生成流**

取 $`p_t=\mathcal N(0,tI_d)`$、$`0<t\le1`$。
其 Score 为 $`s_t=-x/t`$，加噪概率流速度为 $`u_t=x/(2t)`$。
生成 ODE 的显式解为

```math
\dot Z_\tau=-\frac{Z_\tau}{2(1-\tau)},\qquad
 Z_\tau=\sqrt{1-\tau}\,Z_0,\quad Z_0\sim\mathcal N(0,I_d).
```

每个 $`\tau<1`$ 的映射可逆，$`\tau\uparrow1`$ 时输出收敛到 $`\delta_0`$。
停止在 $`t=\epsilon`$ 时，$`W_2(p_\epsilon,\delta_0)=\sqrt{d\epsilon}`$，
而 $`D_{\mathrm{KL}}(\delta_0\|p_\epsilon)=\infty`$，全变差距离为 $`1`$。
端点的收敛结论因此需要明确选用的分布距离。

<a name="sec-5-5"></a>

## 5.5 误差传播与收敛性

<a name="sec-5-5-1"></a>

### 5.5.1 训练回归误差

设训练时间密度为 $`q(t)`$，正权重为 $`w(t)`$，可采样标签为 $`T_t`$。
记 $`m_t(x)=\mathbb E[T_t\mid X_t=x]`$。总体风险满足

<a name="eq-unified-regression-error"></a>

```math
\begin{aligned}
 \mathcal L(\theta)&=\int_0^1q(t)w(t)
             \lVert F_\theta-m_t\rVert_{L^2(p_t)}^2\mathrm{d}  t+C_T,\\
 C_T&=\int_0^1q(t)w(t)\mathbb E\lVert T_t-m_t(X_t)\rVert^2\mathrm{d}  t.
 \end{aligned}
\tag{5.11}
```

第二项表示标签的不可约条件方差，条件标签的期望在固定时刻的联合律下计算。
容量充分时，超出该常数的总体风险就是目标场的加权误差。
有限函数类中还存在表示误差；优化未达到类内最优值时还有优化误差；
经验损失与总体风险之间需要统计泛化控制。
低经验损失要通过这些环节才能用于生成误差结论。

训练抽样覆盖也需与采样区间相容。
若 $`q(t)w(t)`$ 在重要时段接近零，平均风险可能无法控制该区间的场误差。
截断训练区间时，要在采样算法与端点误差中明确相应处理。

<a name="sec-5-5-2"></a>

### 5.5.2 速度场或 Score 的估计误差

对学习速度 $`v_\theta`$ 与参考速度 $`v`$，
设 $`v_\theta`$ 的空间 Lipschitz 常数为 $`L(\tau)`$，
$`\varepsilon_v(\tau)=\lVert v_\theta-v\rVert_{L^2(\rho_\tau)}`$。
在同一初始耦合上积分两个 ODE，使用三角不等式和 Gronwall，可得

<a name="eq-unified-w2-bound"></a>

```math
\begin{aligned}
 W_2(\widehat\rho_1,\rho_1)
 &\le e^{\int_0^1L}\,W_2(\widehat\rho_0,\rho_0)\\
 &\quad+\int_0^1 e^{\int_s^1L(r)\mathrm{d}  r}\varepsilon_v(s)\mathrm{d}  s.
 \end{aligned}
\tag{5.12}
```

其中场误差沿参考边缘取期望，需二阶矩有限。
证明时加减 $`v_\theta(Z_\tau,\tau)`$，
使 Lipschitz 项作用在两条轨迹差上，余项正好沿参考轨迹计算。
若训练风险控制 $`\int qw\varepsilon_v^2`$，Cauchy--Schwarz 还要求
$`\int_0^1 e^{2\int_s^1L}/[q(s)w(s)]\mathrm{d}  s<\infty`$。

对相同空间扩散 $`G`$ 的两个 SDE，可采用同步 Brownian 耦合。
若学习漂移 Lipschitz 常数为 $`L_b`$，$`G`$ 的为 $`L_G`$，
令 $`E_t=\mathbb E\lVert \widehat X_t-X_t\rVert^2`$，Itô 公式给出

```math
E_t'\le(2L_b+L_G^2+1)E_t+
                    \mathbb E\lVert \widehat b_t(X_t)-b_t(X_t)\rVert^2.
```

这里使用 $`2x\cdot y\le\lVert x\rVert^2+\lVert y\rVert^2`$。
该估计再经 Gronwall 控制 $`W_2^2`$。
由 Score 换算漂移时，还需将误差乘以相应的扩散矩阵。

<a name="sec-5-5-3"></a>

### 5.5.3 ODE/SDE 数值离散误差

令 $`\widehat\rho_1`$ 为学习场的精确连续解，$`\widehat\rho_1^h`$ 为数值输出律。
ODE 求解器若在随机初值的 $`L^2`$ 意义下具有全局 $`p`$ 阶误差，
则同初值耦合给出
$`W_2(\widehat\rho_1^h,\widehat\rho_1)\le C h^p`$。
经典光滑条件下 Euler、Heun、RK4 的阶数为 $`1,2,4`$。
SDE 的 Euler–Maruyama 一般具有均方强阶 $`1/2`$，
故相应端点 $`W_2`$ 上界可取 $`C\sqrt h`$，附加正则性可改善阶数。
具体条件和强弱收敛定义见第 [3.8](chapter-3.md#sec-3-8) 节与 〔[20](#ref-higham2001sde)〕。

估计 KL 离散误差时，可对加性、非退化且两边相同的扩散使用连续 Euler 插值。
参考漂移为 $`b(X_t,t)`$，插值漂移为
$`\widehat b(X_{\lfloor t\rfloor_h},\lfloor t\rfloor_h)`$。
把二者差分解成当前状态的场误差与网格冻结误差，
在空间 Lipschitz、时间正则和增量矩控制下，
$`\mathbb E\lVert X_t-X_{\lfloor t\rfloor_h}\rVert^2\le Ch`$，
积分后的冻结项可产生 $`O(h)`$ 的路径 KL 上界。
这个论证需要匹配扩散及 Girsanov 条件；状态依赖扩散的离散化还需相应专门分析。

<a name="sec-5-5-4"></a>

### 5.5.4 初始分布与终端分布匹配误差

以最终数据目标 $`\mu_0`$ 和平滑终点 $`\rho_1^\epsilon`$ 为例，
用三角不等式把数值输出误差写为

<a name="eq-unified-error-budget"></a>

```math
W_2(\widehat\rho_1^h,\mu_0)
 \le W_2(\widehat\rho_1^h,\widehat\rho_1)
     +W_2(\widehat\rho_1,\rho_1^\epsilon)
     +W_2(\rho_1^\epsilon,\mu_0).
\tag{5.13}
```

中间项再由式 [(5.12)](#eq-unified-w2-bound) 拆出先验匹配误差与场误差。
若 $`X_\epsilon=\alpha_\epsilon D+\sigma_\epsilon\varepsilon`$，
使用同一 $`D,\varepsilon`$ 的耦合可得

```math
W_2(p_\epsilon,\mu_0)
 \le |\alpha_\epsilon-1|\,\bigl(\mathbb E\lVert D\rVert^2\bigr)^{1/2}
                       +\sigma_\epsilon\sqrt d.
```

这需要数据二阶矩有限。
若最后另加去噪映射 $`T`$，实际目标是 $`T_\#p_\epsilon`$，
应分析该映射的推前误差；回归后验均值可能改变输出方差。
有限时间前向扩散的终止律与选定先验之间也需计入初始失配。

<a name="sec-5-5-5"></a>

### 5.5.5 Wasserstein 与 KL 误差上界

设 $`P`$ 为精确生成 SDE 路径律、$`\widehat P`$ 为近似路径律，
两者具有相同非退化扩散 $`a`$，且满足 Girsanov 绝对连续与可积条件。
路径相对熵分解为

<a name="eq-unified-path-kl"></a>

```math
D_{\mathrm{KL}}(P\|\widehat P)
 =D_{\mathrm{KL}}(P_0\|\widehat P_0)
 +\tfrac12\mathbb E_P\int_0^1
       \lVert a_\tau^{-1/2}(b_\tau-\widehat b_\tau)(Y_\tau)\rVert^2\mathrm{d} \tau.
\tag{5.14}
```

终点投影给出 $`D_{\mathrm{KL}}(P_1\|\widehat P_1)
\le D_{\mathrm{KL}}(P\|\widehat P)`$。
对各向同性反向扩散，漂移差为 $`g^2(s-s_\theta)`$，
二次能量的时间权重为 $`g^2/2`$。
这种带噪路径上的 $`L^2`$ 控制是扩散生成误差分析的重要工具，
更完整的离散复杂度结论还需相应假设 〔[24](#ref-chen2023sampling)〕。

Pinsker 不等式给出 $`\operatorname{TV}(P_1,\widehat P_1)
\le\sqrt{D_{\mathrm{KL}}(P_1\|\widehat P_1)/2}`$，
这里全变差取 $`\sup_A|P_1(A)-\widehat P_1(A)|`$。
若要由 KL 控制 $`W_2`$，可另假设参考测度满足
$`W_2^2(\nu,Q)\le2C D_{\mathrm{KL}}(\nu\|Q)`$ 的运输不等式。
这是一项额外结构条件，其常数由参考分布决定。
端点奇异的算例说明，不同距离对同一极限可能给出不同性质。

<a name="sec-5-6"></a>

## 5.6 各向异性及状态依赖扩散

<a name="sec-5-6-1"></a>

### 5.6.1 标量扩散系数向矩阵扩散系数推广

设 $`G(x,t)\in\mathbb R^{d\times m}`$。
噪声增量的协方差为 $`G\,\mathbb E[\Delta W\Delta W^\top]G^\top=a\Delta t`$，
所以各方向的扩散强度由 $`a`$ 的特征值决定，相关性由非对角项表达。
标量各向同性扩散对应 $`a=g^2I_d`$。
固定半正定矩阵可通过特征分解选择平方根；正定矩阵也可使用 Cholesky 分解。
退化扩散只在部分方向直接注入噪声。
密度能否在所有方向平滑，还取决于漂移与噪声方向的耦合，需要额外可达性或低正则理论。

<a name="sec-5-6-2"></a>

### 5.6.2 $`G(x,t)G(x,t)^\top`$ 与扩散矩阵

密度生成元只通过 $`a=GG^\top`$ 依赖噪声系数。
若 $`O(x,t)`$ 为正交矩阵，$`\widetilde G=GO`$ 给出相同 $`a`$。
在对应鞅问题唯一且系数满足所需条件时，它们定义相同的弱路径律。
如果在同一个 Brownian 样本上逐点比较，噪声因子的选择仍会改变强耦合。
对于数值同步耦合和路径误差，应记录实际使用的 $`G`$。

若 $`a`$ 可逆，漂移误差的自然噪声能量为
$`e_f^\top a^{-1}e_f`$。若 $`a`$ 退化，
控制只能在 $`\operatorname{Im}G`$ 中改变漂移，并应使用最小噪声坐标控制的能量。
沿核空间的漂移改变可能破坏路径律的绝对连续性。
把矩阵逆换成广义逆时，还必须检查误差位于可控像空间及相应 Girsanov 条件。

<a name="sec-5-6-3"></a>

### 5.6.3 非常系数扩散的 Fokker–Planck 方程

扩散项应逐坐标计算 $`\frac12\sum_{ij}\partial_{ij}(a_{ij}p)`$。
对对称 $`a`$，乘积法则展开为

<a name="eq-unified-variable-diffusion"></a>

```math
\tfrac12\sum_{ij}\partial_{ij}(a_{ij}p)
 =\tfrac12 a:\nabla^2p+(\operatorname{div}a)\cdot\nabla p
                      +\tfrac12p\sum_{ij}\partial_{ij}a_{ij}.
\tag{5.15}
```

空间常系数时后两项消失。
一维取 $`a(x)=1+x^2`$，扩散项为
$`\frac12(1+x^2)p''+2xp'+p`$。
若给定的随机方程采用 Stratonovich 漂移，应先换算成 Itô 漂移，再使用此方程。
这两种修正分别来自随机积分约定和密度方程中的空间乘积导数。

<a name="sec-5-6-4"></a>

### 5.6.4 空间依赖扩散中的概率流速度

给定目标密度 $`\pi(x)>0`$ 与协方差 $`a(x)`$，
选择零概率流的平稳漂移

<a name="eq-unified-stationary-drift"></a>

```math
f(x)=\tfrac12\operatorname{div}a(x)+\tfrac12a(x)\nabla\log\pi(x).
\tag{5.16}
```

代入 $`j=fp-\operatorname{div}(ap)/2`$，在 $`p=\pi`$ 时得 $`j=0`$，
从而在适定和边界条件下保持 $`\pi`$。
若添加满足 $`\nabla\cdot(\pi w)=0`$ 的漂移 $`w`$，
仍可保持同一平稳边缘并产生非零稳态概率流。
因此指定平稳分布后，漂移还具有一定的设计自由度。

**例 5.5 · 二维状态依赖矩阵扩散**

令 $`z=(x,y)^\top`$、$`a(z)=I_2+zz^\top`$、
$`\pi=\mathcal N(0,I_2)`$。逐行求导得到 $`\operatorname{div}a=3z`$，
于是零流漂移为

```math
f(z)=\tfrac12[3z-(I_2+zz^\top)z]
      =z-\tfrac12\lVert z\rVert^2z.
```

直接计算每一分量 $`f_i\pi-\frac12\sum_j\partial_j(a_{ij}\pi)=0`$。
取 $`V(z)=\lVert z\rVert^2`$，生成元满足
$`LV=2+3\lVert z\rVert^2-\lVert z\rVert^4`$，其耗散项可用于非爆破矩控制。
这也给出包含非对角扩散项的可验证模型。

<a name="sec-5-6-5"></a>

### 5.6.5 各向异性情况下的逆时间动力学

在正则时间反演条件下，生成时间的漂移与扩散为

<a name="eq-unified-anisotropic-reverse"></a>

```math
\mathrm{d}  Y_\tau=[-f_t+\operatorname{div}a_t+a_ts_t](Y_\tau)\mathrm{d} \tau
                           +G_t(Y_\tau)\mathrm{d} \overline W_\tau,
 \qquad t=1-\tau.
\tag{5.17}
```

协方差由原 $`a_t`$ 按时间重标记，漂移中包含矩阵散度和矩阵乘 Score
〔[9](#ref-haussmann1986time)〕。
若参考扩散退化，时间反演仍可能成立，但需要针对实际密度和正则性的定理，
不能仅凭形式代数就推出反向 SDE 的存在唯一性。
在线性约束子空间上的模型，可先选独立坐标降低维数，再应用相应的非退化公式。

<a name="sec-5-7"></a>

## 5.7 耦合动力学与物理约束

<a name="sec-5-7-1"></a>

### 5.7.1 耦合 Drift 与状态变量

设联合状态 $`Z_t=(U_t,V_t)\in\mathbb R^{d_1+d_2}`$，考虑

```math
\mathrm{d}  U_t=F_t(U_t,V_t)\mathrm{d}  t+G_t\mathrm{d}  W_t^{(1)},\qquad
 \mathrm{d}  V_t=H_t(U_t,V_t)\mathrm{d}  t+B_t\mathrm{d}  W_t^{(2)}.
```

耦合可以来自漂移对另一状态的依赖，也可以来自噪声相关性。
若 $`\mathbb E[\mathrm{d}  W^{(1)}(\mathrm{d}  W^{(2)})^\top]=R\mathrm{d}  t`$，
联合扩散协方差为

<a name="eq-unified-block-diffusion"></a>

```math
a_t=\begin{pmatrix}GG^\top&GRB^\top\\BR^\top G^\top&BB^\top\end{pmatrix}.
\tag{5.18}
```

Brownian 协方差块矩阵必须半正定，常系数情形等价于 $`\lVert R\rVert_{\mathrm{op}}\le1`$。
实现时可直接对整个合法协方差取平方根，生成一致的联合噪声。
若指定了相关噪声，就需在联合密度与训练样本中保留这种相关性。

<a name="sec-5-7-2"></a>

### 5.7.2 耦合随机过程的联合分布演化

记联合密度为 $`p_t(u,v)`$，协方差块为 $`a_{UU},a_{UV},a_{VV}`$。
联合 Fokker–Planck 方程为

<a name="eq-unified-joint-fp"></a>

```math
\begin{aligned}
 \partial_t p={}&-\nabla_u\cdot(Fp)-\nabla_v\cdot(Hp)\\
 &+\tfrac12\sum_{ij}\partial_{u_i u_j}((a_{UU})_{ij}p)
 +\sum_{ij}\partial_{u_i v_j}((a_{UV})_{ij}p)\\
 &+\tfrac12\sum_{ij}\partial_{v_i v_j}((a_{VV})_{ij}p).
 \end{aligned}
\tag{5.19}
```

交叉项的系数来自对称的两个混合块。
若在 $`v`$ 方向积分且边界项消失，$`U`$ 的边缘漂移为
$`\overline F_t(u)=\mathbb E[F_t(U,V)\mid U=u]`$，
边缘扩散块为 $`\mathbb E[a_{UU}(U,V,t)\mid U=u]`$。
联合 Score 的 $`u`$ 分量对 $`V\mid U=u`$ 平均后才得到边缘 Score。
因此逐变量独立训练需要检查它是否保留了所需的联合结构。

**例 5.6 · 相关 OU 的联合 Score**

取一维 $`U,V`$，漂移分别为 $`-U,-V`$，两个噪声幅度均为 $`\sqrt2`$，
Brownian 相关系数 $`\rho\in(-1,1)`$。
平稳协方差是 $`C=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}`$，满足
$`-C-C+2C=0`$。其联合 Score 为

```math
s(u,v)=-\frac1{1-\rho^2}
           \begin{pmatrix}u-\rho v\\v-\rho u\end{pmatrix}.
```

两个边缘 Score 分别为 $`-u,-v`$。
联合概率流为零，因为 $`f=-z`$、$`a=2C`$、$`s=-C^{-1}z`$。
这个模型可同时核对联合协方差、交叉扩散项与 Score 边缘化。

<a name="sec-5-7-3"></a>

### 5.7.3 物理约束下的速度场与 Score

对于线性约束 $`Az=c`$，取一个特解 $`z_*`$ 和
$`\ker A`$ 的正交基矩阵 $`Q`$，写成 $`z=z_*+Q\xi`$。
在独立坐标 $`\xi`$ 上构造密度、Score 和生成动力学，再映回原空间，
即可精确满足约束。切向速度与协方差可分别取
$`u_z=Qu_\xi`$、$`a_z=Qa_\xi Q^\top`$。
原空间分布可能没有全维密度，其切向 Score 为 $`Q\nabla_\xi\log p_\xi`$。
使用子空间坐标可以明确相应导数与参考测度。

对光滑非线性约束 $`C(z)=0`$，Itô 公式给出保持约束所需的局部条件

<a name="eq-unified-nonlinear-constraint"></a>

```math
\nabla C(z)G(z)=0,\qquad
 \nabla C(z)f(z)+\tfrac12 a(z):\nabla^2 C(z)=0
 \quad\text{在约束流形上}.
\tag{5.20}
```

多个约束逐分量满足。第一式控制随机切向性，第二式包含曲率产生的漂移补偿。
这些条件与流形正则性、SDE 适定性共同用于证明不变性。
软惩罚可以用于近似约束，其误差需要通过实际残差评估。

**例 5.7 · 单位球面上的 Itô 曲率漂移**

在 $`\lVert z\rVert=1`$ 上，取 $`P(z)=I_d-zz^\top`$，令

```math
\mathrm{d}  Z_t=-\tfrac{d-1}{2}Z_t\mathrm{d}  t+P(Z_t)\mathrm{d}  W_t.
```

对 $`C(z)=\lVert z\rVert^2-1`$，随机项为 $`2z^\top P\mathrm{d}  W=0`$，
漂移项为 $`-(d-1)\lVert z\rVert^2+\operatorname{tr}(P^2)=0`$。
因而该曲率补偿与切向噪声相容。只把环境 Brownian 增量投影到切空间时，
还需考虑同样的 Itô 二阶项。

<a name="sec-5-7-4"></a>

### 5.7.4 守恒律、边界条件与生成动力学

无界空间的衰减、周期边界或有界域的无通量条件可以保证概率质量守恒。
对反射扩散，随机方程含边界局部时项，密度满足相应反射通量条件；
反向动力学也要包含对应的边界处理。
吸收边界会改变域内存活质量，条件于存活后的归一化分布还会引入额外修正。
这些情形需按边界模型重新建立方程。

状态空间中的物理守恒量 $`E(z)`$ 与概率总质量具有不同定义。
确定性生成流要保持 $`E`$，可要求 $`\nabla E\cdot u=0`$。
SDE 则需同时满足
$`\nabla E^\top G=0`$ 与
$`\nabla E\cdot f+\frac12a:\nabla^2E=0`$。
若端点数据与先验处于不同的守恒水平集，严格保持该量的路径无法连接它们。
可先对守恒量条件化，或设计具有明确源项的路径。

<a name="sec-5-7-5"></a>

### 5.7.5 生成模型与经典 PDE 之间的联系

Fokker–Planck 方程描述状态空间中的概率密度，物理 PDE 描述空间场随物理时间的变化。
若物理场先做 Galerkin 截断为系数 $`z(r)`$，并满足
$`\dot z=F_{\mathrm{phys}}(z)`$，其随机初值的系数分布满足 Liouville 方程。
加入物理随机强迫 $`B\mathrm{d}  W_r`$，系数分布满足相应 Fokker–Planck 方程。
生成模型可以学习这些快照分布或条件转移分布，物理时间 $`r`$ 的一致性还需另外检验。

对于 $`f=\kappa\nabla\log\pi`$、$`a=2\kappa I_d`$，有
$`u=-\kappa\nabla\log(p/\pi)`$。
在积分可交换、边界项消失时，令 $`\mathcal H(p\mid\pi)=\int p\log(p/\pi)`$，
使用质量守恒及分部积分可得

<a name="eq-unified-entropy-dissipation"></a>

```math
\frac{\mathrm{d} }{\mathrm{d}  t}\mathcal H(p_t\mid\pi)
 =-\kappa\int p_t\lVert \nabla\log(p_t/\pi)\rVert^2\mathrm{d}  x\le0.
\tag{5.21}
```

这把扩散的平衡趋向与相对 Fisher 信息联系起来。
熵单调性本身还需结合额外不等式，才能推出具体指数收敛速率。

<a name="sec-5-8"></a>

## 5.8 条件生成、逆问题与研究应用

<a name="sec-5-8-1"></a>

### 5.8.1 Conditional Score 与 Conditional Flow Matching

设条件信息为 $`c`$，固定条件边缘族 $`p_t(x\mid c)`$。
若噪声独立于数据与条件，$`X_t=\alpha_tD+\sigma_t\varepsilon`$，则

<a name="eq-unified-conditional-fields"></a>

```math
s_t(x\mid c)=-\frac1{\sigma_t}\mathbb E[\varepsilon\mid X_t=x,c],\qquad
 u_t(x\mid c)=\mathbb E[\dot X_t\mid X_t=x,c].
\tag{5.22}
```

条件 Score 可通过带条件的噪声回归估计，条件速度可通过带条件的路径导数回归估计。
CFM 的端点、耦合及辅助变量都要按相同条件联合律抽取。
精确条件生成需从 $`p_1(\cdot\mid c)`$ 出发。
只有当噪声端与 $`c`$ 独立时，所有条件才共享同一个精确先验。
有限噪声时采用统一高斯起点所产生的条件端点失配，应进入误差预算。

<a name="sec-5-8-2"></a>

### 5.8.2 Posterior Sampling 与逆问题

设干净未知量 $`D`$ 的先验为 $`\mu_0`$，观测满足
$`y=\mathcal A(D)+\eta`$，观测噪声密度已知。
目标后验为 $`p_0(d\mid y)\propto p(y\mid d)p_0(d)`$。
在噪声时刻的精确 Bayes 分解为

<a name="eq-unified-posterior-score"></a>

```math
s_t(x\mid y)=s_t(x)+\nabla_x\log p_t(y\mid x),\qquad
 p_t(y\mid x)=\int p(y\mid d)\,p_0(\mathrm{d}  d\mid X_t=x).
\tag{5.23}
```

这里的似然已经对干净数据后验积分。
若用去噪均值 $`d_t^*(x)`$ 代入 $`p(y\mid d_t^*(x))`$，
得到的是一个可计算近似，忽略了条件不确定性及非线性平均效应。
DPS 等方法研究这种后验采样近似 〔[27](#ref-chung2023dps)〕。

**例 5.8 · 线性高斯逆问题的精确噪声似然**

取 $`D\sim\mathcal N(0,C)`$、$`X_t=\alpha_tD+\sigma_t\varepsilon`$，
$`y=AD+\eta`$、$`\eta\sim\mathcal N(0,R)`$，其中 $`R,C`$ 正定且所有噪声独立。
记 $`H_t=\alpha_t^2C+\sigma_t^2I_d`$，则

```math
m_t(x)=\alpha_tCH_t^{-1}x,\qquad
 C_{D\mid X}=C-\alpha_t^2CH_t^{-1}C.
```

于是 $`y\mid X_t=x`$ 为高斯，均值 $`Am_t(x)`$，协方差
$`S_t=R+AC_{D\mid X}A^\top`$。精确的似然梯度为

<a name="eq-unified-linear-likelihood"></a>

```math
\nabla_x\log p_t(y\mid x)
 =(\alpha_tCH_t^{-1})^\top A^\top S_t^{-1}(y-Am_t(x)).
\tag{5.24}
```

用均值直接代入干净似然会把 $`S_t`$ 替换成 $`R`$。
在一维 $`C=R=A=1`$、$`\alpha_t=\sigma_t=1/\sqrt2`$ 时，
$`m_t=x/\sqrt2`$、$`C_{D\mid X}=1/2`$、$`S_t=3/2`$。
精确梯度为 $`\sqrt2(y-x/\sqrt2)/3`$，均值代入梯度为
$`(y-x/\sqrt2)/\sqrt2`$，其幅度相差 $`3/2`$。

<a name="sec-5-8-3"></a>

### 5.8.3 Classifier Guidance 与 Classifier-Free Guidance

若分类器精确估计带噪条件概率 $`p_t(c\mid x)`$，Bayes 公式给出
$`s_t(x\mid c)=s_t(x)+\nabla_x\log p_t(c\mid x)`$。
Classifier Guidance 使用该条件梯度修正生成所需的 Score
〔[25](#ref-dhariwal2021guided)〕。
Classifier-Free Guidance 联合训练有条件与无条件预测，再在同一参数化下组合

<a name="eq-unified-cfg"></a>

```math
s_t^{(w)}=s_t+w(s_t(\cdot\mid c)-s_t),\qquad w\ge0.
\tag{5.25}
```

本书取 $`w=1`$ 为普通条件 Score，$`w=0`$ 为无条件 Score；
不同实现的 guidance 参数需先换算到这一约定。
若用噪声预测组合且两支共享同一 $`\sigma_t`$，对应的 Score 组合相容
〔[26](#ref-ho2022cfg)〕。

在单个时刻，精确场满足
$`s_t^{(w)}=\nabla\log[p_t(x)p_t(c\mid x)^w]`$。
若该密度可归一化，这描述了一个时刻的倾斜密度。
不同时间的这些密度还需共同满足所用前向扩散方程，才能推出对应终点采样结论。
例如对满足 $`(\partial_t+L_t)h=0`$ 的正势，乘积法则给出

```math
(\partial_t+L_t)h^w
 =\tfrac12w(w-1)h^{w-2}(\nabla h)^\top a_t\nabla h.
```

一般 $`w`$ 的势幂会产生额外空间相关项。
因此放大 guidance 后的实际终点律要结合完整动力学研究。
在共享高斯路径上组合 FM 速度，也需通过第 4 章的已知转换系数核对其含义。

<a name="sec-5-8-4"></a>

### 5.8.4 物理条件下的概率路径设计

条件可以是边界值、守恒量、外力、参数或稀疏传感器读数。
对于固定的仿射约束集 $`Az=c`$，可选 $`z_*(c)`$ 与核空间基 $`Q`$，
在 $`z=z_*(c)+Q\xi`$ 上定义条件先验、数据端和插值。
只要端点和所有辅助噪声都位于该仿射子空间，线性插值就保持约束。
对于非凸或非线性约束，端点可行并不能保证中间线段可行，
需使用坐标图、合适的流形插值或带约束的动力学。

**例 5.9 · 固定总量的三维状态**

令约束为 $`z_1+z_2+z_3=m`$，取

```math
z_*=(m/3,m/3,m/3)^\top,\qquad
 Q=\begin{pmatrix}
 1/\sqrt2&1/\sqrt6\\-1/\sqrt2&1/\sqrt6\\0&-2/\sqrt6
 \end{pmatrix}.
```

有 $`Q^\top Q=I_2`$、$`(1,1,1)Q=0`$。
在 $`\xi\in\mathbb R^2`$ 上训练任意合法条件生成路径，
映回 $`z_*+Q\xi`$ 后即保持总量 $`m`$。
这保留了总量约束；若还要求每个分量非负，需进一步限制独立坐标的可行区域。

观测包含噪声时，可以通过观测似然表达软条件，并检验后验预测覆盖率。
若把有噪声观测作为精确等式强制满足，就相当于改变了所求的统计模型。
条件化的每个步骤都应对应明确的数据或物理假设。

<a name="sec-5-8-5"></a>

### 5.8.5 面向流场、涡度场和随机动力系统的生成建模

先固定空间网格或有限维 Galerkin 基，把状态写成系数 $`z`$。
对于周期不可压缩速度场，非零 Fourier 模式可用 Leray 投影

<a name="eq-unified-leray"></a>

```math
P(k)=I-\frac{kk^\top}{|k|^2},\qquad
 \widehat u(k)=P(k)\widehat q(k),\quad k\ne0.
\tag{5.26}
```

它满足 $`k^\top\widehat u(k)=0`$。零频模式对应平均流，需要另行设定或条件化。
实值场还要保持 Fourier 共轭对称性。
可以在无散基的独立系数上训练 FM 或扩散，从表示上保证离散不可压缩性。

对二维周期流，取零均值涡度 $`\omega`$，解
$`-\Delta\psi=\omega`$，并令
$`u=(\partial_y\psi,-\partial_x\psi)+\overline u`$。
于是 $`\nabla\cdot u=0`$、$`\partial_xu_2-\partial_yu_1=\omega`$。
非零模式满足
$`\widehat\psi(k)=\widehat\omega(k)/|k|^2`$。
相应的速度能量使用 $`|k|^{-2}|\widehat\omega(k)|^2`$ 权重，
涡度平方误差则使用等权重，因此两种损失对尺度的关注不同。
有限维上的存在性与数值验证并不能自动给出无限维 SPDE 的极限结论，
网格加密时还需控制紧性、函数空间正则性及离散物理算子的一致性。

可把研究任务落实为下列流程。

```text
Choose a spatial discretization and physical variables.
Specify the data law, conditions, prior, and endpoint coupling.
Build a probability path in admissible coordinates.
Derive velocity and Score labels from that joint law.
Check the continuity or Fokker-Planck equation and endpoint limits.
Train the chosen conditional regressors.
Integrate the sampling dynamics with recorded numerical tolerances.
Evaluate distributions, physical residuals, and uncertainty coverage.
Repeat on refined grids and test conditions.
```

对快照模型，评价边缘统计、能谱、守恒残差和条件覆盖；
对转移模型，再检查多步时间相关性、稳定性与物理时间一致性。
可以研究耦合选择、矩阵扩散或条件似然近似如何影响这些量，
并用本章的回归、正则性和误差框架逐项建立可验证的问题。

### 模型与理论条件对照

| 对象 | 主要学习量 | 需核对的理论条件 |
| --- | --- | --- |
| DDPM / DDIM | 噪声或去噪预测 | 实际噪声边缘、反向核近似、先验匹配与网格 |
| Score SDE | 边缘 Score | 时间反演、扩散正则性、Score 误差与端点截断 |
| FM / CFM | 边缘速度 | 条件路径与耦合、连续性方程、流的适定性 |
| Rectified Flow | 插值速度 | 回归投影、教师端点误差、轨迹加速度与少步误差 |
| Schrödinger Bridge | 控制漂移或双势 | 双端约束、参考路径律、路径 KL 与迭代收敛 |
| 随机插值 | 速度与 Score | 路径导数可积性、内部平滑、构造扩散的适定性 |
| 条件生成 | 条件场或似然梯度 | 条件联合律、条件起点、噪声似然与 guidance 定义 |

**本章小结**

- 概率流将密度方程统一为守恒形式，适定性与匹配初值使该形式对应真实概率演化。
- 路径、耦合、动力学和优化泛函共同决定模型，单时刻边缘只包含其中部分信息。
- 从回归到生成需要控制总体场误差、稳定性、数值离散、先验失配和端点极限。
- 矩阵扩散与联合状态要求保留空间导数和交叉协方差，物理约束还涉及切向性和 Itô 修正。
- 条件生成使用带噪条件分布；后验似然积分、端点条件和 guidance 强度需与动力学一致。
- 流场研究可先在有限维可行坐标中建立模型，再分别验证分布、物理结构与网格收敛。

### 本章练习与核对

1. 对 $`a=2\kappa I_d`$，从概率流公式恢复给定 $`p_t,u_t`$ 的前向与生成 SDE 漂移。<br>
 **核对**  分别为 $`u_t+\kappa s_t`$ 与 $`-u_t+\kappa s_t`$，第二式在 $`t=1-\tau`$ 求值。
1. 计算平稳 OU 过程的两时刻协方差，并与常值标准高斯过程比较。<br>
 **核对**  分别为 $`e^{-|t-s|}I_d`$ 与 $`I_d`$。
1. 取 $`\gamma(t)=ct(1-t)`$，计算 $`\int_0^1\dot\gamma(t)^2\mathrm{d}  t`$。<br>
 **核对**  $`c^2/3`$，辅助噪声导数的时间平方积分为 $`dc^2/3`$。
1. 对到达点质量的高斯生成流，求停止于 $`t=\epsilon`$ 的二阶运输误差。<br>
 **核对**  $`W_2^2=d\epsilon`$。
1. 在 $`v_\theta`$ 的 Lipschitz 常数为 $`L`$、场误差范数恒为 $`\delta`$、初始匹配时，化简 ODE 误差界。<br>
 **核对**  $`W_2\le\delta(e^L-1)/L`$，$`L=0`$ 时取 $`\delta`$。
1. 令一维 $`a=1+x^2`$，完整展开 $`\tfrac12(ap)''`$。<br>
 **核对**  $`\tfrac12(1+x^2)p''+2xp'+p`$。
1. 对 $`a(z)=I_2+zz^\top`$、标准高斯目标，计算 $`\operatorname{div}a`$、零流漂移和 $`L\lVert z\rVert^2`$。<br>
 **核对**  $`3z`$、$`z-\lVert z\rVert^2z/2`$、$`2+3\lVert z\rVert^2-\lVert z\rVert^4`$。
1. 对 $`\lVert z\rVert=1`$，验证球面噪声矩阵 $`P=I-zz^\top`$ 满足 $`P^2=P`$。<br>
 **核对**  $`Pz=0`$、$`\operatorname{tr}P=d-1`$，据此核对曲率漂移。
1. 在线性高斯逆问题算例中，取 $`x=0,y=1`$，计算精确与均值代入的似然梯度。<br>
 **核对**  分别为 $`\sqrt2/3`$ 与 $`1/\sqrt2`$。
1. 对正势 $`h`$ 与生成元 $`L`$，展开 $`(\partial_t+L)h^w`$，说明 $`w=1`$ 的作用。<br>
 **核对**  当 $`(\partial_t+L)h=0`$ 时，剩余项为
 $`w(w-1)h^{w-2}(\nabla h)^\top a\nabla h/2`$。
1. 检查总量约束算例中的 $`Q^\top Q`$ 和 $`(1,1,1)Q`$。<br>
 **核对**  分别为 $`I_2`$ 和零行向量。
1. 对非零 Fourier 模式，证明 Leray 投影对称且幂等，并核对涡度恢复速度的符号。<br>
 **核对**  $`P^\top=P`$、$`P^2=P`$、$`k^\top P=0`$，
 $`u=(\partial_y\psi,-\partial_x\psi)`$ 满足 $`\operatorname{curl}u=-\Delta\psi=\omega`$。

<a name="references"></a>

## 参考文献

<a name="ref-song2021sde"></a>
**[3]** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. ICLR, 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456).

<a name="ref-tong2024otcfm"></a>
**[7]** Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. Transactions on Machine Learning Research, 2024. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482).

<a name="ref-albergo2025si"></a>
**[8]** Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic Interpolants: A Unifying Framework for Flows and Diffusions. Journal of Machine Learning Research, 26(209), 2025. [论文](https://jmlr.org/papers/v26/23-1605.html).

<a name="ref-haussmann1986time"></a>
**[9]** U. G. Haussmann and É. Pardoux. Time Reversal of Diffusions. The Annals of Probability, 14(4), 1188–1205, 1986. [论文](https://www.i2m.univ-amu.fr/perso/etienne.pardoux/_media/trd.pdf).

<a name="ref-benamou2000ot"></a>
**[11]** Jean-David Benamou and Yann Brenier. A Computational Fluid Mechanics Solution to the Monge–Kantorovich Mass Transfer Problem. Numerische Mathematik, 84, 375–393, 2000. [DOI](https://doi.org/10.1007/s002110050002).

<a name="ref-leonard2014schrodinger"></a>
**[13]** Christian Léonard. A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport. Discrete and Continuous Dynamical Systems, 34(4), 1533–1574, 2014. [arXiv:1308.0215](https://arxiv.org/abs/1308.0215).

<a name="ref-higham2001sde"></a>
**[20]** Desmond J. Higham. An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations. SIAM Review, 43(3), 525–546, 2001. [DOI](https://doi.org/10.1137/S0036144500378302).

<a name="ref-trevisan2016wellposed"></a>
**[23]** Dario Trevisan. Well-posedness of Multidimensional Diffusion Processes with Weakly Differentiable Coefficients. Electronic Journal of Probability, 21, Paper 22, 1–41, 2016. [arXiv:1507.01357](https://arxiv.org/abs/1507.01357).

<a name="ref-chen2023sampling"></a>
**[24]** Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R. Zhang. Sampling is as Easy as Learning the Score: Theory for Diffusion Models with Minimal Data Assumptions. ICLR, 2023. [arXiv:2209.11215](https://arxiv.org/abs/2209.11215).

<a name="ref-dhariwal2021guided"></a>
**[25]** Prafulla Dhariwal and Alex Nichol. Diffusion Models Beat GANs on Image Synthesis. NeurIPS, 2021. [arXiv:2105.05233](https://arxiv.org/abs/2105.05233).

<a name="ref-ho2022cfg"></a>
**[26]** Jonathan Ho and Tim Salimans. Classifier-Free Diffusion Guidance. arXiv preprint, 2022. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598).

<a name="ref-chung2023dps"></a>
**[27]** Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, and Jong Chul Ye. Diffusion Posterior Sampling for General Noisy Inverse Problems. ICLR, 2023. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687).
