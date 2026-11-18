# 第 1 章　各生成模型的训练—生成整体框架
[返回项目目录](../README.md#contents) · [上一章：数学基础](chapter-0.md) · [下一章：训练理论](chapter-2.md) · [PDF 版](../output/pdf/model-study.pdf) · [LaTeX 源码](../chapters/01-model-overview.tex)

[1.1 DDPM：离散扩散概率模型](#sec-1-1) · [1.2 DDIM：隐式扩散生成模型](#sec-1-2) · [1.3 Score-based Diffusion / Score SDE](#sec-1-3) · [1.4 Continuous Normalizing Flow](#sec-1-4) · [1.5 Flow Matching 与 Conditional Flow Matching](#sec-1-5) · [1.6 Rectified Flow 与 OT-CFM](#sec-1-6) · [1.7 Schrödinger Bridge](#sec-1-7) · [1.8 Stochastic Interpolants](#sec-1-8)

本章对每种模型追问四件事。它怎样连接数据与先验，训练样本如何取得，
网络学习哪个数学对象，以及给定训练好的网络后怎样生成样本。
正向时间 $`t`$ 从原始数据走向噪声端点。连续采样统一用
$`\tau=1-t`$，从 $`Y_0`$ 出发，在 $`Y_1`$ 读出样本。
离散模型写第 $`k`$ 步的样本 $`x_k`$，相应随机变量简写为 $`X_k`$。
其中 $`x_0`$ 是干净数据，$`x_K`$ 是终止状态，归一化时间为 $`t_k=k/K`$。
本章沿用第 [0.6](chapter-0.md#sec-0-6) 节的端点、漂移和速度记号。

> **边注 · 阅读模型公式**
>
> 计算网络输入前，先确定它接受的是离散步 $`k`$、加噪时间 $`t`$ 还是生成时间 $`\tau`$。三者必须与训练接口一致。

**学习目标**

- 区分训练时采样概率路径与生成时求解动力系统。
- 能写出 DDPM、DDIM 的一步更新，说明共同训练目标如何支持不同采样器。
- 能把 Score、速度和受控漂移与各自的生成方程对应起来。
- 用端点、中间边缘及路径律比较 CNF、FM、Schrödinger Bridge 与随机插值。

<a name="sec-1-1"></a>

## 1.1 DDPM：离散扩散概率模型

<a name="sec-1-1-1"></a>

### 1.1.1 原始数据、噪声先验与离散时间链

DDPM 〔[1](#ref-ho2020ddpm)〕 先规定数据如何逐步受到高斯扰动，再学习从高噪声状态
向低噪声状态移动的条件分布。用 $`q`$ 记给定的训练加噪律，
用 $`p_\theta`$ 记学习到的生成模型。选取 $`K`$ 个方差 $`0<\beta_k<1`$，记

```math
r_k=1-\beta_k,\qquad \bar r_k=\prod_{j=1}^{k}r_j,\qquad
  A_k=\sqrt{\bar r_k},\quad S_k=\sqrt{1-\bar r_k},
  \qquad \bar r_0=1 .
```

连续时间记号与之对应为 $`\alpha_{t_k}=A_k,\ \sigma_{t_k}=S_k`$。
选用标准高斯先验时，理想上希望 $`\bar r_K`$ 很小，使终止边缘接近先验。
有限步中只要 $`r_k>0`$，便有 $`A_K>0`$。终止边缘与先验的偏差可用
[(0.32)](chapter-0.md#eq-foundations-prior-bound) 讨论。

<a name="sec-1-1-2"></a>

### 1.1.2 前向加噪马尔可夫过程

给定 $`x_0`$，前向链按高斯转移及其乘积分解定义

<a name="eq-overview-ddpm-forward"></a>

```math
q(x_k\mid x_{k-1})
    =\mathcal N(\sqrt{r_k}\,x_{k-1},\beta_k I_d),\qquad
  q(x_{1:K}\mid x_0)=\prod_{k=1}^{K}q(x_k\mid x_{k-1}) .
\tag{1.1}
```

条件均值系数逐步相乘，条件方差递推为
$`S_k^2=r_kS_{k-1}^2+\beta_k`$。因此一次采样任意时刻可写为

<a name="eq-overview-ddpm-marginal"></a>

```math
q(x_k\mid x_0)=\mathcal N(A_kx_0,S_k^2I_d),\qquad
  x_k=A_kx_0+S_k\varepsilon,\quad\varepsilon\sim\mathcal N(0,I_d).
\tag{1.2}
```

这里 $`\varepsilon`$ 汇集了前 $`k`$ 步的扰动。对于固定 $`k`$，
公式 [(1.2)](#eq-overview-ddpm-marginal) 与逐步加噪得到同一条件边缘。
若用同一个 $`\varepsilon`$ 生成所有 $`k`$ 的状态，
则选定的是另一种跨时间耦合。训练噪声回归时只需单时刻样本。

<a name="sec-1-1-3"></a>

### 1.1.3 噪声预测网络与训练损失

训练时从数据集采 $`x_0`$，抽取 $`k`$ 与独立高斯 $`\varepsilon`$，
再按 [(1.2)](#eq-overview-ddpm-marginal) 计算 $`x_k`$。
典型网络 $`\varepsilon_\theta(x_k,k)`$ 读取当前带噪样本与步号，
输出与样本同维度的噪声估计。原论文常用的简化目标为

<a name="eq-overview-ddpm-loss"></a>

```math
\mathcal L_{\mathrm{simple}}(\theta)
    =\mathbb E_{k,x_0,\varepsilon}
      \lVert \varepsilon-\varepsilon_\theta(A_kx_0+S_k\varepsilon,k) \rVert^2 .
\tag{1.3}
```

在无限容量的平方损失回归中，最优输出为
$`\mathbb E[\varepsilon\mid X_k=x,k]`$。除以 $`-S_k`$ 得到边缘 Score，
见 [(0.5)](chapter-0.md#eq-foundations-score-noise)。以同一输出还可形成数据估计
$`\widehat x_0=(x_k-S_k\varepsilon_\theta)/A_k`$。
原始变分目标包含先验匹配、逐步后验匹配及数据重构。固定反向方差后，
中间步的高斯 KL 给出有权噪声 MSE。简化目标改用统一权重，
完整 ELBO 与参数化转换在第 2 章展开。

<a name="sec-1-1-4"></a>

### 1.1.4 反向去噪马尔可夫链

生成模型从规定的 $`p_{\mathrm{prior}}(x_K)`$ 出发，逐步使用
$`p_\theta(x_{k-1}\mid x_k)`$。
真实的反转核 $`q(x_{k-1}\mid x_k)`$ 还要按 $`q(x_0\mid x_k)`$
混合条件后验，一般为高斯混合。对 $`k\ge 2`$，给定 $`x_0`$ 时后验是高斯，
方差为 $`\widetilde\beta_k=\beta_k(1-\bar r_{k-1})/(1-\bar r_k)`$。
将前向噪声表达式代入后验均值，再用网络噪声替代真值，
得到常用固定方差 DDPM 的更新

<a name="eq-overview-ddpm-update"></a>

```math
\begin{aligned}
  m_\theta(x_k,k)
    &=\frac{1}{\sqrt{r_k}}\left[x_k-
                     \frac{\beta_k}{S_k}\varepsilon_\theta(x_k,k)\right],\\
  x_{k-1}&=m_\theta(x_k,k)+\sqrt{\nu_k}\,z_k,\qquad
  z_k\sim\mathcal N(0,I_d).
  \end{aligned}
\tag{1.4}
```

中间步的 $`\nu_k`$ 可以按设计取 $`\widetilde\beta_k`$ 或 $`\beta_k`$。
最终一步常用均值输出，即设 $`z_1=0`$。变分模型中数据端似然
仍要单独指定，例如为量化像素定义离散解码概率。

> **边注 · 两种噪声**
>
> 训练的 $`\varepsilon`$ 来自 $`x_0`$ 到 $`x_k`$ 的边缘扰动。 采样的 $`z_k`$ 用于从 $`x_k`$ 向 $`x_{k-1}`$ 的单步转移。

<a name="sec-1-1-5"></a>

### 1.1.5 DDPM 的完整训练与生成算法

训练循环每次采数据、步号和高斯噪声，合成 $`x_k`$，前向运行一次网络，
按 [(1.3)](#eq-overview-ddpm-loss) 反向传播。网络参数在整个时间网格共享，
步号作为条件提示当前噪声强度。生成时固定参数，从 $`x_K\sim p_{\mathrm{prior}}`$
出发，遍历 $`K,K-1,\ldots,1`$，每步重新在当前 $`x_k`$ 上预测噪声，
用 [(1.4)](#eq-overview-ddpm-update) 更新，最终读出 $`x_0`$。
对这套相邻步采样器，一次样本通常需要 $`K`$ 次网络求值。
训练的一次单时刻回归无需运行这条反向链。

| 阶段 | 流程 |
| --- | --- |
| 训练 | 数据、步号、高斯噪声 → 用闭式加噪合成样本 → 回归 $`\varepsilon`$ |
| 生成 | 从高斯先验开始 → 重复相邻反向高斯步 → 得到 $`x_0`$ |

<a name="sec-1-2"></a>

## 1.2 DDIM：隐式扩散生成模型

<a name="sec-1-2-1"></a>

### 1.2.1 DDIM 与 DDPM 的共同训练目标

DDIM 〔[2](#ref-song2021ddim)〕 保留 $`q(x_k\mid x_0)=\mathcal N(A_kx_0,S_k^2I_d)`$，
并改变给定 $`x_0`$ 后各时刻之间怎样联合采样。
因为 [(1.3)](#eq-overview-ddpm-loss) 每项只访问
$`(x_0,k,x_k)`$ 及构成 $`x_k`$ 的噪声，条件边缘保持后，
同一个噪声预测模型可以供两种采样器使用。
复用预训练网络时需要对齐原噪声表、网络时间标识和输出参数化。
例如输出 $`\widehat x_0`$ 或 $`v^{\mathrm{pred}}`$ 的网络须先换算到采样公式要求的量。

<a name="sec-1-2-2"></a>

### 1.2.2 非马尔可夫前向过程

任选较早一步 $`j<k`$，且 $`j\ge1`$。对给定的 $`x_0`$，
若先从正确边缘取 $`X_k`$，再独立抽取 $`z\sim\mathcal N(0,I_d)`$，可令

<a name="eq-overview-ddim-coupling"></a>

```math
X_j=A_jx_0+\sqrt{S_j^2-\omega_{k,j}^2}\,
                    \frac{X_k-A_kx_0}{S_k}
            +\omega_{k,j} z,\qquad 0\le\omega_{k,j}\le S_j .
\tag{1.5}
```

给定 $`x_0`$ 后，标准化残差与 $`z`$ 都是独立标准高斯，
所以右侧均值为 $`A_jx_0`$，方差为 $`S_j^2I_d`$。
由终止边缘和这族依赖 $`(x_k,x_0)`$ 的条件核可以定义联合过程。
边缘约束允许不同的 $`\omega_{k,j}`$，相邻时刻相关性随它改变。
这种构造一般不满足原 DDPM 的正向马尔可夫分解。

<a name="sec-1-2-3"></a>

### 1.2.3 确定性与随机采样参数

生成时以 $`\widehat\varepsilon=\varepsilon_\theta(x_k,k)`$ 重构
$`\widehat x_0=(x_k-S_k\widehat\varepsilon)/A_k`$，
把它代入 [(1.5)](#eq-overview-ddim-coupling)，得到

<a name="eq-overview-ddim-update"></a>

```math
\begin{aligned}
  x_j &= A_j\widehat x_0
          +\sqrt{S_j^2-\omega_{k,j}^2}\,\widehat\varepsilon
          +\omega_{k,j}z,\\
  \omega_{k,j} &=
     \eta\,\sqrt{\frac{1-\bar r_j}{1-\bar r_k}}\,
                   \sqrt{1-\frac{\bar r_k}{\bar r_j}},\qquad 0\le\eta\le1 .
  \end{aligned}
\tag{1.6}
```

$`\eta=0`$ 时，给定 $`x_K`$ 后路径由模型与时间网格确定。
$`\eta>0`$ 时，中间步骤可引入新的高斯随机量。
当 $`\eta=1`$ 且 $`j=k-1`$，上式方差是
$`\omega_{k,k-1}^2=\widetilde\beta_k`$，代数化简可回到取此方差的 DDPM 更新。
令 $`j=0`$ 时 $`A_0=1,\ S_0=0,\ \omega_{k,0}=0`$，直接输出
$`\widehat x_0`$。确定性 DDIM 的样本多样性仍来自不同的初始先验样本。

<a name="sec-1-2-4"></a>

### 1.2.4 时间步跳跃与采样加速

采样时选择严格递减的子序列 $`K=k_M>k_{M-1}>\cdots>k_0=0`$，
在每个选中时刻求值网络，用 [(1.6)](#eq-overview-ddim-update)
跨到下一个选中时刻。一次生成通常调用网络 $`M`$ 次。
时间表可按训练步号或某个噪声坐标选择，系数仍取训练时相应步号的值。
直接把 DDPM 相邻步更新套到非相邻步会改变所用核。

在 $`\eta=0`$ 时设 $`y_k=x_k/A_k,\ \lambda_k=S_k/A_k`$，
用 $`A_j>0`$ 除更新公式，得到

<a name="eq-overview-ddim-rescaled"></a>

```math
y_j=y_k+(\lambda_j-\lambda_k)\,
                        \varepsilon_\theta(A_ky_k,k).
\tag{1.7}
```

当 $`\lambda_j<\lambda_k`$ 时，这形如噪声坐标中的反向 Euler 步。
它说明跳步是在扩大积分步长。减少网络求值次数后，
离散化误差与网络估计误差仍需分别评估。

<a name="sec-1-2-5"></a>

### 1.2.5 DDPM 与 DDIM 的核心差异

比较二者时，先固定网络和训练噪声表，再看采样器。
经典 DDPM 使用相邻反向随机链，DDIM 可以选择子序列和随机强度，
并在 $`\eta=0`$ 得到给定初始噪声的确定性更新。
同一训练边缘支持这些不同路径构造。若要比较分布质量，
还要给出步数、噪声日程、方差策略与网络求值次数。

| 阶段 | 流程 |
| --- | --- |
| 训练 | 与 DDPM 共用边缘加噪 → 按同一目标训练或复用模型 → 得到 $`\widehat\varepsilon`$ |
| 生成 | 先验、子序列与 $`\eta`$ → 估计 $`\widehat x_0`$ 后跳步 → 得到 $`x_0`$ |

<a name="sec-1-3"></a>

## 1.3 Score-based Diffusion / Score SDE

<a name="sec-1-3-1"></a>

### 1.3.1 Score-based 生成模型的基本思想

典型框架 〔[3](#ref-song2021sde)〕 选定从数据到噪声的 Itô SDE

```math
\mathop{}\!\mathrm{d} X_t=f_t(X_t)\,\mathop{}\!\mathrm{d} t+g(t)\,\mathop{}\!\mathrm{d} W_t,\qquad X_0\sim\mu_{\mathrm{data}} .
```

给定前向漂移和扩散强度，反向动力学还需要每个时刻的边缘 Score
$`s_t(x)=\nabla_x\log p_t(x)`$。训练把未知边缘 Score 换成可计算的条件核目标，
拟合一个同时接受 $`x,t`$ 的网络 $`s_\theta(x,t)`$。
采样时把这个估计代入反向 SDE 或概率流 ODE。
前向 SDE 决定加噪边缘，网络提供沿所选路径去噪所需的信息。

<a name="sec-1-3-2"></a>

### 1.3.2 VE、VP、sub-VP 等正向 SDE

本节中 $`k_t(\cdot\mid x_0)=\mathcal N(\alpha_t x_0,\sigma_t^2I_d)`$。
定义 $`B_t=\int_0^t\beta(r)\,\mathop{}\!\mathrm{d} r`$，其中 $`\beta\ge0`$。
三类常见设计为

- VE 取 $`f_t=0`$，并用递增尺度 $`\varsigma(t)`$ 定义 $`g(t)^2=(\varsigma(t)^2)'`$。 从干净初值直接积分得 $`\alpha_t=1`$、 $`\sigma_t^2=\varsigma(t)^2-\varsigma(0)^2`$。 若把一个初始平滑量吸收进模型，训练所用边缘标准差也要与这个定义对齐。 噪声尺度足够大时，常以具有相应大方差的高斯近似终点。
- VP 取 $`f_t(x)=-\tfrac12\beta(t)x`$，$`g(t)^2=\beta(t)`$， 此时 $`\alpha_t=e^{-B_t/2}`$，$`\sigma_t^2=1-e^{-B_t}`$。 若 $`X_0\sim\mathcal N(0,I_d)`$，所有时刻保持标准高斯。 对一般数据，终点在 $`B_1`$ 足够大时接近标准高斯。
- sub-VP 保留 VP 漂移，扩散强度改成 $`g(t)^2=\beta(t)(1-e^{-2B_t})`$。它保持相同的均值系数， 条件标准差为 $`\sigma_t=1-e^{-B_t}`$。 用积分因子得到条件方差 $`e^{-B_t}\int_0^t e^{B_r}\beta(r)(1-e^{-2B_r})\,\mathop{}\!\mathrm{d} r =(1-e^{-B_t})^2`$。

在固定 $`\beta`$ 的 VP 与 sub-VP 比较中，sub-VP 的条件方差更小。
训练前应核对网络输入用的是 $`t`$、噪声尺度 $`\sigma_t`$ 还是其他单调变换，
以及边缘系数约定的起始扰动是否为零。

<a name="sec-1-3-3"></a>

### 1.3.3 Score 网络的训练目标

抽取 $`x_0,t,\varepsilon`$，合成 $`x=\alpha_tx_0+\sigma_t\varepsilon`$。
对 $`\sigma_t>0`$，条件核的 Score 为 $`-\varepsilon/\sigma_t`$，
典型的去噪 Score Matching 损失是

<a name="eq-overview-score-loss"></a>

```math
\mathcal L_{\mathrm{DSM}}(\theta)
  =\mathbb E_{t,x_0,\varepsilon}
   \left[w(t)\lVert s_\theta(x,t)+\varepsilon/\sigma_t \rVert^2\right] .
\tag{1.8}
```

当 $`w(t)>0`$ 且回归目标在该权重下平方可积时，逐时刻最优回归解为 $`s_t`$，
这是 [(0.2)](chapter-0.md#eq-foundations-score-identity) 与
[(0.3)](chapter-0.md#eq-foundations-regression) 的直接应用。
若参数化为 $`s_\theta=-\varepsilon_\theta/\sigma_t`$ 并取 $`w(t)=\sigma_t^2`$，
则目标变成噪声预测 MSE。改变 $`w`$ 或时间采样分布会重新分配
训练预算，有限容量模型下可影响各噪声尺度的精度。
实践中通常从 $`\sigma_t>0`$ 的时间范围取样，避免零噪声处除零。

<a name="sec-1-3-4"></a>

### 1.3.4 Reverse-time SDE 生成

在生成时钟 $`\tau`$ 下令 $`t=1-\tau`$。
将 $`s_t`$ 替换为 $`s_\theta(\cdot,t)`$ 后，
[(0.19)](chapter-0.md#eq-foundations-reverse-sde) 的标量扩散版本为

<a name="eq-overview-score-reverse"></a>

```math
\mathop{}\!\mathrm{d} Y_\tau
    =\bigl[-f_t(Y_\tau)+g(t)^2s_\theta(Y_\tau,t)\bigr]\,\mathop{}\!\mathrm{d}\tau
       +g(t)\,\mathop{}\!\mathrm{d}\overline W_\tau,\qquad Y_0\sim\mu_{\mathrm{prior}} .
\tag{1.9}
```

若正步长为 $`h`$，Euler–Maruyama 的一步是在当前状态
加上 $`h[-f_t+g(t)^2s_\theta]`$，再加 $`g(t)\sqrt h\,\xi`$，
其中 $`\xi`$ 为本步独立标准高斯。
在预测–校正采样中，predictor 沿上述反向动力学推进时间，
corrector 在一个固定噪声时刻执行若干 Langevin 更新，
例如加上 $`\delta s_\theta(y,t)+\sqrt{2\delta}\,\xi`$。
准确 Score 与合适正则性下，该时刻的连续 Langevin 动力学以 $`p_t`$ 为不变密度。
有限校正步和离散步长会引入额外近似。

<a name="sec-1-3-5"></a>

### 1.3.5 Probability Flow ODE 生成

在相同 $`t`$ 坐标下，由前向 SDE 导出的概率流速度为
$`u_t=f_t-\tfrac12g(t)^2s_t`$。相应的生成 ODE 是

<a name="eq-overview-score-ode"></a>

```math
\frac{\mathop{}\!\mathrm{d} Y_\tau}{\mathop{}\!\mathrm{d}\tau}
   =-f_t(Y_\tau)+\tfrac12 g(t)^2s_\theta(Y_\tau,t),\qquad t=1-\tau .
\tag{1.10}
```

理论上的边缘等价要求准确 Score、起点分布与前向终点匹配，
并且相应方程在所考虑的范围内唯一可解。
此时反向 SDE 与概率流 ODE 从同一边缘出发，
每个生成时刻具有同一边缘密度。网络近似与有限步求解会影响这一结论。
ODE 在给定初始噪声后是确定性的，可以用常微分方程求解器推进，
并可像 CNF 一样用散度积分计算密度。

> **边注 · 同一个 Score**
>
> 生成 SDE 的 Score 系数是 $`g^2`$，生成 ODE 的系数是 $`g^2/2`$。 两式都在网络中传入 $`t=1-\tau`$。

| 阶段 | 流程 |
| --- | --- |
| 训练 | 数据与选定的前向 SDE → 采样高斯条件核 → 回归条件 Score |
| 生成 | 从匹配尺度的先验开始 → 反向 SDE 或概率流 ODE → 读出数据端状态 |

<a name="sec-1-4"></a>

## 1.4 Continuous Normalizing Flow

<a name="sec-1-4-1"></a>

### 1.4.1 Normalizing Flow 与可逆映射

令 $`T_\theta`$ 是从先验空间到数据空间的可逆可微映射。
若 $`z\sim p_{\mathrm{prior}}`$、$`x=T_\theta(z)`$，
模型密度由变量变换得到

```math
\log p_{\theta,\mathrm{data}}(x)
   =\log p_{\mathrm{prior}}(T_\theta^{-1}(x))
      +\log\lvert \det D_xT_\theta^{-1}(x) \rvert .
```

因而同一个可逆模型既可以从 $`z`$ 生成，也可以从 $`x`$ 逆向计算似然。
离散 Normalizing Flow 用可逆映射的有限复合。CNF 以 ODE 的流映射
构造 $`T_\theta`$，体积变化改由时间积分计算 〔[4](#ref-chen2018neuralode)〕。

<a name="sec-1-4-2"></a>

### 1.4.2 连续时间流映射

令 $`F^\theta_{s,t}`$ 表示 $`\dot X_t=u_\theta(X_t,t)`$ 的流。
在本书的 $`t`$ 坐标中，$`F^\theta_{0,1}`$ 把数据态送到先验坐标。
模型用 $`p_{\mathrm{prior}}`$ 定义数据端密度，生成映射为

```math
T_\theta=F^\theta_{1,0}=(F^\theta_{0,1})^{-1}.
```

典型充分条件是速度场具有全局 Lipschitz 性、线性增长，
并对状态有适当的连续可微性。此时不同初值的轨迹由流的唯一性区分。
实际数值求解允许误差，所以有限步的正反求解复合一般只近似恒等映射。
若目标数据律是经验原子分布，可逆光滑流所定义的模型仍具有连续密度。
建模要说明训练所用的连续数据或去量化处理。

<a name="sec-1-4-3"></a>

### 1.4.3 速度场参数化

以神经网络 $`u_\theta(x,t)\in\mathbb{R}^d`$ 表示每一点的瞬时速度。
输入是当前状态和时间，输出维度等于状态维度。
速度场可由时间编码、残差块或注意力网络构造，数学上关注的是
其积分是否稳定以及空间导数能否用于散度计算。
用 $`\tau`$ 训练时也可直接表示 $`v_\theta`$，此时数据到先验的速度是
$`u_\theta(x,t)=-v_\theta(x,1-t)`$。
CNF 描述模型的动力学形式，最大似然和后面的 FM 是不同的拟合方法。

<a name="sec-1-4-4"></a>

### 1.4.4 CNF 的似然训练

对每个数据样本 $`x_0`$ 从 $`t=0`$ 积分到 $`t=1`$，同时累加散度。
令 $`\ell`$ 是积分状态，初始 $`\ell_0=0`$，使用
$`\dot\ell_t=\nabla\cdot u_\theta(X_t,t)`$。
由 [(0.15)](chapter-0.md#eq-foundations-cnf-likelihood)，负对数似然目标为

<a name="eq-overview-cnf-loss"></a>

```math
\mathcal L_{\mathrm{CNF}}(\theta)
     =-\mathbb E_{x_0}
         \bigl[\log p_{\mathrm{prior}}(F_{0,1}^\theta(x_0))+\ell_1\bigr].
\tag{1.11}
```

高维中 $`\nabla\cdot u_\theta=\operatorname{tr}(D_x u_\theta)`$ 可用
Hutchinson 恒等式估计。若随机向量 $`\xi`$ 满足
$`\mathbb E[\xi\xi^\top]=I_d`$，则
$`\mathbb E[\xi^\top (D_xu_\theta)\xi]=\operatorname{tr}(D_xu_\theta)`$。
这一估计在固定状态和时间下无偏，可通过 Jacobian 向量积计算
〔[16](#ref-grathwohl2019ffjord)〕。似然训练通常需要对 ODE 求解或相应的
灵敏度方程求导，并须控制数值容差与迹估计方差。

<a name="sec-1-4-5"></a>

### 1.4.5 CNF 的 ODE 采样

采样从 $`Y_0\sim p_{\mathrm{prior}}`$ 出发，积分
$`\dot Y_\tau=-u_\theta(Y_\tau,1-\tau)`$。
只要生成样本时不需要密度，可以只演化 $`Y_\tau`$。
若要一起计算样本对数密度，则再累计
$`-\nabla\cdot v_\theta(Y_\tau,\tau)`$。
例如 $`u_\theta(x,t)=c(t)x`$ 时，记 $`C=\int_0^1c(t)\,\mathop{}\!\mathrm{d} t`$，
数据到先验的映射为 $`x_1=e^Cx_0`$。
先验若是标准高斯，生成端协方差为 $`e^{-2C}I_d`$，
而似然中的 Jacobian 修正为 $`dC`$，可直接检验积分符号。

| 阶段 | 流程 |
| --- | --- |
| 训练 | 观测的数据样本 → 积分状态与散度 → 最小化负对数似然 |
| 生成 | 从先验取样 → 反向积分同一个流 → 得到生成样本 |

<a name="sec-1-5"></a>

## 1.5 Flow Matching 与 Conditional Flow Matching

<a name="sec-1-5-1"></a>

### 1.5.1 概率路径设计

Flow Matching（FM）先指定一族概率边缘 $`\mu_t`$ 和相应的目标速度 $`u_t`$，
再训练 CNF 的速度网络 〔[5](#ref-lipman2023fm)〕。
条件路径可以来自 Gaussian path、耦合端点的插值或某个扩散的边缘。
给定路径条件变量 $`Z`$ 及其分布 $`\pi`$，令

```math
\mu_t(\mathop{}\!\mathrm{d} x)=\int\mu_t(\mathop{}\!\mathrm{d} x\mid c)\,\pi(\mathop{}\!\mathrm{d} c).
```

在本书的坐标中，路径从 $`\mu_{\mathrm{data}}`$ 到 $`\mu_{\mathrm{prior}}`$。
例如取 $`D\sim\mu_{\mathrm{data}}`$ 与独立标准高斯 $`N`$，令
$`I_t=\alpha_tD+\sigma_tN`$。若系数端点为 $`(1,0)`$ 和 $`(0,1)`$，
就精确连接这两个边缘。FM 原论文时间沿噪声到数据，
其中保留数据端小噪声 $`\sigma_{\min}`$ 的线性条件路径换成 $`t`$ 后是

```math
I_t=(1-t)D+\bigl[\sigma_{\min}+(1-\sigma_{\min})t\bigr]N .
```

这时 $`t=0`$ 的边缘是平滑过的数据。公式中的小噪声要与所选目标端点一致。

<a name="sec-1-5-2"></a>

### 1.5.2 条件概率路径与条件速度场

对一次采到的条件 $`D=d_0`$，上面的 Gaussian path 相当于把同一个
$`N`$ 经时变仿射映射送入样本空间。沿路径求导有
$`\dot I_t=\alpha_t'D+\sigma_t'N`$。当 $`\sigma_t>0`$ 时，
给定 $`D=d_0`$ 与 $`I_t=x`$ 就可用
$`N=(x-\alpha_td_0)/\sigma_t`$ 消去噪声，得到

<a name="eq-overview-fm-conditional"></a>

```math
u_t(x\mid d_0)=\alpha_t'd_0
                   +\frac{\sigma_t'}{\sigma_t}(x-\alpha_td_0).
\tag{1.12}
```

它生成条件边缘的连续性方程。训练时直接计算
$`\alpha_t'D+\sigma_t'N`$ 可避免显式除以很小的 $`\sigma_t`$。
若条件选择为端点对 $`Z=(D,N)`$，线性插值的条件边缘是一个点质量，
其导数为 $`N-D`$。此时用第 0 章的弱形式，也可定义并边缘化速度目标。

<a name="sec-1-5-3"></a>

### 1.5.3 FM 与 CFM 的损失函数

记 $`u_\theta(x,t)`$ 为网络速度，令 $`t`$ 按约定的训练分布抽样。
理想 FM 直接拟合边缘场，Conditional Flow Matching（CFM）
则拟合可以用条件样本计算的目标 $`U_t`$。写成

<a name="eq-overview-fm-losses"></a>

```math
\begin{aligned}
  \mathcal L_{\mathrm{FM}}
      &=\mathbb E\,\lVert u_\theta(I_t,t)-u_t(I_t) \rVert^2,\\
  \mathcal L_{\mathrm{CFM}}
      &=\mathbb E\,\lVert u_\theta(I_t,t)-U_t \rVert^2,\qquad
        U_t=u_t(I_t\mid Z)\ \text{或}\ U_t=\dot I_t .
  \end{aligned}
\tag{1.13}
```

取相同的样本权重、时间分布，并满足有限二阶矩时，
两损失相差 $`\mathbb E\lVert U_t-u_t(I_t) \rVert^2`$，它与 $`\theta`$ 无关。
训练可用可采样的条件量计算梯度。即使网络已达到理想的边缘场，
单个样本的条件速度回归残差也可以非零。

<a name="sec-1-5-4"></a>

### 1.5.4 条件训练与边缘速度场之间的关系

若条件边缘有密度，其概率流密度为 $`p_t(x\mid c)u_t(x\mid c)`$。
在可对条件连续性方程作积分的条件下，

<a name="eq-overview-fm-marginal"></a>

```math
p_t(x)u_t(x)
    =\int p_t(x\mid c)u_t(x\mid c)\,\pi(\mathop{}\!\mathrm{d} c),\qquad
  u_t(x)=\mathbb E[U_t\mid I_t=x,t] .
\tag{1.14}
```

第二式用 Bayes 公式得到，给出的权重是后验 $`Z\mid I_t=x`$，
会随当前状态改变。没有条件密度时，改用测试函数。
$`\mathbb E[\nabla\varphi(I_t)\cdot U_t]`$
与 $`\mathbb E[\nabla\varphi(I_t)\cdot u_t(I_t)]`$ 相等。
这将 [(0.12)](chapter-0.md#eq-foundations-continuity-weak) 与条件期望的投影性质接起来。
因此网络输入只需当前状态和时间。未观测的条件端点用于构造训练目标。

<a name="sec-1-5-5"></a>

### 1.5.5 训练阶段与生成阶段的区别

训练时抽样 $`t,D,N`$，构造 $`(I_t,U_t)`$ 并作一次速度回归。
生成时从 $`Y_0\sim\mu_{\mathrm{prior}}`$ 出发，
积分 $`\dot Y_\tau=-u_\theta(Y_\tau,1-\tau)`$。
求解时的状态由前一步更新产生。若 $`u_\theta=u_t`$、
初始律与端点匹配且连续性方程唯一，则生成 ODE 沿指定的边缘路径返回数据端。
把 $`I_t`$ 构造为直线，可以简化条件目标。
ODE 是否沿直线运动取决于边缘场沿自身轨迹的变化。

> **边注 · 训练目标的方向**
>
> 用 $`t`$ 拟合线性路径时目标为 $`N-D`$。 用 $`\tau`$ 直接训练生成场时目标为 $`D-N`$。

| 阶段 | 流程 |
| --- | --- |
| 训练 | 数据、先验与条件路径 → 抽取状态和条件速度 → 拟合边缘速度 |
| 生成 | 从噪声先验开始 → 积分生成时钟下的 ODE → 到达数据端 |

<a name="sec-1-6"></a>

## 1.6 Rectified Flow 与 OT-CFM

<a name="sec-1-6-1"></a>

### 1.6.1 线性随机插值

给定端点耦合 $`(D,N)\sim\pi`$，本书的加噪路径写成
$`I_t=(1-t)D+tN`$。Rectified Flow 讨论从源到目标的定向传输。
用于生成时，把源设为噪声 $`N`$，目标设为数据 $`D`$，
则在论文原方向有

```math
J_\tau=(1-\tau)N+\tau D=I_{1-\tau},\qquad
  \dot J_\tau=D-N .
```

每对端点给出一条直线。许多端点对在中间时刻可能经过相同或相邻区域，
而其目标方向不同。速度网络只看到该时刻的位置，平方回归会对这些方向条件平均。

<a name="sec-1-6-2"></a>

### 1.6.2 Rectified Flow 的速度回归

直接用生成时钟训练的目标为

<a name="eq-overview-rectified-loss"></a>

```math
\mathcal L_{\mathrm{RF}}
       =\mathbb E\lVert v_\theta(J_\tau,\tau)-(D-N) \rVert^2 ,
  \qquad v_\tau(x)=\mathbb E[D-N\mid J_\tau=x,\tau] .
\tag{1.15}
```

从 $`Y_0=N`$ 解 $`\dot Y_\tau=v_\tau(Y_\tau)`$。
在路径可整直、ODE 存在并能保持插值边缘的条件下，
每个时刻的 $`Y_\tau`$ 与 $`J_\tau`$ 同边缘。
新的端点配对为 $`(Y_0,Y_1)`$。相对于训练用的随机配对，
这组配对由 ODE 流映射决定，适合用于下一轮训练
〔[6](#ref-liu2023rectified)〕。

<a name="sec-1-6-3"></a>

### 1.6.3 Reflow 与轨迹整直

一次 Reflow 先冻结现有生成场，从若干先验样本 $`N`$ 求出模型输出
$`\widehat D`$，收集 $`(N,\widehat D)`$。
随后用这组新端点对构造线性中间样本，再训练一个速度模型。
理想精确流下，原来正确的端点边缘可在下一轮保持。
实际数据侧边缘来自前一轮模型，模型偏差或求解误差可能进入新配对。

重新配对的作用可以从凸代价看出。设 $`c`$ 是可积的凸位移代价。
由 $`Y_1-Y_0=\int_0^1 v_\tau(Y_\tau)\,\mathop{}\!\mathrm{d}\tau`$，
先对时间用 Jensen，再利用 $`Y_\tau`$ 和 $`J_\tau`$ 同边缘，
最后对条件期望用 Jensen，得到

```math
\mathbb E c(Y_1-Y_0)
    \le\int_0^1\mathbb E c(v_\tau(Y_\tau))\,\mathop{}\!\mathrm{d}\tau
    \le \mathbb E c(D-N) .
```

因此在这些假设下，重配对不会增加该凸代价。
原论文进一步给出理想 Reflow 的整直性结果。固定训练轮数及有限容量下
还要检查轨迹与采样误差，凸代价下降本身不能确定所需的步数。

<a name="sec-1-6-4"></a>

### 1.6.4 独立耦合与 OT 耦合

独立耦合分别采样 $`D`$ 与 $`N`$，实现简单。
理想 OT-CFM 改为选取最小化 $`\mathbb E_\pi\lVert D-N \rVert^2`$
的端点耦合，再沿相应端点对构造条件路径。
实际中常抽取等量数据和噪声组成小批次，计算代价矩阵
$`C_{ij}=\lVert D_i-N_j \rVert^2`$，求满足经验边缘约束的批内运输方案。
再按该方案取端点对并回归速度 〔[7](#ref-tong2024otcfm)〕。
熵正则化可以产生软配对。批内最优方案对应的是经验边缘，
其总体效果取决于批量与求得的耦合。FM 原论文的条件最优传输路径
只针对固定数据点下的两个条件分布，OT-CFM 的端点运输则优化整对边缘的配对。

<a name="sec-1-6-5"></a>

### 1.6.5 路径几何与生成步数的关系

设 $`v_\tau`$ 可微，沿生成轨迹的速度变化为

```math
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d}\tau}v_\tau(Y_\tau)
     =\partial_\tau v_\tau(Y_\tau)
          +D_xv_\tau(Y_\tau)\,v_\tau(Y_\tau).
```

若右侧在一步区间内范数有界且不超过 $`M`$，从准确状态做
长度 $`h`$ 的 Euler 步，局部误差至多为 $`Mh^2/2`$。
由此可见，速度沿轨迹越接近恒定，越有可能用较少步数积分。
只看端点之间的欧氏距离，无法判断这条 ODE 的积分误差。

**例 · 训练直线与一步生成的差异**

令 $`D,N`$ 为相互独立的标准高斯向量。写
$`S(\tau)=(1-\tau)^2+\tau^2`$。
由高斯条件均值公式，

```math
v_\tau(x)=\frac{2\tau-1}{S(\tau)}x,\qquad
  Y_\tau=\sqrt{S(\tau)}\,N .
```

精确积分满足 $`Y_1=N`$，因为两个端点边缘本来相同。
但一步 Euler 从 $`\tau=0`$ 出发给 $`N+v_0(N)=0`$，
会把样本全部映到原点。用精确流产生的端点对 $`(N,N)`$ 再做
Reflow，则差向量为零，可得到恒零速度场。

| 阶段 | 流程 |
| --- | --- |
| 训练 | 独立、OT 或 Reflow 配对 → 在端点连线上取样 → 回归定向位移 |
| 生成 | 先验样本 → 积分整直后的速度场 → 生成样本 |

<a name="sec-1-7"></a>

## 1.7 Schrödinger Bridge

<a name="sec-1-7-1"></a>

### 1.7.1 给定初始分布与终端分布的桥接问题

Schrödinger Bridge（SB）同时指定 $`t=0`$ 的数据律 $`\mu_0`$ 和
$`t=1`$ 的先验律 $`\mu_1`$。所求对象是连接两端的随机路径律。
另外选取一个参考扩散 $`R`$，例如 Brownian 扰动或均值回归扩散。
参考过程只提供基准动力学，自己的端点 $`R_0,R_1`$ 可以与目标不同。
如果 $`R`$ 已满足两个边缘约束，最优解就是 $`R`$。
当 $`R`$ 的终止边缘与给定先验不合时，桥通过调整动力学满足目标终点，
无需依赖参考过程自身在有限时长内达到该先验。

<a name="sec-1-7-2"></a>

### 1.7.2 参考随机过程与路径空间 KL 最小化

第 0 章的路径空间问题为
$`\inf_{P:\,P_0=\mu_0,\,P_1=\mu_1}D_{\mathrm{KL}}(P\|R)`$。
根据 [(0.29)](chapter-0.md#eq-foundations-sb-chain)，
在给定端点耦合 $`\pi`$ 后，可以沿用 $`R`$ 的条件桥。
因此最优解的端点耦合由下列静态问题确定

<a name="eq-overview-sb-static"></a>

```math
\pi^\star\in\operatorname*{arg\,min}_{\pi\in\Pi(\mu_0,\mu_1)}
      D_{\mathrm{KL}}(\pi\|R_{01}) .
\tag{1.16}
```

例如参考为 $`X_t=X_0+\sqrt{\epsilon}\,W_t`$，
则端点核的密度含有 $`\exp[-\lVert x_1-x_0 \rVert^2/(2\epsilon)]`$，
静态问题因而与二次代价的熵正则化传输有关。
在相同扩散因子、受控漂移且满足 Girsanov 条件时，
路径 KL 可写为初始相对熵与二次控制能量之和，
见 [(0.30)](chapter-0.md#eq-foundations-control-energy)。
有限 KL 还要求目标端点与参考端点相容。
例如参考端点有密度时，其绝对连续的改权无法赋予单点正质量。

<a name="sec-1-7-3"></a>

### 1.7.3 Schrödinger Potentials

在 Markov 参考律、端点相容、且 Schrödinger 系统有正势函数解的情形，
最优桥可由端点加权表示 〔[13](#ref-leonard2014schrodinger)〕

<a name="eq-overview-sb-potentials"></a>

```math
\begin{gathered}
  \frac{\mathop{}\!\mathrm{d} P^\star}{\mathop{}\!\mathrm{d} R}=\ell(X_0)h(X_1),\\
  \widehat\psi_t(x)=\mathbb E_R[\ell(X_0)\mid X_t=x],\qquad
  \psi_t(x)=\mathbb E_R[h(X_1)\mid X_t=x].
  \end{gathered}
\tag{1.17}
```

给定 $`X_t`$ 后，参考过程过去与未来条件独立，
故若 $`R_t`$ 有密度 $`r_t`$，则桥的边缘满足
$`p_t^\star=r_t\widehat\psi_t\psi_t`$。
将 $`t=0,1`$ 代入，可写出耦合的两个端点方程
$`\ell\,\psi_0\,r_0=p_{\mathrm{data}}`$、
$`h\,\widehat\psi_1\,r_1=p_{\mathrm{prior}}`$。
因而一个端点的权重会通过参考转移核改变另一个端点的边缘。
迭代拟合要轮流修正这两项。

<a name="sec-1-7-4"></a>

### 1.7.4 正向与反向动力学

设参考正向漂移为 $`f^R_t`$，扩散矩阵为 $`a_t=G_tG_t^\top`$。
若 $`\psi_t`$ 可微且严格为正，向前转移核由 Doob $`h`$-变换改为
$`P^R_{s,t}(x,\mathop{}\!\mathrm{d} y)\psi_t(y)/\psi_s(x)`$。
短时展开给出漂移的修正 $`a_t\nabla\log\psi_t`$。
对其应用 [(0.19)](chapter-0.md#eq-foundations-reverse-sde)，再将
$`\nabla\log p_t^\star= \nabla\log r_t+\nabla\log\widehat\psi_t+\nabla\log\psi_t`$ 代入，
便得到

<a name="eq-overview-sb-drifts"></a>

```math
\begin{aligned}
  f_t^\star &= f_t^R+a_t\nabla\log\psi_t,\\
  b_\tau^\star
     &= b_\tau^R+a_t\nabla\log\widehat\psi_t,
         \qquad t=1-\tau,\\
  b_\tau^R &= -f_t^R+\operatorname{div}a_t+a_t\nabla\log r_t .
  \end{aligned}
\tag{1.18}
```

正向与反向保持相应时刻的扩散协方差。
生成时从 $`Y_0\sim\mu_1`$ 使用反向漂移 $`b_\tau^\star`$。
式中 $`b_\tau^R`$ 是参考过程的反向漂移，已经包含参考边缘的 Score。
因此即使选无漂移的参考 Brownian 扩散，反向参考漂移也可能非零。

<a name="sec-1-7-5"></a>

### 1.7.5 Diffusion Schrödinger Bridge 的训练与生成流程

原始 DSB 〔[14](#ref-debortoli2021dsb)〕 用神经网络近似迭代比例拟合（IPF）。
设 $`P^0=R`$ 且 $`R_0=\mu_0`$。在离散路径网格上，
一次理想 IPF 的两半步是

<a name="eq-overview-dsb-ipf"></a>

```math
\begin{aligned}
  Q^n(\mathop{}\!\mathrm{d} x_{0:K})&=\mu_1(\mathop{}\!\mathrm{d} x_K)\,P^n(\mathop{}\!\mathrm{d} x_{0:K-1}\mid x_K),\\
  P^{n+1}(\mathop{}\!\mathrm{d} x_{0:K})&=\mu_0(\mathop{}\!\mathrm{d} x_0)\,Q^n(\mathop{}\!\mathrm{d} x_{1:K}\mid x_0).
  \end{aligned}
\tag{1.19}
```

第一步保留反向条件律并重设噪声端边缘，第二步保留正向条件律并重设数据端边缘。
KL 链式法则解释了为什么这种条件保持是单边 KL 投影。
两边约束在理想迭代收敛时同时满足。

实际算法在相邻时刻上近似正、反向核。
以论文中方差为 $`2h_k I_d`$ 的高斯步为例，令 $`F^n_{k-1}`$ 是
$`x_{k-1}\mapsto x_k`$ 的正向均值映射，$`B^n_k`$ 是
$`x_k\mapsto x_{k-1}`$ 的反向均值映射。对 $`P^n`$ 轨迹，
给 $`B^n_k(x_k)`$ 的均方回归目标是

```math
x_k+F^n_{k-1}(x_{k-1})-F^n_{k-1}(x_k).
```

对从先验用 $`B^n_k`$ 模拟的 $`Q^n`$ 轨迹，
给 $`F^{n+1}_{k-1}(x_{k-1})`$ 的回归目标则是

```math
x_{k-1}+B^n_k(x_k)-B^n_k(x_{k-1}).
```

以第一式为例，设 $`F^n_{k-1}(y)=y+h_k f^n_{k-1}(y)`$。
写 $`s^n_k=\nabla\log p_k^n`$ 为正向边缘的 Score。对正向高斯核使用
[(0.2)](chapter-0.md#eq-foundations-score-identity)，得到
$`\mathbb E[F^n_{k-1}(X_{k-1})\mid X_k=y]=y+2h_k s^n_k(y)`$。
因此回归目标的条件均值为
$`y+h_k[-f^n_{k-1}(y)+2s^n_k(y)]`$，与反向 Euler 漂移的符号一致。
这些拟合用到两种方向上模拟的相邻样本对。
训练结束后保留最后一轮的反向网络 $`B_k`$，从 $`x_K\sim\mu_1`$ 出发，
重复 $`x_{k-1}=B_k(x_k)+\sqrt{2h_k}\,z_k`$。
有限网格、有限迭代和均值回归误差会影响端点匹配。
这与固定一个前向 SDE 后只学习其反转的 Score SDE 训练流程不同。

| 阶段 | 流程 |
| --- | --- |
| 训练 | 数据、先验与参考扩散 → 交替模拟正反向轨迹 → 学习双向均值或漂移 |
| 生成 | 从先验开始 → 用最终反向核推进 → 得到数据端样本 |

<a name="sec-1-8"></a>

## 1.8 Stochastic Interpolants

<a name="sec-1-8-1"></a>

### 1.8.1 两个端点随机变量的联合耦合

随机插值直接规定如何构造中间随机变量 〔[8](#ref-albergo2025si)〕。
先取 $`(D,N)\sim\pi`$，其中两个边缘分别为数据与先验。
可取独立配对、由最优传输构造的配对，或应用中已有的对应关系。
再独立抽取 $`\zeta\sim\mathcal N(0,I_d)`$，作为路径内部的扰动。
即使 $`N`$ 也是标准高斯，$`N`$ 仍是端点，而 $`\zeta`$ 用于调节中间边缘。
在给定端点时，$`\zeta`$ 的存在使内部时刻仍可以具有非退化条件密度。

<a name="sec-1-8-2"></a>

### 1.8.2 随机插值路径的定义

令 $`\mathcal I(t,D,N)`$ 是对时间可微的插值函数，
满足 $`\mathcal I(0,D,N)=D,\ \mathcal I(1,D,N)=N`$。
取 $`\gamma(0)=\gamma(1)=0`$，定义

<a name="eq-overview-si-path"></a>

```math
I_t=\mathcal I(t,D,N)+\gamma(t)\zeta,\qquad
  \dot I_t=\partial_t\mathcal I(t,D,N)+\gamma'(t)\zeta .
\tag{1.20}
```

常见选择为 $`\mathcal I(t,D,N)=\alpha_tD+\sigma_tN`$。
当 $`\gamma(t)>0`$ 时，给定 $`(D,N)`$ 的 $`I_t`$ 服从
$`\mathcal N(\mathcal I(t,D,N),\gamma(t)^2 I_d)`$。
逐次训练采样只需求值 [(1.20)](#eq-overview-si-path)。
若 $`\zeta`$ 在一条插值轨迹上固定，轨迹的时间导数按普通链式法则计算。
由随机初值构造的这条曲线与用 Brownian 增量模拟的扩散具有不同的路径结构。

<a name="sec-1-8-3"></a>

### 1.8.3 速度场与 Score 的学习

沿插值曲线求导，并对当前状态条件化，得到输运速度
$`u_t(x)=\mathbb E[\dot I_t\mid I_t=x,t]`$。
在 $`\gamma(t)>0`$ 的时刻，对高斯条件核求 Score 并边缘化，得到

<a name="eq-overview-si-targets"></a>

```math
u_t(x)=\mathbb E[\dot I_t\mid I_t=x,t],\qquad
  s_t(x)=-\frac{1}{\gamma(t)}\,\mathbb E[\zeta\mid I_t=x,t] .
\tag{1.21}
```

因此可用 $`\lVert u_\theta(I_t,t)-\dot I_t \rVert^2`$ 拟合速度，
用 $`\lVert z_\phi(I_t,t)-\zeta \rVert^2`$ 拟合辅助噪声的条件均值，
在采样时换算 $`s_\phi=-z_\phi/\gamma(t)`$。
直接回归 $`-\zeta/\gamma(t)`$ 也可，但靠近端点训练权重的设计不同，
并应避开公式中 $`\gamma(t)=0`$ 的时刻。
速度的条件目标包含 $`\gamma'(t)\zeta`$。若忽略它，回归的是插值确定部分
的条件导数，与完整输运速度之间相差 $`-\gamma'(t)\gamma(t)s_t`$。

<a name="sec-1-8-4"></a>

### 1.8.4 对应的 ODE 与 SDE 生成

记 $`\rho_\tau=p_{1-\tau}`$、$`v_\tau=-u_{1-\tau}`$，
并取非负时间系数 $`\kappa(\tau)`$。
基于同一插值边缘，可以选择以下生成 SDE

<a name="eq-overview-si-sampling"></a>

```math
\mathop{}\!\mathrm{d} Y_\tau=\bigl[v_\tau(Y_\tau)+\kappa(\tau)
               s_{1-\tau}(Y_\tau)\bigr]\mathop{}\!\mathrm{d}\tau
                 +\sqrt{2\kappa(\tau)}\,\mathop{}\!\mathrm{d}\overline W_\tau .
\tag{1.22}
```

理由是 Fokker–Planck 方程中的新增漂移给出
$`-\kappa\nabla\cdot(\rho_\tau\nabla\log\rho_\tau) =-\kappa\Delta\rho_\tau`$，
恰抵消随机扩散造成的 $`+\kappa\Delta\rho_\tau`$。
因此在相应方程唯一可解、初始律为 $`\mu_1`$ 且使用准确向量场时，
其边缘仍是 $`\rho_\tau`$。$`\kappa=0`$ 就给出确定性 ODE。
选择 $`\kappa>0`$ 时，训练引入的插值扰动强度 $`\gamma`$
与采样时的扩散强度 $`\sqrt{2\kappa}`$ 分别起作用。

> **边注 · 插值噪声与采样噪声**
>
> $`\gamma(t)\zeta`$ 改变训练时刻的边缘。 $`\kappa(\tau)`$ 改变采样动力学，配合准确 Score 可保持该边缘。

<a name="sec-1-8-5"></a>

### 1.8.5 与 Flow Matching 和 Diffusion Models 的关系

当目标只需生成 ODE 时，拟合 $`\mathbb E[\dot I_t\mid I_t,t]`$
的问题正是随机路径上的速度回归，和 CFM 使用同一个条件期望机制。
最简单的线性插值可省去额外的 $`\zeta`$，由独立高斯端点 $`N`$ 直接提供噪声。
在这类单侧路径里，只要 $`\sigma_t>0`$，就可以用
$`-\mathbb E[N\mid I_t=x,t]/\sigma_t`$ 计算 Score。

若 $`\alpha_t,\sigma_t`$ 选用某个扩散的条件高斯系数，
用固定端点噪声构造的 $`I_t=\alpha_tD+\sigma_tN`$
与该扩散共享同一族条件边缘。模型可以从这些边缘训练速度与 Score，
再根据所选动力学生成。是否等同于给定扩散的反向路径律，
还要核对扩散系数及漂移。特别地，选取
$`\kappa(\tau)=g(1-\tau)^2/2`$ 且令 $`u_t`$ 为该扩散对应的概率流速度，
[(1.22)](#eq-overview-si-sampling) 就化为
[(1.9)](#eq-overview-score-reverse) 的精确 Score 版本。

| 阶段 | 流程 |
| --- | --- |
| 训练 | 端点耦合与内部扰动 → 采样位置、导数及辅助噪声 → 回归速度和 Score |
| 生成 | 从噪声端点开始 → 选择 ODE 或可调扩散 SDE → 返回指定数据边缘 |

#### 八条主线对照

| 模型 | 路径如何确定 | 网络学习对象 | 生成时的运算 |
| --- | --- | --- | --- |
| DDPM | 固定前向高斯链 | 噪声或反向均值 | 相邻反向随机步 |
| DDIM | 同一条件边缘，改变联合过程 | 通常复用噪声网络 | 可跳步的隐式更新 |
| Score SDE | 选定前向扩散 | 边缘 Score | 反向 SDE 或概率流 ODE |
| CNF | 由可逆 ODE 场产生 | 用似然拟合速度 | 反向积分可逆流 |
| FM / CFM | 预设条件路径并混合 | 边缘速度 | 积分 ODE |
| Rectified Flow | 线性插值与端点耦合 | 条件平均位移 | 积分生成速度 |
| Schrödinger Bridge | 端点约束与参考路径 KL | 双向漂移或条件均值 | 反向受控随机步 |
| Stochastic Interpolants | 端点与随时间的内部扰动 | 速度及 Score | 可选 ODE 或 SDE |

后续第 2 章将展开这些回归、似然与变分目标的来历。第 3 章会分别推导
训练完成后的采样更新、ODE 与 SDE 的求解方法，以及有限步误差的来源。

<a name="references"></a>

## 参考文献

<a name="ref-ho2020ddpm"></a>
**[1]** Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).

<a name="ref-song2021ddim"></a>
**[2]** Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. ICLR, 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502).

<a name="ref-song2021sde"></a>
**[3]** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. ICLR, 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456).

<a name="ref-chen2018neuralode"></a>
**[4]** Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural Ordinary Differential Equations. NeurIPS, 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366v1).

<a name="ref-lipman2023fm"></a>
**[5]** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching for Generative Modeling. ICLR, 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).

<a name="ref-liu2023rectified"></a>
**[6]** Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. ICLR, 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003).

<a name="ref-tong2024otcfm"></a>
**[7]** Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. Transactions on Machine Learning Research, 2024. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482).

<a name="ref-albergo2025si"></a>
**[8]** Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic Interpolants: A Unifying Framework for Flows and Diffusions. Journal of Machine Learning Research, 26(209), 2025. [论文](https://jmlr.org/papers/v26/23-1605.html).

<a name="ref-leonard2014schrodinger"></a>
**[13]** Christian Léonard. A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport. Discrete and Continuous Dynamical Systems, 34(4), 1533–1574, 2014. [arXiv:1308.0215](https://arxiv.org/abs/1308.0215).

<a name="ref-debortoli2021dsb"></a>
**[14]** Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling. NeurIPS, 2021. [论文](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html).

<a name="ref-grathwohl2019ffjord"></a>
**[16]** Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. ICLR, 2019. [arXiv:1810.01367](https://arxiv.org/abs/1810.01367).
