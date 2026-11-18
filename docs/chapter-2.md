# 第 2 章　训练阶段的理论与数学推导
[返回项目目录](../README.md#contents) · [上一章：训练与生成框架](chapter-1.md) · [PDF 版](../output/pdf/model-study.pdf) · [LaTeX 源码](../chapters/02-training.tex)

[2.1 DDPM 的训练理论](#sec-2-1) · [2.2 DDIM 的训练理论](#sec-2-2) · [2.3 Score-based Models 的训练理论](#sec-2-3) · [2.4 Flow Matching 的训练理论](#sec-2-4) · [2.5 Rectified Flow 与 OT-CFM 的训练理论](#sec-2-5) · [2.6 Schrödinger Bridge 的训练理论](#sec-2-6) · [2.7 Stochastic Interpolants 的训练理论](#sec-2-7) · [2.8 训练目标之间的等价性与差异](#sec-2-8)

第 [1](chapter-1.md) 章给出了各模型的训练和生成流程。本章从联合律、
可采样的监督目标与约束优化出发，推导网络实际最小化的损失。
离散扩散沿用步号 $`k=1,\ldots,K`$ 与系数 $`r_k,\bar r_k,A_k,S_k`$。
连续时间 $`t=0`$ 为数据端，$`t=1`$ 为噪声端。记 $`D\sim\mu_0=p_{\mathrm{data}}`$、
$`N\sim\mu_1=p_{\mathrm{prior}}`$，其联合律随方法指定。
生成时间为 $`\tau=1-t`$，对应的速度是 $`v_\tau=-u_{1-\tau}`$。
除非另行说明，抽取的训练时间与端点、辅助噪声独立，所有平方回归均假定目标平方可积。

**学习目标**

- 推出 DDPM 的闭式边缘、条件后验与完整变分损失，比较 DDIM 的路径族。
- 从条件期望证明 DSM 与 CFM 的训练恒等式，并追踪时间权重。
- 解释端点耦合如何改变线性插值、Reflow 和 OT-CFM 的监督信号。
- 从路径 KL 推出 Schrödinger Bridge 的控制代价与 IPF，并构造随机插值的速度及 Score 目标。

<a name="sec-2-1"></a>

## 2.1 DDPM 的训练理论

<a name="sec-2-1-1"></a>

### 2.1.1 前向马尔可夫链的联合概率分解

设 $`X_0\sim\mu_0`$，给定噪声表 $`0<\beta_k<1`$，令
$`r_k=1-\beta_k`$、$`\bar r_k=\prod_{j=1}^kr_j`$、$`\bar r_0=1`$，
以及 $`A_k=\sqrt{\bar r_k}`$、$`S_k=\sqrt{1-\bar r_k}`$。
取相互独立且与 $`X_0`$ 独立的标准高斯 $`\xi_k`$，定义
$`X_k=\sqrt{r_k}X_{k-1}+\sqrt{\beta_k}\xi_k`$。
条件于 $`X_{k-1}`$ 后，该转移只使用当前状态与新的噪声，故

```math
q(\mathrm{d} x_{0:K})=\mu_0(\mathrm{d} x_0)\prod_{k=1}^K q(\mathrm{d} x_k\mid x_{k-1}),
\qquad q(x_k\mid x_{k-1})=\mathcal N(\sqrt{r_k}x_{k-1},\beta_k I_d).
```

以 $`\mu_0(\mathrm{d} x_0)`$ 书写联合律允许数据是离散的或集中在低维集合上。
中间时刻的条件高斯核始终有密度。本节采用 Ho 等人 〔[1](#ref-ho2020ddpm)〕 的前向链和变分构造。

<a name="sec-2-1-2"></a>

### 2.1.2 $`q(x_k\mid x_0)`$ 的闭式推导 ★

给定 $`X_0=x_0`$，令当前步的条件均值和协方差为 $`m_k`$ 与 $`c_kI_d`$。
由独立高斯的线性组合，$`m_k=\sqrt{r_k}m_{k-1}`$ 且
$`c_k=r_k c_{k-1}+\beta_k`$。从 $`m_0=x_0,c_0=0`$ 出发，
均值系数相乘给出 $`m_k=A_k x_0`$，方差递推可改写为
$`1-c_k=r_k(1-c_{k-1})`$，所以 $`c_k=S_k^2`$。由此

<a name="eq-training-ddpm-marginal"></a>

**公式（2.1）**

```math
q(x_k\mid x_0)=\mathcal N(A_kx_0,S_k^2I_d),\qquad
X_k\mid X_0=x_0\ \overset{\mathrm d}{=}\ A_kx_0+S_k\varepsilon,
\quad \varepsilon\sim\mathcal N(0,I_d).
```

将所有单步噪声展开也得到
$`X_k=A_k X_0+\sum_{j=1}^k\sqrt{\beta_j\prod_{\ell=j+1}^k r_\ell}\,\xi_j`$。
加权和的方差为 $`S_k^2I_d`$，除以 $`S_k`$ 后就是该步的标准化总噪声。
训练时采一个步号和一个高斯即可取得正确的单时刻样本。

<a name="sec-2-1-3"></a>

### 2.1.3 $`q(x_{k-1}\mid x_k,x_0)`$ 的后验推导 ★

对 $`k\ge 2`$，给定 $`(x_k,x_0)`$ 后，令 $`y=x_{k-1}`$。
Bayes 公式说明后验正比于 $`q(x_k\mid y)q(y\mid x_0)`$。
将其负对数乘以 $`2`$，含 $`y`$ 的项为

```math
\frac{\lVert x_k-\sqrt{r_k}y\rVert^2}{\beta_k}
+\frac{\lVert y-A_{k-1} x_0\rVert^2}{S_{k-1}^2}.
```

配方时二次项系数为 $`r_k/\beta_k+1/S_{k-1}^2`$，线性项为

```math
-2y^\top\left(\frac{\sqrt{r_k}x_k}{\beta_k}
                    +\frac{A_{k-1}x_0}{S_{k-1}^2}\right).
```

利用 $`r_k S_{k-1}^2+\beta_k=S_k^2`$，得到

<a name="eq-training-ddpm-posterior"></a>

**公式（2.2）**

```math
\begin{aligned}
q(y\mid x_k,x_0)&=\mathcal N(\widetilde m_k,\widetilde\beta_k I_d),\\
\widetilde\beta_k&=\frac{\beta_k S_{k-1}^2}{S_k^2},\\
\widetilde m_k&=\frac{A_{k-1}\beta_k}{S_k^2}x_0
    +\frac{\sqrt{r_k} S_{k-1}^2}{S_k^2}x_k .
\end{aligned}
```

将 $`x_0=(x_k-S_k\varepsilon)/A_k`$ 代回均值，再用 $`A_k=\sqrt{r_k}A_{k-1}`$，可得

<a name="eq-training-ddpm-eps-mean"></a>

**公式（2.3）**

```math
\widetilde m_k=\frac{1}{\sqrt{r_k}}
              \left(x_k-\frac{\beta_k}{S_k}\varepsilon\right).
```

当 $`k=1`$ 时，给定 $`x_0`$ 的后验是点质量 $`\delta_{x_0}`$，
变分目标对此步使用数据似然。实际反转核 $`q(\mathrm{d} y\mid x_k)`$ 是
将 [(2.2)](#eq-training-ddpm-posterior) 对 $`q(\mathrm{d} x_0\mid x_k)`$ 混合得到的，
通常超出单个各向同性高斯的表达能力。由全协方差公式，其协方差包含
$`\widetilde\beta_k I_d`$ 和
$`\operatorname{Cov}(\widetilde m_k(X_0,x_k)\mid X_k=x_k)`$ 两项。

**例 2.1 · 两步条件后验**

取 $`\beta_1=1/5,\ \beta_2=1/2`$。于是
$`S_1^2=1/5,\ S_2^2=3/5`$，第二步后验方差为 $`1/6`$，
均值为 $`(5/6)\sqrt{4/5}\,x_0+(\sqrt{1/2}/3)x_2`$。
同一组参数用于公式 [(2.3)](#eq-training-ddpm-eps-mean) 会得到相同结果。

<a name="sec-2-1-4"></a>

### 2.1.4 ELBO 的构造、KL 项分解与化简 ★

生成联合律定义为 $`p_\theta(x_{0:K})=p_{\mathrm{prior}}(x_K) \prod_{k=1}^Kp_\theta(x_{k-1}\mid x_k)`$。
对固定的 $`x_0`$，以已知 $`q(x_{1:K}\mid x_0)`$ 作为变分分布。
把 $`\log p_\theta(x_0)`$ 写成对该分布的密度比期望的对数并用 Jensen 不等式，
得到

```math
\log p_\theta(x_0)\ge
\mathbb E_{q(x_{1:K}\mid x_0)}
\left[\log p_\theta(x_{0:K})-\log q(x_{1:K}\mid x_0)\right].
```

这里 $`p_\theta(x_0\mid x_1)`$ 按数据类型表示概率质量或密度。
前向链给定 $`x_0`$ 后也有反向分解
$`q(x_{1:K}\mid x_0)=q(x_K\mid x_0) \prod_{k=2}^K q(x_{k-1}\mid x_k,x_0)`$。
将两个分解代入负 ELBO，并对训练数据取期望，便得到

<a name="eq-training-ddpm-elbo"></a>

**公式（2.4）**

```math
\begin{aligned}
\mathcal L_{\mathrm{VLB}}
=\mathbb E_q\bigg[
&\underbrace{D_{\mathrm{KL}}\big(q(X_K\mid X_0)\|
                                    p_{\mathrm{prior}}\big)}_{L_K}\\
&+\sum_{k=2}^K\underbrace{D_{\mathrm{KL}}\big(q(X_{k-1}\mid X_k,X_0)
                                \|p_\theta(X_{k-1}\mid X_k)\big)}_{L_{k-1}}\\
&+\underbrace{-\log p_\theta(X_0\mid X_1)}_{L_0}\bigg].
\end{aligned}
```

KL 中的 $`q(X_{k-1}\mid X_k,X_0)`$ 等记号表示条件律。
先验与前向过程固定时，$`L_K`$ 与 $`\theta`$ 无关。
若先验为标准高斯，对固定 $`x_0`$ 有

```math
L_K=\tfrac12\big[A_K^2\lVert x_0\rVert^2
           +d(S_K^2-1-\log S_K^2)\big].
```

它仍决定变分界的值。$`L_0`$ 由解码器决定。
量化数据可对离散区间积分得到类别概率，连续数据可指定非退化高斯解码器。
变分界与对数似然之间的差由真实模型后验
$`p_\theta(x_{1:K}\mid x_0)`$ 与变分分布之间的 KL 给出，见
[(0.8)](chapter-0.md#eq-foundations-elbo)。

<a name="sec-2-1-5"></a>

### 2.1.5 从 ELBO 到噪声预测 MSE 损失 ★

对 $`k\ge 2`$，设生成核为 $`p_\theta(x_{k-1}\mid x_k) =\mathcal N(m_\theta(x_k,k),\nu_k I_d)`$，且方差 $`\nu_k>0`$ 固定。
由高斯 KL 式 [(0.7)](chapter-0.md#eq-foundations-gaussian-kl)，

```math
L_{k-1}=\frac{\lVert \widetilde m_k-m_\theta\rVert^2}{2\nu_k}
  +\frac d2\left(\frac{\widetilde\beta_k}{\nu_k}-1+
                     \log\frac{\nu_k}{\widetilde\beta_k}\right).
```

在 [(2.3)](#eq-training-ddpm-eps-mean) 中用 $`\varepsilon_\theta(X_k,k)`$
替换 $`\varepsilon`$，便定义了可学习均值 $`m_\theta`$。
代入 KL 后整理出

<a name="eq-training-ddpm-weight"></a>

**公式（2.5）**

```math
L_{k-1}=w_k\lVert \varepsilon-\varepsilon_\theta(X_k,k)\rVert^2+C_k,\qquad
w_k=\frac{\beta_k^2}{2\nu_k r_k S_k^2}.
```

若步号按概率 $`\pi_k>0`$ 抽样，将每项乘 $`w_k/\pi_k`$，即可无偏估计中间步的
加权*和*。若还要估计完整目标，应加入先验项和第一步解码项。
常用的 $`L_{\mathrm{simple}}`$ 对 $`k=1,\ldots,K`$ 均匀采样，
使用单位权重的噪声 MSE。若方差也由网络学习，
协方差项和均值误差前的因子都会依赖网络输出。

> **边注 · 权重与共享参数**
>
> 正权重保持无约束回归的逐时刻最优预测。一个网络同时拟合所有步号时，
> 权重影响各时刻的误差折中。

<a name="sec-2-1-6"></a>

### 2.1.6 $`\epsilon`$-prediction、$`x_0`$-prediction 与 $`v`$-prediction

由于 $`A_k^2+S_k^2=1`$，可对 $`(X_0,\varepsilon)`$ 做二维正交变换

<a name="eq-training-ddpm-vpred"></a>

**公式（2.6）**

```math
\begin{aligned}
X_k&=A_k X_0+S_k\varepsilon,&
V_k^{\mathrm{diff}}&=A_k\varepsilon-S_k X_0,\\
X_0&=A_k X_k-S_k V_k^{\mathrm{diff}},&
\varepsilon&=S_k X_k+A_k V_k^{\mathrm{diff}} .
\end{aligned}
```

$`v`$-prediction 用 $`V_k^{\mathrm{diff}}`$ 作为监督，参见
〔[15](#ref-salimans2022distill)〕。给定 $`X_k=x`$，三种预测可相互换算。
例如由噪声输出得到 $`\widehat x_0=(x-S_k\widehat\varepsilon)/A_k`$。
若这些输出由同一 $`V_k^{\mathrm{diff}}`$ 预测换算而来，
则误差满足

<a name="eq-training-ddpm-parameterization"></a>

**公式（2.7）**

```math
\lVert \widehat\varepsilon-\varepsilon\rVert^2
=\frac{A_k^2}{S_k^2}\lVert \widehat x_0-X_0\rVert^2
=A_k^2\lVert \widehat V_k^{\mathrm{diff}}-V_k^{\mathrm{diff}}\rVert^2.
```

因而将单位噪声 MSE 写成等值的 $`x_0`$ MSE 时，权重为信噪比
$`\mathrm{SNR}_k=A_k^2/S_k^2`$。零噪声或零信号系数处，含相应除法的转换不可用，
而由 $`x,V^{\mathrm{diff}}`$ 同时还原两个分量的线性公式仍成立。
若连续路径写成 $`A=\cos\vartheta(t),\,S=\sin\vartheta(t)`$，
固定 $`(X_0,\varepsilon)`$ 后有 $`\dot X_t=\vartheta'(t)V_t^{\mathrm{diff}}`$。
加噪输运速度还需对 $`X_t=x`$ 取条件期望，生成速度还需反转时间和符号。

<a name="sec-2-2"></a>

## 2.2 DDIM 的训练理论

<a name="sec-2-2-1"></a>

### 2.2.1 共享 DDPM 边缘加噪分布的条件

沿用 $`A_k,S_k`$ 和数据律。DDIM 〔[2](#ref-song2021ddim)〕 要求每一步条件边缘
仍为 [(2.1)](#eq-training-ddpm-marginal)。于是由
$`\varepsilon=(X_k-A_kX_0)/S_k`$ 定义的监督标签，连同
$`(X_0,X_k,k)`$，具有与 DDPM 相同的联合律。若步号概率和损失权重也相同，
两者的单时刻噪声回归损失对任意网络均相等。

<a name="sec-2-2-2"></a>

### 2.2.2 非马尔可夫前向联合分布的构造

先取 $`X_K\mid X_0`$ 为指定的末端高斯，再定义给定 $`x_0`$ 的反向条件核。
对 $`k\ge2`$，选择 $`0\le\delta_k^2\le S_{k-1}^2`$ 并令
$`B_k=\sqrt{S_{k-1}^2-\delta_k^2}/S_k`$，取

<a name="eq-training-ddim-kernel"></a>

**公式（2.8）**

```math
X_{k-1}=A_{k-1}x_0+B_k(X_k-A_kx_0)+\delta_k Z_k,
\qquad Z_k\sim\mathcal N(0,I_d)
```

且 $`Z_k`$ 与 $`X_k`$ 在给定 $`x_0`$ 后独立。
若 $`X_k\mid x_0\sim\mathcal N(A_kx_0,S_k^2I_d)`$，则上式均值为 $`A_{k-1}x_0`$，
方差为 $`(B_k^2 S_k^2+\delta_k^2)I_d=S_{k-1}^2I_d`$。
从 $`K`$ 倒推即验证了所有要求的边缘。反向条件律的乘积定义了整个联合分布。
取 $`\delta_k^2=\widetilde\beta_k`$ 时，
$`B_k=\sqrt{r_k}S_{k-1}^2/S_k^2`$，恢复 DDPM 的条件后验。
取 $`\delta_k=0`$ 时，相邻标准化残差相同，整个序列共用一个高斯。

<a name="sec-2-2-3"></a>

### 2.2.3 DDIM 训练目标与 DDPM 简化损失的联系

训练噪声网络后可令
$`\widehat x_0=(x_k-S_k\varepsilon_\theta(x_k,k))/A_k`$，
并在 [(2.8)](#eq-training-ddim-kernel) 的均值中用 $`\widehat x_0`$ 替换 $`x_0`$。
写 $`C_k=A_{k-1}-B_k A_k`$，则条件均值是 $`B_kx_k+C_k x_0`$，
模型均值是 $`B_kx_k+C_k\widehat x_0`$。当 $`\delta_k>0`$ 且模型核取同样方差时，
中间步高斯 KL 为

<a name="eq-training-ddim-weight"></a>

**公式（2.9）**

```math
\frac{C_k^2}{2\delta_k^2}\lVert x_0-\widehat x_0\rVert^2
=\frac{C_k^2 S_k^2}{2\delta_k^2 A_k^2}
        \lVert \varepsilon-\varepsilon_\theta(x_k,k)\rVert^2 .
```

将 DDIM 的反向条件乘积分解代入负变分界，可重得
[(2.4)](#eq-training-ddpm-elbo) 的三类项，其中 $`q`$ 改为 DDIM 的联合律，
中间步使用上式。末端先验项沿用同一条件边缘。若 $`k=1`$ 采用
方差为 $`\nu_1 I_d`$、均值为 $`\widehat x_0`$ 的连续高斯解码器，
其负对数似然等于 $`\lVert x_0-\widehat x_0\rVert^2/(2\nu_1)`$ 加常数。
从而这一设定下的全部参数依赖项可写成正权重的噪声回归。
共享的简化目标直接在固定边缘上定义。确定性情形可使用这一训练结果生成，
逐步非退化高斯 KL 的公式要求 $`\delta_k>0`$。

<a name="sec-2-2-4"></a>

### 2.2.4 不同路径与相同边缘分布的区别

在 [(2.8)](#eq-training-ddim-kernel) 下有

```math
\operatorname{Cov}(X_{k-1},X_k\mid X_0=x_0)
  = S_k\sqrt{S_{k-1}^2-\delta_k^2}\,I_d .
```

计算时协方差只来自 $`B_k(X_k-A_kx_0)`$ 一项。
改变 $`\delta_k`$ 保持了每个时刻的边缘，却改变了这项跨时刻统计量。
一般参数下，正向条件 $`q_\delta(x_k\mid x_{0:k-1})`$
还依赖 $`x_0`$。相同的单时刻训练对由此可以支持不同的联合动力学。

<a name="sec-2-2-5"></a>

### 2.2.5 采样方式改变是否要求重新训练

切换 DDPM 与 DDIM 的采样器时，只要仍在网络训练所用的加噪强度和步号上求值，
并正确换算噪声、干净数据或 $`V^{\mathrm{diff}}`$ 输出，就能复用该回归模型。
例如跳步网格保留了训练网格上的 $`k`$，网络输入仍应是原来的步号。
若更换训练数据、条件信息，或以原步号表示不同的 $`(A_k,S_k)`$，
所需回归目标会重新变化。复用网络也不保证粗网格能准确重建目标边缘，
有限步采样误差在第 3 章讨论。

<a name="sec-2-3"></a>

## 2.3 Score-based Models 的训练理论

<a name="sec-2-3-1"></a>

### 2.3.1 Fisher Divergence 与 Score Matching

固定 $`t`$ 且令 $`p=p_t`$，Score 为 $`s(x)=\nabla_x\log p(x)`$。
对可微候选场 $`s_\theta(x)`$ 定义
$`\mathcal J_{\mathrm{SM}}=\tfrac12\mathbb E_{p}\lVert s_\theta(X)-s(X)\rVert^2`$。
若 $`s_\theta=\nabla\log p_\theta`$，它就是带 $`1/2`$ 系数的 Fisher 散度。
也可令神经网络直接输出一般向量场，在总体回归达到最优时有
$`s_\theta=s`$（$`p`$-几乎处处）。基于 Score 的估计只使用未归一化密度
对 $`x`$ 的对数导数，归一化常数在求导时消失。

<a name="sec-2-3-2"></a>

### 2.3.2 Hyvärinen Score Matching 的推导 ★

设 $`p`$ 足够光滑、相关积分有限，并且空间无穷远处的边界通量消失。
对有边界的区域则需相应的边界条件。展开平方得

```math
\mathcal J_{\mathrm{SM}}
=\tfrac12\mathbb E_p\lVert s_\theta\rVert^2
 -\int s_\theta^\top\nabla p\,\mathrm{d} x
 +\tfrac12\mathbb E_p\lVert s\rVert^2 .
```

由局部乘积法则
$`\nabla\cdot(p s_\theta)=s_\theta^\top\nabla p+p\nabla\cdot s_\theta`$，
积分并去掉零边界项，得到 Hyvärinen 的目标 〔[17](#ref-hyvarinen2005sm)〕

<a name="eq-training-implicit-sm"></a>

**公式（2.10）**

```math
\mathcal J_{\mathrm{SM}}(\theta)
 =\mathbb E_p\!\left[\tfrac12\lVert s_\theta(X)\rVert^2
                     +\nabla\cdot s_\theta(X)\right]+C.
```

这里 $`C=\frac12\mathbb E_p\lVert s\rVert^2`$ 与参数无关。
散度为 $`\operatorname{tr}(\nabla_x s_\theta)`$。
对 $`p=\mathcal N(0,\sigma^2)`$、$`s_\theta(x)=-c x`$，
可计算的部分为 $`\frac12c^2\sigma^2-c`$，最优 $`c=1/\sigma^2`$。

<a name="sec-2-3-3"></a>

### 2.3.3 Denoising Score Matching 的推导 ★

给定 $`D`$ 后可从已知扰动核 $`k_t(x\mid D)`$ 采样 $`X_t`$。
令 $`p_t(x)=\int k_t(x\mid d)\mu_0(\mathrm{d} d)`$，
$`R_t=\nabla_x\log k_t(X_t\mid D)`$。
在允许交换微分与积分、且 $`p_t(x)>0`$ 的点，

<a name="eq-training-dsm-projection"></a>

**公式（2.11）**

```math
\nabla\log p_t(x)
=\frac{\int \nabla_x k_t(x\mid d)\,\mu_0(\mathrm{d} d)}{p_t(x)}
=\mathbb E[R_t\mid X_t=x] .
```

因此可用 Vincent 的去噪目标 〔[18](#ref-vincent2011dsm)〕。令
$`M_t=\mathbb E[R_t\mid X_t]`$ 并展开
$`\lVert s_\theta(X_t,t)-R_t\rVert^2=\lVert s_\theta-M_t\rVert^2+ \lVert R_t-M_t\rVert^2-2(s_\theta-M_t)^\top(R_t-M_t)`$。
条件于 $`X_t`$ 后交叉项的期望为零，得到

<a name="eq-training-dsm-loss"></a>

**公式（2.12）**

```math
\mathcal J_{\mathrm{DSM},t}
 =\mathcal J_{\mathrm{SM},t}+\tfrac12\mathbb E\lVert R_t-M_t\rVert^2 ,
```

其中两项损失均带 $`1/2`$ 系数。常数项是给定带噪样本后监督标签的剩余方差。
高斯扰动 $`X_t=\alpha_tD+\sigma_t\varepsilon`$ 下，
$`R_t=-(X_t-\alpha_tD)/\sigma_t^2=-\varepsilon/\sigma_t`$。
于是 $`s_t(x)=-\mathbb E[\varepsilon\mid X_t=x]/\sigma_t`$。

**例 2.2 · 可验算的去噪回归**

令一维 $`D\sim\mathcal N(0,a^2)`$，设 $`H_t=\alpha_t^2a^2+\sigma_t^2>0`$。
联合高斯回归给出 $`\mathbb E[\varepsilon\mid X_t=x]=\sigma_t x/H_t`$。
代入上式得 $`s_t(x)=-x/H_t`$，与直接对 $`\mathcal N(0,H_t)`$ 求导相同。
当 $`\sigma_t>0`$ 时，条件 Score 监督的剩余方差为
$`1/\sigma_t^2-1/H_t`$。

<a name="sec-2-3-4"></a>

### 2.3.4 多噪声尺度与连续时间损失

在多个噪声水平上重复这一回归，由同一个 $`s_\theta(x,t)`$ 接收时间条件。
例如 VP SDE 的 $`f_t(x)=-\beta(t)x/2,\ g(t)=\sqrt{\beta(t)}`$，
积分因子给出
$`\alpha_t=\exp[-\frac12\int_0^t\beta(r)\,\mathrm{d} r]`$ 与
$`\sigma_t^2=\int_0^t(\alpha_t/\alpha_r)^2\beta(r)\,\mathrm{d} r=1-\alpha_t^2`$。
每次更新可直接采样 $`D,t,\varepsilon`$ 并合成 $`X_t`$
〔[3](#ref-song2021sde)〕。若转移核没有显式条件 Score，
则应另找可计算的监督或采用隐式 Score Matching。
当 $`\sigma_t\to0`$ 时，高斯条件标签的二阶矩 $`d/\sigma_t^2`$ 增长，
可用内点截断 $`t\ge t_{\min}>0`$、可积的时间权重或噪声参数化处理。

<a name="sec-2-3-5"></a>

### 2.3.5 时间权重与网络优化目标

设 $`t`$ 按密度 $`m(t)`$ 抽样。对给定目标时间权重 $`\lambda(t)`$，
只要 $`m(t)>0`$ 覆盖 $`\lambda(t)>0`$ 的区域，且期望有限，就有

<a name="eq-training-score-time-weight"></a>

**公式（2.13）**

```math
\begin{aligned}
&\frac12\int \lambda(t)\mathbb E\lVert s_\theta(X_t,t)-R_t\rVert^2\,\mathrm{d} t\\
&\qquad=\mathbb E_{t\sim m,D,\varepsilon}
   \left[\frac{\lambda(t)}{2m(t)}
          \lVert s_\theta(X_t,t)+\varepsilon/\sigma_t\rVert^2\right]
\end{aligned}
```

上式右侧采用高斯扰动。令 $`s_\theta=-\varepsilon_\theta/\sigma_t`$ 后，
被采样的噪声 MSE 系数为 $`\lambda(t)/(2m(t)\sigma_t^2)`$。
单位权重的噪声回归对应 $`\lambda(t)\propto m(t)\sigma_t^2`$。
对于标量扩散强度 $`g(t)>0`$，反向漂移误差为 $`g(t)^2(s_\theta-s_t)`$。
记 $`p_{\theta,0}`$ 为模型生成的末端数据边缘。
在适用第 [2.6.3](#sec-2-6-3) 小节的路径 KL 恒等式时，
对反向路径律的 KL 取末端坐标并使用数据处理不等式，得到

```math
\begin{aligned}
D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{\theta,0})
\le {}&D_{\mathrm{KL}}(p_1\|p_{\mathrm{prior}})\\
     &+\tfrac12\int_0^1 g(t)^2
        \mathbb E_{p_t}\lVert s_\theta(X_t,t)-s_t(X_t)\rVert^2\,\mathrm{d} t .
\end{aligned}
```

右侧将真实反向路径律放在路径 KL 的第一项，因此期望沿真实边缘 $`p_t`$ 计算。
权重 $`g^2`$ 可用于似然上界的训练
〔[19](#ref-song2021likelihood)〕。
对重要性采样而言，修改 $`m`$ 并补偿 $`1/m`$ 会保持总体目标，
也会改变随机梯度方差。

<a name="sec-2-4"></a>

## 2.4 Flow Matching 的训练理论

<a name="sec-2-4-1"></a>

### 2.4.1 连续性方程约束下的速度场学习

Flow Matching 〔[5](#ref-lipman2023fm)〕 先给定一族边缘 $`p_t`$，
再拟合一个满足 $`\partial_t p_t+\nabla\cdot(p_tu_t)=0`$ 的速度场。
在光滑密度下，若 $`u_t`$ 可行且 $`\nabla\cdot(p_t w_t)=0`$，
则 $`u_t+w_t`$ 也满足同一个连续性方程。条件路径及其速度提供了选择 $`u_t`$ 的规则。
当 ODE 流与相应连续性方程在所考虑的解类中唯一可解，
从 $`p_0`$ 出发沿 $`u_t`$ 推进得到 $`p_t`$。
生成时从 $`p_1`$ 出发用 $`v_\tau=-u_{1-\tau}`$，沿反向流回到数据端。
训练的平方积分只控制时空分布几乎处处的误差，建立流的适定性还需要对场的正则性作假设。

<a name="sec-2-4-2"></a>

### 2.4.2 概率路径 $`p_t`$ 的指定

取条件标签 $`C\sim\pi`$，为每个 $`c`$ 指定条件路径 $`p_t(\mathrm{d} x\mid c)`$，
混合得到 $`p_t(\mathrm{d} x)=\int p_t(\mathrm{d} x\mid c)\pi(\mathrm{d} c)`$。
例如 $`C=D`$，使用 $`X_t=\alpha_tD+\sigma_t\varepsilon`$，
$`\varepsilon`$ 为独立标准高斯。若
$`(\alpha_0,\sigma_0)=(1,0)`$、$`(\alpha_1,\sigma_1)=(0,1)`$，
便得到精确的数据与标准高斯端点。
当 $`\sigma_t>0`$ 时，给定 $`D=d`$ 的轨迹可由当前点恢复内部高斯，
其条件速度为

<a name="eq-training-fm-cond-gauss"></a>

**公式（2.14）**

```math
u_t(x\mid d)=\dot\alpha_t d+
                 \frac{\dot\sigma_t}{\sigma_t}(x-\alpha_t d).
```

训练时可直接用 $`\dot\alpha_tD+\dot\sigma_t\varepsilon`$ 作为监督。
该写法免去显式除以 $`\sigma_t`$，但仍要求导数的时间积分有意义。
如果终点留有残余信号或额外高斯扰动，应按实际端点律解释所拟合的流。

<a name="sec-2-4-3"></a>

### 2.4.3 边缘速度场与条件速度场

设每条条件路径都有满足弱连续性方程的速度 $`u_t(\cdot\mid c)`$。
若可交换对 $`t`$ 的微分与对 $`c`$ 的积分，
对任意光滑紧支撑的测试函数 $`\varphi`$ 有

<a name="eq-training-fm-current"></a>

**公式（2.15）**

```math
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} t}\mathbb E[\varphi(X_t)]
  &=\mathbb E[\nabla\varphi(X_t)^\top u_t(X_t\mid C)]\\
  &=\mathbb E[\nabla\varphi(X_t)^\top u_t(X_t)],\\
u_t(x)&=\mathbb E[u_t(X_t\mid C)\mid X_t=x].
\end{aligned}
```

最后一式定义边缘场，前两式表明它满足弱连续性方程。
存在条件密度时，它可写为

```math
p_t(x)u_t(x)=\int p_t(x\mid c)u_t(x\mid c)\,\pi(\mathrm{d} c).
```

因此速度平均使用当前位置下的后验权重
$`p_t(x\mid c)\pi(\mathrm{d} c)/p_t(x)`$。
弱式也覆盖条件路径为点质量的情形，例如给定两个端点的确定性线段。

<a name="sec-2-4-4"></a>

### 2.4.4 Flow Matching 理想损失的推导

设可直接取得 $`p_t`$ 的抽样及上述选定的 $`u_t(x)`$，
就在目标概率路径上求解 $`L^2`$ 拟合

<a name="eq-training-fm-loss"></a>

**公式（2.16）**

```math
\mathcal L_{\mathrm{FM}}(\theta)
  =\mathbb E_{t,X_t}\lVert u_\theta(X_t,t)-u_t(X_t)\rVert^2 ,
\qquad X_t\sim p_t.
```

无约束场的最优值为零，且最优场等于 $`u_t`$（训练时空分布几乎处处）。
指定连续性方程本身只约束通量散度，上式额外指定了要逼近的场。
有限容量下，该损失衡量所选模型类对 $`u_t`$ 的近似误差。

#### 与 CNF 的似然训练

同一个可逆 ODE 也可作为 CNF。按加噪方向从 $`X_0=x_0`$ 积分模型场至 $`X_1`$，
由瞬时变量变换式 [(0.15)](chapter-0.md#eq-foundations-cnf-likelihood) 得

```math
-\log p_\theta(x_0)=-\log p_{\mathrm{prior}}(X_1)
             -\int_0^1\nabla\cdot u_\theta(X_t,t)\,\mathrm{d} t .
```

这需要沿模型轨迹评估散度。FM 固定训练路径并直接匹配速度，
其目标是 [(2.16)](#eq-training-fm-loss)。若场精确、流适定且训练路径的端点与先验匹配，
各自的生成流在数据端恢复指定分布。

<a name="sec-2-4-5"></a>

### 2.4.5 Conditional Flow Matching 损失的推导 ★

混合边缘的速度含有对条件标签的后验平均。在训练时先采
$`C,t`$，再采 $`X_t\sim p_t(\cdot\mid C)`$，便可直接评估
$`U_t=u_t(X_t\mid C)`$。据此定义

<a name="eq-training-cfm-loss"></a>

**公式（2.17）**

```math
\mathcal L_{\mathrm{CFM}}(\theta)
    =\mathbb E_{t,C,X_t}\lVert u_\theta(X_t,t)-U_t\rVert^2 .
```

由 [(2.15)](#eq-training-fm-current)，$`u_t(X_t)=\mathbb E[U_t\mid X_t,t]`$。
因此 $`U_t`$ 是条件于网络输入的无偏监督。对于高斯条件路径，
每次取一份数据 $`D`$、一个时间和一个高斯，即可同时形成
$`(X_t,U_t)=(\alpha_tD+\sigma_t\varepsilon,\dot\alpha_tD+\dot\sigma_t\varepsilon)`$。
网络只读取 $`(X_t,t)`$。如果把用来构造路径的私有标签 $`C`$ 也输入网络，
回归的对象会变为针对新输入的条件期望。

<a name="sec-2-4-6"></a>

### 2.4.6 CFM 与 FM 梯度等价的证明 ★

记 $`M=u_t(X_t)`$。展开 $`u_\theta-U_t=(u_\theta-M)+(M-U_t)`$，
对 $`(X_t,t)`$ 取条件期望，有
$`\mathbb E[(u_\theta-M)^\top(M-U_t)\mid X_t,t]=0`$。
再取总体期望得到

<a name="eq-training-cfm-identity"></a>

**公式（2.18）**

```math
\mathcal L_{\mathrm{CFM}}(\theta)
  =\mathcal L_{\mathrm{FM}}(\theta)+\mathbb E\lVert U_t-M\rVert^2 .
```

路径与抽样规则固定时，后一项与 $`\theta`$ 无关，
因此两者在同一模型类中具有相同的最优解，
并在可微处具有相同的总体梯度。
如果微分可交换入期望，单样本梯度
$`2(\partial_\theta u_\theta)^\top(u_\theta-U_t)`$ 在给定 $`(X_t,t)`$
后取平均也正好是理想梯度。条件标签的波动仍会影响梯度方差。
同样的推导允许用相同的非负时间权重乘两种损失。
若利用正在更新的网络重新定义耦合，须固定当轮的采样律，
或在求总导数时计入采样律的参数依赖。

<a name="sec-2-5"></a>

## 2.5 Rectified Flow 与 OT-CFM 的训练理论

<a name="sec-2-5-1"></a>

### 2.5.1 线性插值的条件速度目标

取 $`(D,N)\sim\pi\in\Pi(\mu_0,\mu_1)`$，定义
$`I_t=(1-t)D+tN`$ 与 $`\Delta=N-D`$。
每一对端点的条件路径为 $`\delta_{I_t}`$，条件速度恒为 $`\Delta`$。
因此以 $`(I_t,t,\Delta)`$ 作为训练三元组。
若用生成时间写，同一训练点为
$`(1-\tau)N+\tau D=I_{1-\tau}`$，监督改为 $`D-N=-\Delta`$。
Rectified Flow 与 OT-CFM 都可以采用这一线性目标。
不同的端点配对会同时改变输入 $`I_t`$ 与标签 $`\Delta`$ 的联合律。

<a name="sec-2-5-2"></a>

### 2.5.2 直线插值与真实 ODE 轨迹的区别

给定 $`I_t=x`$，多条端点线段可能贡献不同的 $`\Delta`$，
而 ODE 场必须给同一 $`(x,t)`$ 一个速度，平方损失选取这些速度的后验平均。
即使 $`I_t`$ 的速度沿每一条训练线段恒定，边缘速度沿生成解 $`Z_t`$ 的导数为
$`\partial_tu_t(Z_t)+(\nabla_xu_t)(Z_t)u_t(Z_t)`$（场足够光滑时）。
这项沿相应轨迹消失时，速度才沿该轨迹恒定。
例 [2.3](#ex-training-rf-gaussian) 给出同样端点边缘下，两种耦合产生的轨迹差异。

<a name="sec-2-5-3"></a>

### 2.5.3 Rectified Flow 的条件期望表达式 ★

把线性训练目标和输入一起平方回归，

<a name="eq-training-rf-conditional"></a>

**公式（2.19）**

```math
\begin{aligned}
u_t(x)&=\mathbb E[\Delta\mid I_t=x],\\
\inf_{u}\mathbb E_{t,\pi}\lVert u_t(I_t)-\Delta\rVert^2
  &=\mathbb E_{t,\pi}\lVert \Delta-\mathbb E[\Delta\mid I_t,t]\rVert^2 .
\end{aligned}
```

目标的 $`L^2`$ 正交分解见 [(2.18)](#eq-training-cfm-identity)。
对测试函数直接求 $`\mathrm{d}\mathbb E[\varphi(I_t)]/\mathrm{d} t`$，
可用 $`\mathbb E[\nabla\varphi(I_t)^\top\Delta]`$
重得相同的连续性方程。若 $`Z_0\sim\mu_0`$，
且以 $`u_t`$ 推进的连续性方程具有唯一解，则 $`Z_t`$ 与 $`I_t`$ 具有相同边缘。

<a name="ex-training-rf-gaussian"></a>

**例 2.3 · 独立与同向高斯耦合**

令一维 $`D,N`$ 的边缘均为标准高斯。独立时 $`I_t`$ 的方差为
$`H_t=(1-t)^2+t^2`$，$`\operatorname{Cov}(\Delta,I_t)=2t-1`$。
联合高斯回归给出 $`u_t(x)=(2t-1)x/H_t`$，ODE 解为 $`Z_t=\sqrt{H_t}\,Z_0`$。
它在中间先收缩再扩张，终点 $`Z_1=Z_0`$。
若改用 $`N=D`$ 的耦合，方差恒为 $`1`$，监督和速度均为零。
独立耦合的输入位移平方期望为 $`2`$，而它的精确流所产生的新端点耦合的位移为零。

<a name="sec-2-5-4"></a>

### 2.5.4 轨迹整直的理论与 Reflow

Liu 等人 〔[6](#ref-liu2023rectified)〕 将精确流给出的 $`(Z_0,Z_1)`$ 作为新耦合，
再用它构造线段和位移标签训练下一轮。这是 Reflow 的基本迭代。
对凸位移代价 $`c`$（假设相关期望有限），若流匹配插值边缘，
可连续使用两次 Jensen 不等式，

<a name="eq-training-rf-convex"></a>

**公式（2.20）**

```math
\begin{aligned}
\mathbb E c(Z_1-Z_0)
  &=\mathbb E c\left(\int_0^1u_t(Z_t)\,\mathrm{d} t\right)\\
  &\le \int_0^1\mathbb E c(u_t(Z_t))\,\mathrm{d} t\\
  &=\int_0^1\mathbb E c\big(\mathbb E[\Delta\mid I_t]\big)\,\mathrm{d} t
   \le \mathbb E c(\Delta).
\end{aligned}
```

对 $`c(z)=\lVert z\rVert^2`$，分别用条件期望的正交分解和时间平均的方差分解，可得

<a name="eq-training-rf-straightness"></a>

**公式（2.21）**

```math
\begin{aligned}
 &\mathbb E\lVert \Delta\rVert^2-\mathbb E\lVert Z_1-Z_0\rVert^2\\
 &\quad=\int_0^1\mathbb E\lVert \Delta-u_t(I_t)\rVert^2\,\mathrm{d} t\\
 &\qquad\quad+\int_0^1\mathbb E\lVert u_t(Z_t)-(Z_1-Z_0)\rVert^2\,\mathrm{d} t .
\end{aligned}
```

第一项度量当前位置下不同训练位移的波动，第二项度量流轨迹上速度偏离整段平均速度的程度。
若每轮都精确实现边缘并使用精确流产生新耦合，
迭代的平方位移代价下降且有下界，因而这些差量的总和有界。
实际用有限求解步数生成配对时，新耦合的端点分布与上述理想情形可能有偏差。

<a name="sec-2-5-5"></a>

### 2.5.5 最优耦合与训练目标之间的关系

选择二次 OT 耦合
$`\pi_{\mathrm{OT}}\in\operatorname*{arg\,min}_{\pi\in\Pi(\mu_0,\mu_1)} \mathbb E_\pi\lVert D-N\rVert^2`$，再沿其端点线性插值，
得到 $`W_2`$ 位移插值。对于平方代价，任意耦合产生的条件期望速度满足

```math
W_2^2(\mu_0,\mu_1)
\le\int_0^1\mathbb E\lVert u_t(I_t)\rVert^2\,\mathrm{d} t
\le\mathbb E_\pi\lVert N-D\rVert^2 .
```

第一个不等式来自 Benamou–Brenier 动态公式
[(0.26)](chapter-0.md#eq-foundations-bb)，第二个来自条件 Jensen。
在精确 OT、无内部扰动的线性路径中两端相等，
故监督的剩余方差在时空积分下为零。独立配对通常会保留这一方差。
OT-CFM 也可为给定配对选取具有正方差的条件高斯路径
〔[7](#ref-tong2024otcfm)〕，此时需按具体路径再计算其速度与方差。
实际的 minibatch OT 在各自的经验端点上解运输问题，再从该计划抽取训练对。
批内均匀权重保持批内两个经验边缘。对随机批取平均后，耦合仍连接两个人口边缘，
但一般与全局 OT 耦合有差异。

<a name="sec-2-6"></a>

## 2.6 Schrödinger Bridge 的训练理论

<a name="sec-2-6-1"></a>

### 2.6.1 路径空间相对熵的定义

令 $`\Omega=C([0,1],\mathbb R^d)`$，以 $`X_t`$ 表示坐标过程。
若 $`P\ll R`$，定义
$`D_{\mathrm{KL}}(P\|R)=\mathbb E_P[\log(\mathrm{d} P/\mathrm{d} R)]`$。
若 $`P`$ 对 $`R`$ 不绝对连续，取 $`+\infty`$。
在离散网格上，若两律均为正向 Markov，
对联合密度比取对数与期望，可得

```math
D_{\mathrm{KL}}(P\|R)
=D_{\mathrm{KL}}(P_0\|R_0)+
  \sum_{k=1}^K\mathbb E_{P_{k-1}}
         D_{\mathrm{KL}}(P_{k\mid k-1}\|R_{k\mid k-1}).
```

路径 KL 因而依赖相邻时刻的条件律。在连续时间更换扩散系数
需要先检查绝对连续性。比如两种非零常数扩散强度不同的 Brownian 律，
二次变差几乎必然取不同值，路径律互相奇异。

<a name="sec-2-6-2"></a>

### 2.6.2 Schrödinger Bridge 约束优化问题

选定参考扩散 $`R`$，在路径律上最小化

<a name="eq-training-sb-primal"></a>

**公式（2.22）**

```math
\inf_{P:\,P_0=\mu_0,\ P_1=\mu_1}D_{\mathrm{KL}}(P\|R).
```

把 $`P`$ 和 $`R`$ 都分解为端点律与条件桥。记 $`\pi=P_{01}`$，
由 KL 链式法则

```math
D_{\mathrm{KL}}(P\|R)
 =D_{\mathrm{KL}}(\pi\|R_{01})
    +\int D_{\mathrm{KL}}(P^{x,y}\|R^{x,y})\,\pi(\mathrm{d} x,\mathrm{d} y),
```

其中 $`P^{x,y}`$ 是给定 $`X_0=x,X_1=y`$ 的条件路径律。
对每个可行且有限 KL 的 $`\pi`$，令 $`P^{x,y}=R^{x,y}`$
使第二项为零。于是只需在 $`\Pi(\mu_0,\mu_1)`$
上解静态目标 $`\inf_\pi D_{\mathrm{KL}}(\pi\|R_{01})`$
〔[13](#ref-leonard2014schrodinger)〕。
若 $`R_0=\mu_0`$，参考为常数扩散强度 $`\sqrt{\eta}`$ 的 Brownian 过程，
则 $`R_{01}(\mathrm{d} x,\mathrm{d} y)=\mu_0(\mathrm{d} x)\mathcal N(x,\eta I_d)(\mathrm{d} y)`$。
在 $`\mu_1`$ 有密度、相关熵与二阶矩有限的情形，
把 $`\mathrm{d} \pi/\mathrm{d} R_{01}`$ 分解为相对于 $`\mu_0\otimes\mu_1`$ 的密度比
和 $`\mu_1`$ 相对于高斯端点核的密度比，可得

<a name="eq-training-sb-entropic-cost"></a>

**公式（2.23）**

```math
\eta D_{\mathrm{KL}}(\pi\|R_{01})
 =\tfrac12\mathbb E_\pi\lVert X_1-X_0\rVert^2
         +\eta D_{\mathrm{KL}}(\pi\|\mu_0\otimes\mu_1)+C .
```

常数 $`C`$ 只由 $`\eta,d,\mu_1`$ 决定。
参考端点的绝对连续性限制了哪些目标边缘可在有限 KL 下实现。

<a name="sec-2-6-3"></a>

### 2.6.3 Girsanov 定理与漂移控制代价 ★

设参考过程为 $`\mathrm{d} X_t=f_t^R(X_t)\,\mathrm{d} t+G_t(X_t)\,\mathrm{d} W_t^R`$，
此处取 $`G_t`$ 为可逆方阵。考虑同一扩散因子和漂移 $`f_t^R+G_t h_t`$ 的受控律 $`P`$。
令 $`h_t`$ 是可从路径得到的适应控制，先把 $`R`$ 按初始密度比
$`\mathrm{d} P_0/\mathrm{d} R_0(X_0)`$ 加权为参考律 $`\widetilde R`$。
在受控律由 Girsanov 变换给出、指数局部鞅在 $`\widetilde R`$ 下为真鞅
（如满足相应的 Novikov 条件）、且控制能量有限时，

```math
\log\frac{\mathrm{d} P}{\mathrm{d} R}
 =\log\frac{\mathrm{d} P_0}{\mathrm{d} R_0}(X_0)
    +\int_0^1 h_t^\top\,\mathrm{d} W_t^R-\tfrac12\int_0^1\lVert h_t\rVert^2\,\mathrm{d} t .
```

在 $`P`$ 下 $`W_t^P=W_t^R-\int_0^t h_s\,\mathrm{d} s`$ 为 Brownian 运动。
代入上式后，随机积分对 $`P`$ 取期望为零，二次项留下正的系数 $`1/2`$。
记 $`c_t=G_th_t`$ 为漂移增量、$`a_t=G_tG_t^\top`$，得到

<a name="eq-training-girsanov"></a>

**公式（2.24）**

```math
D_{\mathrm{KL}}(P\|R)
  =D_{\mathrm{KL}}(P_0\|R_0)
     +\tfrac12\mathbb E_P\int_0^1 c_t^\top a_t^{-1}c_t\,\mathrm{d} t .
```

因此对于固定的初始律，Schrödinger 问题可在相应的可容许控制类中写成
带终点分布约束的最小能量控制。若 $`a_t=g(t)^2 I_d`$，代价密度为
$`\lVert c_t\rVert^2/(2g(t)^2)`$，期望由受控过程 $`P`$ 给出。

**例 2.4 · 平移的 Brownian 桥接**

若 $`a=\eta I_d`$、$`f^R=0`$、$`R_0=\mu_0`$，并将目标终点取为
参考终点 $`R_1`$ 的平移 $`m`$，则常控制 $`c_t=m`$ 可行。
当端点具有有限一阶矩时，任何可行且有限能量的控制满足
$`\mathbb E_P\int_0^1c_t\,\mathrm{d} t=m`$。
Jensen 不等式给出代价下界 $`\lVert m\rVert^2/(2\eta)`$，常控制达到该下界。

<a name="sec-2-6-4"></a>

### 2.6.4 Schrödinger 系统及其势函数

考虑存在正势函数的情形，取

```math
\pi^\star(\mathrm{d} x,\mathrm{d} y)=\ell(x)h(y)R_{01}(\mathrm{d} x,\mathrm{d} y).
```

可以先验证其最优性。若 $`\ell,h`$ 使两端边缘正确且积分有限，
则任意可行 $`\pi`$ 的
$`\int\log(\ell(x)h(y))\,\pi(\mathrm{d} x,\mathrm{d} y)`$
都取同一值。将密度比分成
$`\mathrm{d}\pi/\mathrm{d}\pi^\star`$ 和 $`\mathrm{d}\pi^\star/\mathrm{d} R_{01}`$ 后，
目标差正好是 $`D_{\mathrm{KL}}(\pi\|\pi^\star)`$。
由条件桥表示，相应的路径密度比为 $`\ell(X_0)h(X_1)`$。

设 $`R`$ 为 Markov 律，边缘密度为 $`r_t(x)`$。
定义 $`\widehat\psi_t(x)=\mathbb E_R[\ell(X_0)\mid X_t=x]`$、
$`\psi_t(x)=\mathbb E_R[h(X_1)\mid X_t=x]`$。
由过去与未来在 $`X_t`$ 下条件独立，

<a name="eq-training-sb-system"></a>

**公式（2.25）**

```math
p_t^\star=r_t\widehat\psi_t\psi_t,\qquad
\ell\,\psi_0\,r_0=p_{\mathrm{data}},\qquad
h\,\widehat\psi_1\,r_1=p_{\mathrm{prior}} .
```

同时按 $`c`$ 与 $`1/c`$ 缩放 $`\ell`$ 和 $`h`$ 保持路径律。
若 $`\psi_t>0`$ 足够光滑，桥的正向转移通过
$`R_{s,t}(x,\mathrm{d} y)\psi_t(y)/\psi_s(x)`$ 加权。
参考生成元对乘积 $`\psi_t\varphi`$ 的交叉项是
$`(\nabla\varphi)^\top a_t\nabla\psi_t`$，
对 $`\psi_t`$ 归一化后给出正向漂移修正 $`a_t\nabla\log\psi_t`$。
用反向参考过程作相同计算，生成时间下的漂移修正是
$`a_{1-\tau}\nabla\log\widehat\psi_{1-\tau}`$，见
[(1.18)](chapter-1.md#eq-overview-sb-drifts)。

<a name="sec-2-6-5"></a>

### 2.6.5 Iterative Proportional Fitting（IPF）

设当前路径律 $`P^n`$ 尚未满足终点约束。
在 $`Q_1=\mu_1`$ 下最小化 $`D_{\mathrm{KL}}(Q\|P^n)`$ 时，
链式分解的端点项固定，条件路径 KL 在保留 $`P^n(\cdot\mid X_1)`$
时为零，因此 $`Q^n`$ 是终点密度比加权的 $`P^n`$。
起点投影同理。只要涉及的边缘允许相应密度比，迭代可写为

<a name="eq-training-ipf"></a>

**公式（2.26）**

```math
\frac{\mathrm{d} Q^n}{\mathrm{d} P^n}=\frac{\mathrm{d}\mu_1}{\mathrm{d} P^n_1}(X_1),\qquad
\frac{\mathrm{d} P^{n+1}}{\mathrm{d} Q^n}=\frac{\mathrm{d}\mu_0}{\mathrm{d} Q^n_0}(X_0),
\qquad P^0=R.
```

每半步精确匹配一端并保留对应的条件律，另一端一般随之变化。
在静态离散情形，这正是交替进行列缩放与行缩放。
在正的有限参考矩阵和正目标边缘下，迭代收敛到满足两个约束的 KL 投影。

**例 2.5 · 一次 IPF 的两个半步**

设 $`R_{01}`$ 的两行是 $`(3/8,1/8)`$、$`(1/8,3/8)`$，
目标行和为 $`(1/2,1/2)`$，目标列和为 $`(1/4,3/4)`$。
先把两列分别乘 $`1/2,3/2`$，行和变成 $`(3/8,5/8)`$。
再把两行分别乘 $`4/3,4/5`$，行和正确，列和变为 $`(3/10,7/10)`$。
每个半步得到的矩阵可直接核对其所满足的边缘约束。

<a name="sec-2-6-6"></a>

### 2.6.6 Diffusion Schrödinger Bridge 的迭代训练

DSB 用交替模拟与回归近似 [(2.26)](#eq-training-ipf)
〔[14](#ref-debortoli2021dsb)〕。记网格步长为 $`h_k=t_k-t_{k-1}>0`$。
从数据端模拟正向核
$`X_k=F_{k-1}^n(X_{k-1})+\sqrt{2h_k}\,\xi_k`$，
其中 $`F_{k-1}^n(x)=x+h_k f_{k-1}^n(x)`$。
对正向边缘 $`p_k^n`$ 应用条件 Score 恒等式，得到

```math
2h_k\nabla\log p_k^n(y)=
  \mathbb E[F_{k-1}^n(X_{k-1})-y\mid X_k=y].
```

Euler 反向漂移取 $`-f_{k-1}^n(y)+2\nabla\log p_k^n(y)`$。
将上式代入其均值映射 $`y+h_k[-f_{k-1}^n(y)+2\nabla\log p_k^n(y)]`$，
即可将反向训练写成一次条件回归。两轮的样本监督为

<a name="eq-training-dsb-regression"></a>

**公式（2.27）**

```math
\begin{aligned}
B_k^n(X_k)&\quad\text{拟合}\quad
    X_k+F_{k-1}^n(X_{k-1})-F_{k-1}^n(X_k),\\
F_{k-1}^{n+1}(X_{k-1})&\quad\text{拟合}\quad
    X_{k-1}+B_k^n(X_k)-B_k^n(X_{k-1}).
\end{aligned}
```

第一式在从数据出发的正向样本对上训练，第二式在从先验出发、
用已拟合的 $`B_k^n`$ 模拟出的反向样本对上训练。
两个目标都来自高斯转移核的 Score 恒等式，最优回归器拟合相应的 Euler 均值。
有限网格中，真实离散时间反向条件核可能超出同方差高斯模型的表达能力。
精确 IPF 的路径投影、Euler 核的近似以及有限样本的网络回归各有自己的误差来源。

<a name="sec-2-7"></a>

## 2.7 Stochastic Interpolants 的训练理论

<a name="sec-2-7-1"></a>

### 2.7.1 随机插值函数及端点条件

取 $`(D,N)\sim\pi`$，另取独立的 $`\zeta\sim\mathcal N(0,I_d)`$，令

<a name="eq-training-si-path"></a>

**公式（2.28）**

```math
\begin{gathered}
I_t=\mathcal I(t,D,N)+\gamma(t)\zeta,\\
\mathcal I(0,D,N)=D,\quad\mathcal I(1,D,N)=N,\quad
\gamma(0)=\gamma(1)=0.
\end{gathered}
```

当内点 $`\gamma(t)>0`$，给定端点后的条件律为
$`\mathcal N(\mathcal I(t,D,N),\gamma(t)^2 I_d)`$。
这提供了一条可采样且端点精确的训练路径 〔[8](#ref-albergo2025si)〕。
端点 $`N`$ 参与耦合，辅助噪声 $`\zeta`$ 独立于两个端点。
两者即使有相同的边缘分布，也在回归标签中起不同作用。

<a name="sec-2-7-2"></a>

### 2.7.2 插值变量的时间导数

在每条曲线上固定一次抽取的 $`D,N,\zeta`$。当系数可微时有

```math
\dot I_t=\partial_t\mathcal I(t,D,N)+\gamma'(t)\zeta .
```

例如 $`\mathcal I=\alpha_tD+\sigma_tN`$ 时，标签为
$`\dot\alpha_tD+\dot\sigma_tN+\gamma'(t)\zeta`$。
选择绝对连续而端点导数可能奇异的系数时，应检查
$`\mathbb E\lVert \dot I_t\rVert^2`$ 在实际训练时间律下能否积分。
逐次训练只取一个时刻。若为每个时刻重新抽噪声，
仍可得到相同的单时刻边缘，上述路径导数则对应固定 $`\zeta`$ 的构造。

<a name="sec-2-7-3"></a>

### 2.7.3 速度场的条件期望表示 ★

对于 $`\varphi\in C_c^\infty(\mathbb R^d)`$，在可交换求导与期望时，
普通链式法则和条件化给出

<a name="eq-training-si-velocity"></a>

**公式（2.29）**

```math
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} t}\mathbb E\varphi(I_t)
 &=\mathbb E[\nabla\varphi(I_t)^\top\dot I_t]
  =\mathbb E[\nabla\varphi(I_t)^\top u_t(I_t)],\\
u_t(x)&=\mathbb E[\dot I_t\mid I_t=x] .
\end{aligned}
```

这给出 $`p_t=\operatorname{Law}(I_t)`$ 的弱连续性方程。
因此包括 $`\gamma'(t)\zeta`$ 在内的完整时间导数都进入速度标签。
舍去这一项时，最优回归场少了 $`\gamma'(t)\mathbb E[\zeta\mid I_t=x]`$。

<a name="sec-2-7-4"></a>

### 2.7.4 Score 的条件回归表示

对条件高斯密度求梯度，得到标签
$`-(I_t-\mathcal I(t,D,N))/\gamma(t)^2=-\zeta/\gamma(t)`$。
按 [(2.11)](#eq-training-dsm-projection) 对端点积分，

<a name="eq-training-si-score"></a>

**公式（2.30）**

```math
s_t(x)=-\gamma(t)^{-1}\mathbb E[\zeta\mid I_t=x],
\qquad \gamma(t)>0 .
```

因此也可先拟合噪声条件均值，再在内点除以 $`-\gamma(t)`$。
端点 $`\gamma(t)=0`$ 时应依据端点自身的密度或取适当极限。
由 [(2.29)](#eq-training-si-velocity) 和 [(2.30)](#eq-training-si-score)，
$`u_t(x)=\mathbb E[\partial_t\mathcal I(t,D,N)\mid I_t=x]-\gamma'(t)\gamma(t)s_t(x)`$。

<a name="sec-2-7-5"></a>

### 2.7.5 二次回归损失与其最优解 ★

令 $`U_\theta(I_t,t)`$ 回归 $`\dot I_t`$，
令 $`Z_\phi(I_t,t)`$ 回归 $`\zeta`$。
两项损失可联合写为

<a name="eq-training-si-loss"></a>

**公式（2.31）**

```math
\mathcal L(\theta,\phi)=\mathbb E\!\left[
     \lambda_u(t)\lVert U_\theta(I_t,t)-\dot I_t\rVert^2+
     \lambda_z(t)\lVert Z_\phi(I_t,t)-\zeta\rVert^2\right].
```

对任一标签 $`T`$，条件于 $`(I_t,t)`$ 将平方展开，
得到与 $`\mathbb E[T\mid I_t,t]`$ 的平方距离加上该条件下的剩余方差。
于是权重为正、容量充分且目标可积时，
两个最优输出分别是 $`u_t`$ 与 $`-\gamma(t)s_t`$。
也可以使用形如 $`\mathbb E[\tfrac12\lVert U_\theta\rVert^2-U_\theta^\top\dot I_t]`$
的二次泛函，它与半倍速度 MSE 只差标签二次矩的一半。
直接拟合 Score 时，条件 Score MSE 与噪声 MSE 相差 $`1/\gamma(t)^2`$ 的因子，
所以两种写法的时间权重需相应换算。

<a name="sec-2-7-6"></a>

### 2.7.6 不同耦合与随机插值系数的影响

耦合改变给定 $`I_t`$ 后的端点后验，$`\gamma`$ 改变内点平滑程度及派生监督。
给定拟合的 $`u_t,s_t`$，将 $`f_t=u_t+\kappa(t)s_t`$
与扩散强度 $`\sqrt{2\kappa(t)}`$ 配对，可构造有相同边缘的 SDE。
在 Fokker–Planck 方程中，$`-\kappa\nabla\cdot(s_tp_t)`$ 抵消
$`\kappa\Delta p_t`$。实现这族边缘还要求方程适定、初始律匹配且场准确。
这里的 $`\kappa`$ 选择采样动力学，训练路径已由 $`\pi,\mathcal I,\gamma`$ 给定。

**例 2.6 · 用高斯插值同时核对速度与 Score**

若 $`D,N,\zeta`$ 是独立标准高斯，$`\mathcal I=(1-t)D+tN`$，
则各坐标的方差为 $`H_t=(1-t)^2+t^2+\gamma(t)^2`$。
由联合高斯的条件均值得
$`u_t(x)=H_t' x/(2H_t)`$ 和
$`\mathbb E[\zeta\mid I_t=x]=\gamma(t)x/H_t`$。
因此 $`s_t(x)=-x/H_t`$，且 ODE $`\dot Z_t=u_t(Z_t)`$ 的解为
$`Z_t=\sqrt{H_t/H_0}\,Z_0`$。
两式分别重现直接计算的高斯 Score 和连续性方程的方差变化。

<a name="sec-2-8"></a>

## 2.8 训练目标之间的等价性与差异

<a name="sec-2-8-1"></a>

### 2.8.1 噪声预测与 Score Matching

在 $`X_t=\alpha_tD+\sigma_t\varepsilon`$、$`\sigma_t>0`$ 的固定高斯路径上，
令预测满足 $`s_\theta(x,t)=-\varepsilon_\theta(x,t)/\sigma_t`$，
把总体噪声损失对条件均值展开，便有逐时刻恒等式

<a name="eq-training-eps-score"></a>

**公式（2.32）**

```math
\begin{aligned}
\mathbb E\lVert \varepsilon_\theta(X_t,t)-\varepsilon\rVert^2
 &=\sigma_t^2\mathbb E\lVert s_\theta(X_t,t)-s_t(X_t)\rVert^2\\
 &\quad+\mathbb E\lVert \varepsilon-\mathbb E[\varepsilon\mid X_t]\rVert^2 .
\end{aligned}
```

固定时刻的最优预测可相互换算。对时间取平均时，直接使用相同权重
会得到不同的损失度量。若噪声协方差为已知的可逆矩阵 $`LL^\top`$，
且 $`X_t=\alpha_tD+L\varepsilon`$，条件 Score 标签为 $`-L^{-\top}\varepsilon`$，
误差的欧氏范数按相应矩阵变换。

#### 同一路径上的速度换算

当 $`\alpha_t\ne0`$ 时，边缘速度 $`u_t(x)=\mathbb E[\dot X_t\mid X_t=x]`$ 可写为

<a name="eq-training-velocity-score"></a>

**公式（2.33）**

```math
\begin{aligned}
u_t(x)&=\frac{\dot\alpha_t}{\alpha_t}x+C_t s_t(x),\\
C_t&=\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2-\sigma_t\dot\sigma_t .
\end{aligned}
```

这来自用 $`D=(X_t-\sigma_t\varepsilon)/\alpha_t`$ 代入
$`\dot X_t=\dot\alpha_tD+\dot\sigma_t\varepsilon`$，再使用 [(2.11)](#eq-training-dsm-projection)。
对已知路径，$`C_t\ne0`$ 才能从这一速度恢复 Score。
对 VP 路径，$`C_t=-\beta(t)/2`$，重得概率流速度。

<a name="sec-2-8-2"></a>

### 2.8.2 Score Matching 与 Denoising Score Matching

式 [(2.12)](#eq-training-dsm-loss) 给出同一时刻、同一带噪边缘上的加常数恒等式。
若边界条件也允许分部积分，Hyvärinen 形式 [(2.10)](#eq-training-implicit-sm)
就是该边缘 Score 损失的另一种加常数表示。
两种可计算损失需要的量不同。隐式形式使用网络散度，
DSM 使用扰动核的条件 Score。若把隐式 Score Matching 直接用在干净数据上，
它拟合的是数据端的 Score。该场与正噪声时刻 $`s_t`$ 一般不同，
并且当干净数据没有全维密度时，普通数据端 Score 可能无定义。

<a name="sec-2-8-3"></a>

### 2.8.3 Flow Matching 与 Conditional Flow Matching

式 [(2.18)](#eq-training-cfm-identity) 成立于固定路径与固定边缘速度选择。
边缘匹配本身只给出 $`\partial_t p_t`$。如果改变条件路径或路径的耦合，
即便时刻边缘一致，也可能改变条件速度的后验平均。
例如二维标准高斯在恒等流和恒角速度旋转下均保持不变。
两种速度场都使 $`\nabla\cdot(p_tu_t)=0`$，且它们对应不同的路径律和 FM 回归目标。

<a name="sec-2-8-4"></a>

### 2.8.4 Drift Matching、Velocity Matching 与 Score Matching

给定前向 SDE 的漂移 $`f_t`$ 与扩散矩阵 $`a_t`$，选用其概率流速度
[(0.24)](chapter-0.md#eq-foundations-probability-velocity)，有

<a name="eq-training-drift-score"></a>

**公式（2.34）**

```math
u_t=f_t-\tfrac12\operatorname{div}a_t-\tfrac12a_t s_t,
\qquad
b_\tau=-f_t+\operatorname{div}a_t+a_t s_t,\quad t=1-\tau .
```

$`b_\tau`$ 是递增生成时间下的反向 SDE 漂移。
固定 $`f_t,a_t`$ 时，如果三个预测通过上式显式关联，
其误差满足 $`\delta u_t=-a_t\delta s_t/2`$、$`\delta b_\tau=a_t\delta s_t`$。
因此速度欧氏 MSE 等于以 $`a_t^\top a_t/4`$ 加权的 Score MSE。
反向漂移欧氏 MSE 则对应 $`a_t^\top a_t`$。
若采用路径 KL 的漂移度量 $`a_t^{-1}`$，有相应的权矩阵 $`a_t`$。
这些关系需要使用同一边缘并已知扩散与前向漂移。
若 $`a_t`$ 奇异，从 $`a_t s_t`$ 只能识别 Score 在其像空间上的投影。

<a name="sec-2-8-5"></a>

### 2.8.5 各类训练目标的适用条件与不等价情形

- **损失相差常数。** 固定训练时空分布与模型输出，DSM 和边缘 Score Matching、
   CFM 和 FM 分别满足条件投影分解。常数与模型参数无关，有限容量下的总体最优解也一致。
- **监督可以换算。** 对同一高斯加噪路径，噪声、干净数据、扩散
   $`V^{\mathrm{diff}}`$ 与 Score 在相应系数非退化处可互换。
   要使参数化后的损失逐项相等，须同时变换误差度量和时间权重。
- **目标分布或路径改变。** DDPM、DDIM 共享训练边缘，但反向核可以不同。
   FM 中改变耦合会改变监督。Schrödinger Bridge 还需要相对于指定参考律最小化路径 KL，
   两端匹配和逐时刻回归精确只构成其中的部分条件。

**例 2.7 · 相同逐时刻目标在共享参数下的差异**

考虑两个同概率的时刻，标量回归真值分别为 $`0,1`$，网络只能输出同一个常数 $`c`$。
若两个时刻权重相同，最优 $`c=1/2`$。若后一个时刻权重为前一个的三倍，
则最优 $`c=3/4`$。这说明即使两种回归有相同的逐时刻最优值，
时间加权也会改变共享网络的总体最优参数。

#### 本章练习与核对

1. 若 DDPM 的反向方差固定为 $`\nu_k=\widetilde\beta_k`$，
   化简 [(2.5)](#eq-training-ddpm-weight) 中 $`k\ge2`$ 的权重。
   **核对** $`\beta_k/(2r_k S_{k-1}^2)`$。
1. 在 [(2.8)](#eq-training-ddim-kernel) 中将 $`\delta_k`$ 从 $`0`$
   改为 $`S_{k-1}`$，求两个极端的条件协方差。
   **核对** 分别为 $`S_k S_{k-1}I_d`$ 与 $`0`$。
1. 对 $`D\sim\mathcal N(0,4)`$、$`X=(D+\varepsilon)/\sqrt 2`$
   求最优噪声回归与 Score。
   **核对** $`\sqrt 2\,x/5`$ 与 $`-2x/5`$。
1. 对二维单位高斯，令 $`J`$ 为非零反对称矩阵。
   验证速度 $`u(x)=Jx`$ 满足静态连续性方程。
   **核对** $`\operatorname{tr}J=0`$，且 $`x^\top Jx=0`$。
1. 若随机插值取 $`\gamma(t)=\sqrt{t(1-t)}`$，
   问在均匀时间采样下，独立噪声的导数项是否平方可积。
   **核对** $`\gamma'(t)^2=(1-2t)^2/[4t(1-t)]`$，
   两端积分发散。可检查截断或权重能否使所用训练目标有限。

<a name="references"></a>

## 参考文献

<a name="ref-ho2020ddpm"></a>
**[1]** Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).

<a name="ref-song2021ddim"></a>
**[2]** Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. ICLR, 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502).

<a name="ref-song2021sde"></a>
**[3]** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations. ICLR, 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456).

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

<a name="ref-salimans2022distill"></a>
**[15]** Tim Salimans and Jonathan Ho. Progressive Distillation for Fast Sampling of Diffusion Models. ICLR, 2022. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512).

<a name="ref-hyvarinen2005sm"></a>
**[17]** Aapo Hyvärinen. Estimation of Non-Normalized Statistical Models by Score Matching. Journal of Machine Learning Research, 6(24), 695–709, 2005. [论文](https://www.jmlr.org/papers/v6/hyvarinen05a.html).

<a name="ref-vincent2011dsm"></a>
**[18]** Pascal Vincent. A Connection Between Score Matching and Denoising Autoencoders. Neural Computation, 23(7), 1661–1674, 2011. [DOI](https://doi.org/10.1162/NECO_a_00142).

<a name="ref-song2021likelihood"></a>
**[19]** Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum Likelihood Training of Score-Based Diffusion Models. NeurIPS, 2021. [arXiv:2101.09258](https://arxiv.org/abs/2101.09258).
