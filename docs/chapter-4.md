# 第 4 章　Score、Velocity、Drift 的计算、估计与相互转换

[返回项目目录](../README.md#contents) · [上一章](chapter-3.md) · [下一章](chapter-5.md) · [PDF 版](../output/pdf/model-study.pdf) · [LaTeX 源码](../chapters/04-score-velocity-drift.tex)

[4.1 Score 函数的基本理论](#sec-4-1) · [4.2 Score Matching 的数学推导](#sec-4-2) · [4.3 Score 与噪声预测之间的关系](#sec-4-3) · [4.4 Velocity Field 的数学推导](#sec-4-4) · [4.5 Drift Field 与 Score 的关系](#sec-4-5) · [4.6 Flow Matching 与 Score Matching 的联系](#sec-4-6) · [4.7 Schrödinger Bridge 中的 Score 与漂移](#sec-4-7) · [4.8 Score 及向量场的计算实现](#sec-4-8)

本章围绕同一族边缘分布 $`p_t`$ 组织各类场的定义、回归目标与转换。
沿用 $`t=0`$ 为数据端、$`t=1`$ 为噪声端，生成时间为 $`\tau=1-t`$。
加噪时间的边缘速度记为 $`u_t`$，生成速度为 $`v_\tau=-u_{1-\tau}`$，
前向 Itô 漂移为 $`f_t`$，递增生成时间的反向漂移为 $`b_\tau`$。
空间 Score 始终定义为 $`s_t(x)=\nabla_x\log p_t(x)`$。
反转时间后，$`\rho_\tau(x)=p_{1-\tau}(x)`$ 的空间 Score 为 $`s_{1-\tau}(x)`$。

涉及经典导数时，假设相应内部时刻具有正的光滑密度，且微分、积分、条件期望的
交换有可积控制。分部积分另需无穷远衰减或适当边界条件。
干净端点允许是一般概率测度，含有 $`1/\sigma_t`$、$`1/\alpha_t`$
或矩阵逆的公式只在所列系数可逆处使用。

**学习目标**

- 从密度导数与条件期望推出边缘 Score、Stein 恒等式和 DSM 投影。
- 完整换算噪声、干净样本、扩散预测标签与输运速度，并检查端点退化。
- 从概率流推导一般扩散中的 Drift、Velocity 与 Score 关系。
- 用桥势函数解释双向漂移，建立解析求导、回归、散度估计与残差核对流程。

<a name="sec-4-1"></a>

## 4.1 Score 函数的基本理论

<a name="sec-4-1-1"></a>

### 4.1.1 Score 的定义及几何意义

设 $`p`$ 在开集 $`\Omega\subset\mathbb R^d`$ 上正且可微，定义

<a name="eq-fields-score-definition"></a>

```math
s(x)=\nabla_x\log p(x)=\frac{\nabla_xp(x)}{p(x)},\qquad
 \log p(x+h)=\log p(x)+s(x)^\top h+o(\lVert h\rVert).
\tag{4.1}
```

对单位方向 $`e`$，$`e^\top s`$ 是对数密度的方向导数。
非零 Score 指向对数密度局部增长最快的方向，其大小表示局部相对变化率。
临界点的 Score 为零，驻点类型还需 Hessian 判别。
若 $`p=\widetilde p/Z`$ 且 $`Z`$ 与 $`x`$ 无关，
则 $`s=\nabla\log\widetilde p`$。这使未归一化能量模型可以直接计算 Score。
本文求导变量为样本位置 $`x`$，统计模型对参数求导的参数 Score 另有自己的定义。

对可逆线性坐标变换 $`y=Bx+c`$，密度变换给出
$`s_Y(y)=B^{-\top}s_X(B^{-1}(y-c))`$。
因此数据归一化会改变 Score 的尺度，恢复原始坐标时需一并变换。
若使用非线性坐标变换，还需对 Jacobian 行列式项求导。

<a name="sec-4-1-2"></a>

### 4.1.2 高斯分布 Score 的显式计算 ★

令 $`X\sim\mathcal N(m,\Sigma)`$，$`\Sigma`$ 对称正定。
将密度取对数得到

```math
\log p(x)=-\tfrac d2\log(2\pi)-\tfrac12\log\det\Sigma
               -\tfrac12(x-m)^\top\Sigma^{-1}(x-m).
```

由于 $`\Sigma^{-1}`$ 对称，二次型求导给出

<a name="eq-fields-gaussian-score"></a>

```math
s(x)=-\Sigma^{-1}(x-m),\qquad
 \nabla_xs(x)=-\Sigma^{-1},\qquad
 \nabla\cdot s(x)=-\operatorname{tr}(\Sigma^{-1}).
\tag{4.2}
```

在协方差特征向量方向，Score 的恢复强度是对应方差的倒数。
计算时可求解线性系统 $`\Sigma z=x-m`$，再令 $`s=-z`$。
若协方差奇异，分布集中在低维仿射子空间，普通全维 Lebesgue 密度需重新定义。
对 $`p_t=\mathcal N(m_t,\Sigma_t)`$，同一公式逐时刻成立。

**例 4.1 · 相关高斯**

取 $`m=0`$、$`\Sigma=\begin{pmatrix}2&1\\1&2\end{pmatrix}`$，则

```math
s(x)=-\tfrac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix}x.
```

沿 $`(1,1)`$ 的方差为 $`3`$，沿 $`(1,-1)`$ 的方差为 $`1`$，
所以后一方向的 Score 恢复强度更大。

<a name="sec-4-1-3"></a>

### 4.1.3 条件 Score 与边缘 Score

设隐变量 $`Z\sim\pi`$，$`p(x)=\int p(x\mid z)\pi(\mathrm{d}  z)`$。
将 $`\nabla_xp(x)`$ 移入积分，再乘除条件密度，得到

<a name="eq-fields-fisher-identity"></a>

```math
s(x)=\int\nabla_x\log p(x\mid z)\,\pi(\mathrm{d}  z\mid x)
     =\mathbb E[\nabla_x\log p(X\mid Z)\mid X=x].
\tag{4.3}
```

平均权重是观察到 $`x`$ 后的后验。直接按先验权重平均条件 Score
通常得到另一个场。
若 $`p(x)=\sum_jw_j\varphi_j(x)`$ 是高斯混合，定义责任权重
$`\omega_j(x)=w_j\varphi_j(x)/\sum_\ell w_\ell\varphi_\ell(x)`$，则
$`s(x)=-\sum_j\omega_j(x)\Sigma_j^{-1}(x-m_j)`$。
该表达式也给出稳定实现方法，先用 log-sum-exp 计算责任权重。

**例 4.2 · 两分量混合中的后验权重**

设 $`p=\tfrac12\mathcal N(-m,c)+\tfrac12\mathcal N(m,c)`$，$`m,c>0`$。
计算两个分量的密度比，得到

```math
s(x)=\frac{-x+m\tanh(mx/c)}c.
```

在 $`x=0`$ 处 Score 为零，且 $`s'(0)=(m^2-c)/c^2`$。
当 $`m^2>c`$ 时原点是局部密度低谷。驻点处的 Score 数值需要结合导数解释。

<a name="sec-4-1-4"></a>

### 4.1.4 Score 的积分恒等式

在相关量可积且边界通量消失时，$`\int\nabla p=0`$，所以
$`\mathbb E_p s(X)=0`$。对光滑向量测试函数 $`h`$，
积分 $`\nabla\cdot(ph)=p\nabla\cdot h+h^\top\nabla p`$，得到 Stein 恒等式

<a name="eq-fields-stein"></a>

```math
\mathbb E_p[s(X)^\top h(X)+\nabla\cdot h(X)]=0.
\tag{4.4}
```

取第 $`i`$ 个分量为 $`x_j`$、其余为零，得
$`\mathbb E[s_i(X)X_j]=-\delta_{ij}`$。
取 $`h=s`$ 则得
$`\mathbb E\lVert s\rVert^2=-\mathbb E\nabla\cdot s`$，前提是相应积分仍成立。
若支持有边界，右端需保留 $`\int_{\partial\Omega}ph\cdot n`$；
例如区间上的均匀密度在内部 Score 为零，但边界项会影响上述恒等式。

<a name="sec-4-1-5"></a>

### 4.1.5 Score 在概率密度高维估计中的作用

Score 允许通过局部导数构建回归与动力学。
若在连通区域上已知精确的保守场 $`s=\nabla\log p`$，沿任意连接路径 $`\Gamma`$
有 $`\log p(x)-\log p(x_*)=\int_\Gamma s\cdot\mathrm{d}  x`$。
恢复完整密度还需确定归一化常数，多个连通分量之间的相对质量也需额外信息。
一般向量网络的 Jacobian 可以不对称，不能自动保证它是某个密度的梯度。
对梯度场有 $`\partial_i s_j=\partial_j s_i`$；在单连通区域中，
足够光滑的无旋场可局部到整体地构造势，所得 $`\exp`$ 势还需可积。
因此“向量场拟合准确”“能量可归一化”与“采样分布准确”分别需要对应条件。

<a name="sec-4-2"></a>

## 4.2 Score Matching 的数学推导

<a name="sec-4-2-1"></a>

### 4.2.1 Fisher Divergence 展开

固定密度 $`p`$ 与候选场 $`s_\theta`$，定义

<a name="eq-fields-fisher-expansion"></a>

```math
\begin{aligned}
 \mathcal J(\theta)&=\tfrac12\mathbb E_p\lVert s_\theta(X)-s(X)\rVert^2\\
 &=\tfrac12\mathbb E_p\lVert s_\theta(X)\rVert^2
   -\int s_\theta(x)^\top\nabla p(x)\mathrm{d}  x+C_p,\\
 C_p&=\tfrac12\mathbb E_p\lVert s(X)\rVert^2.
 \end{aligned}
\tag{4.5}
```

当 $`s_\theta=\nabla\log p_\theta`$ 时，它是带 $`1/2`$ 系数的 Fisher 散度。
若直接参数化向量场，可将其视为加权 $`L^2(p)`$ 回归风险。
有限 Fisher 信息保证 $`C_p`$ 有限；否则需要根据可计算泛函另行建立适定性。
候选密度正而光滑、支持连通且风险为零时，
$`\nabla\log(p_\theta/p)=0`$，归一化使 $`p_\theta=p`$。
这些假设也解释了 Score 风险辨识密度时的适用范围。

<a name="sec-4-2-2"></a>

### 4.2.2 分部积分消除未知真实 Score ★

设 $`\Omega_R`$ 是逐渐扩张的光滑区域。散度定理给出

```math
\int_{\Omega_R}s_\theta^\top\nabla p
 =\int_{\partial\Omega_R}p s_\theta\cdot n
                         -\int_{\Omega_R}p\nabla\cdot s_\theta.
```

在边界项趋零且其余项可取极限时，式 [(4.5)](#eq-fields-fisher-expansion) 变为

<a name="eq-fields-hyvarinen"></a>

```math
\mathcal J_{\mathrm H}(\theta)
 =\mathbb E_p\left[\tfrac12\lVert s_\theta(X)\rVert^2+
                                      \nabla\cdot s_\theta(X)\right],
 \qquad \mathcal J=\mathcal J_{\mathrm H}+C_p.
\tag{4.6}
```

这给出 Hyvärinen 目标 〔[17](#ref-hyvarinen2005sm)〕。
若 $`p_\theta\propto\exp[-E_\theta]`$，有
$`s_\theta=-\nabla E_\theta`$，目标为
$`\mathbb E[\lVert \nabla E_\theta\rVert^2/2-\Delta E_\theta]`$。
实际样本均值可由数据估计，但计算散度涉及网络对输入的导数。
优化参数时还需对这些导数反向传播。

<a name="sec-4-2-3"></a>

### 4.2.3 Denoising Score Matching 的条件期望推导 ★

取干净数据 $`D\sim\mu_0`$，扰动核 $`q_t(x\mid D)`$，并令
$`T_t=\nabla_x\log q_t(X_t\mid D)`$。
由式 [(4.3)](#eq-fields-fisher-identity)，$`\mathbb E[T_t\mid X_t]=s_t(X_t)`$。
将误差拆成 $`s_\theta-T_t=(s_\theta-s_t)+(s_t-T_t)`$，平方后取期望，
交叉项因条件期望为零而消失，于是

<a name="eq-fields-dsm-projection"></a>

```math
\mathbb E\lVert s_\theta(X_t,t)-T_t\rVert^2
 =\mathbb E\lVert s_\theta(X_t,t)-s_t(X_t)\rVert^2
                         +\mathbb E\lVert T_t-s_t(X_t)\rVert^2.
\tag{4.7}
```

右端第二项与网络无关。给定同一函数类，两个总体风险有相同最优参数，
容量充分时输出为边缘 Score。此推导采用条件投影，适用于能计算条件 Score 的核
〔[18](#ref-vincent2011dsm)〕。
在时间平均中，正权重保持逐时刻无约束最优解，但共享网络的最优参数会受权重影响。

<a name="sec-4-2-4"></a>

### 4.2.4 Sliced Score Matching

取与 $`X`$ 独立的随机方向 $`z`$，$`M=\mathbb E[zz^\top]`$ 为固定正定矩阵。
投影风险为 $`\mathcal J_M=\frac12\mathbb E_{X,z}[z^\top(s_\theta-s)]^2`$。
将平方展开，对交叉项按固定方向分部积分，得到

<a name="eq-fields-sliced"></a>

```math
\mathcal J_{\mathrm{SSM}}(\theta)=
 \mathbb E_{X,z}\left[\tfrac12(z^\top s_\theta(X))^2+
                              z^\top J_{s_\theta}(X)z\right].
\tag{4.8}
```

它与 $`\mathcal J_M`$ 相差参数无关的常数，见 〔[21](#ref-song2019sliced)〕。
当 $`M=I_d`$，该风险等于普通 Score 风险。
若方向在单位球面均匀分布，$`M=I_d/d`$，目标整体相差 $`1/d`$。
可把球面方向乘 $`\sqrt d`$ 调整到单位协方差。
若 $`M`$ 奇异，投影只约束其像空间内的误差分量。

在 $`M=I_d`$ 时，可将二次项的方向期望解析化，使用
$`\lVert s_\theta\rVert^2/2+z^\top J_{s_\theta}z`$ 作估计量。
这与在所有项上都用随机投影具有相同的期望，抽样方差则由具体分布决定。
固定方向的二次型可用 Jacobian--vector 或 vector--Jacobian 乘积计算。

<a name="sec-4-2-5"></a>

### 4.2.5 隐式与显式 Score Matching 的区别

显式边缘风险需要真实 $`s_t`$，主要用于理论分析或解析密度算例。
隐式 Hyvärinen 目标使用样本和网络散度，要求边界条件。
DSM 使用扰动样本与条件 Score 标签，要求可采样的扰动核及可积监督。
Sliced 方法以方向导数估计散度，另需控制随机方向的方差。
对于固定边缘、固定权重和正确假设，这些形式可联系到同一总体风险。
如果将隐式目标用在干净数据上、DSM 用在带噪数据上，二者拟合的密度随之不同。

对 $`X_t=\alpha_tD+\sigma_t\varepsilon`$，DSM 的噪声标签范数均值为
$`\mathbb E\lVert \varepsilon/\sigma_t\rVert^2=d/\sigma_t^2`$。
小噪声时标签二次矩增大，即使投影后的边缘 Score 风险较小，训练样本方差仍可很大。
时间权重、输出预条件和端点截断需结合该尺度选择。

<a name="sec-4-3"></a>

## 4.3 Score 与噪声预测之间的关系

<a name="sec-4-3-1"></a>

### 4.3.1 高斯扰动核的条件 Score

设 $`\varepsilon\sim\mathcal N(0,I_d)`$ 与 $`D`$ 独立，
$`X_t=\alpha_tD+\sigma_t\varepsilon`$，$`\sigma_t>0`$。
给定 $`D=d`$ 的条件核是 $`\mathcal N(\alpha_td,\sigma_t^2I_d)`$，故

<a name="eq-fields-conditional-noise-score"></a>

```math
\nabla_x\log q_t(x\mid d)
 =-\frac{x-\alpha_td}{\sigma_t^2},\qquad
 \nabla_x\log q_t(X_t\mid D)=-\frac{\varepsilon}{\sigma_t}.
\tag{4.9}
```

若改成 $`X_t=\alpha_tD+L_t\varepsilon`$，$`L_t`$ 可逆，
$`\Sigma_t=L_tL_t^\top`$，则条件 Score 为
$`-\Sigma_t^{-1}(x-\alpha_td)=-L_t^{-\top}\varepsilon`$。
矩阵噪声下必须保留转置和逆矩阵的次序。
这些表达式对加噪后的条件核成立，数据端本身可以没有密度。

<a name="sec-4-3-2"></a>

### 4.3.2 噪声预测网络的最优目标

对平方损失 $`\mathcal L_\varepsilon=\mathbb E\lVert \varepsilon_\theta(X_t,t)-\varepsilon\rVert^2`$，
条件期望投影给出 $`\varepsilon^*(x,t)=\mathbb E[\varepsilon\mid X_t=x]`$。
由条件 Score 的边缘化可得

<a name="eq-fields-noise-to-score"></a>

```math
s_t(x)=-\frac{\varepsilon^*(x,t)}{\sigma_t},\qquad
 s_\theta(x,t)=-\frac{\varepsilon_\theta(x,t)}{\sigma_t}.
\tag{4.10}
```

第二式定义网络输出的换算。只有在回归误差受控时，它才准确逼近第一式。
原始噪声样本包含给定 $`X_t`$ 后仍未知的随机性，网络总体最优输出是其后验均值。
若 $`\delta\varepsilon=\varepsilon_\theta-\varepsilon^*`$，
则 $`\delta s=-\delta\varepsilon/\sigma_t`$，小噪声会放大绝对 Score 误差。

<a name="sec-4-3-3"></a>

### 4.3.3 Tweedie 公式的推导 ★

将式 [(4.9)](#eq-fields-conditional-noise-score) 条件于 $`X_t=x`$，
并把与 $`D`$ 无关的 $`x`$ 提出，得到
$`\sigma_t^2s_t(x)=-x+\alpha_t\mathbb E[D\mid X_t=x]`$。
若 $`\alpha_t\ne0`$，则

<a name="eq-fields-tweedie"></a>

```math
d_t^*(x):=\mathbb E[D\mid X_t=x]
       =\frac{x+\sigma_t^2s_t(x)}{\alpha_t}.
\tag{4.11}
```

经典加性高斯情形 $`\alpha_t=1`$ 即为 Tweedie 后验均值公式 〔[22](#ref-efron2011tweedie)〕。
一般协方差形式为
$`\alpha_t d_t^*(x)=x+\Sigma_t s_t(x)`$。
当 $`\alpha_t=0`$，观测完全来自独立噪声，Score 为 $`-x/\sigma_t^2`$，
等式两边相消而无法从该式辨识数据后验均值。

进一步令 $`Y=\alpha_tD`$、固定 $`t`$。对后验均值的商式求导，
后验密度导数为 $`(Y-\mathbb E[Y\mid x])/\sigma_t^2`$ 乘以后验密度。
因此 $`\nabla_x\mathbb E[Y\mid x]=\operatorname{Cov}(Y\mid x)/\sigma_t^2`$。
再对 Tweedie 等式求导，得到

<a name="eq-fields-second-tweedie"></a>

```math
\alpha_t^2\operatorname{Cov}(D\mid X_t=x)
   =\sigma_t^2I_d+\sigma_t^4\nabla_xs_t(x).
\tag{4.12}
```

该关系提供 Hessian 与后验不确定性的核对方式，需相应条件矩存在。

**例 4.3 · 非单位方差数据的去噪**

设 $`D\sim\mathcal N(m,C)`$、$`C`$ 正定，令
$`H_t=\alpha_t^2C+\sigma_t^2I_d`$。直接由高斯边缘计算
$`s_t(x)=-H_t^{-1}(x-\alpha_tm)`$。
Tweedie 公式给出

```math
d_t^*(x)=m+\alpha_t C H_t^{-1}(x-\alpha_tm),\qquad
 \varepsilon^*(x,t)=\sigma_t H_t^{-1}(x-\alpha_tm).
```

其中 $`I_d-\sigma_t^2H_t^{-1}=\alpha_t^2CH_t^{-1}`$。
若 $`m=0,C=4I_d,\alpha_t=\sigma_t=1/\sqrt2`$，
则 $`s_t=-2x/5`$、$`d_t^*=4\sqrt2x/5`$、$`\varepsilon^*=\sqrt2x/5`$。

<a name="sec-4-3-4"></a>

### 4.3.4 Score、Noise 与 Clean Data 之间的转换

固定同一路径并在 $`\alpha_t\sigma_t\ne0`$ 时，换算关系为

<a name="eq-fields-three-parameterizations"></a>

```math
\begin{aligned}
 \widehat d_t&=(x-\sigma_t\widehat\varepsilon_t)/\alpha_t,
 &\widehat s_t&=-\widehat\varepsilon_t/\sigma_t,\\
 \widehat\varepsilon_t&=(x-\alpha_t\widehat d_t)/\sigma_t,
 &\widehat d_t&=(x+\sigma_t^2\widehat s_t)/\alpha_t.
 \end{aligned}
\tag{4.13}
```

它们为同一个输出提供不同参数化。若三个网络分别训练，有限容量与优化误差
可以使它们不满足这些代数关系。
在同一输入上比较精确后验均值与一致换算的预测，误差满足
$`\delta d=-(\sigma_t/\alpha_t)\delta\varepsilon`$、
$`\delta s=-\delta\varepsilon/\sigma_t`$。所以

```math
\lVert \delta\varepsilon\rVert^2
 =\sigma_t^2\lVert \delta s\rVert^2
 =\frac{\alpha_t^2}{\sigma_t^2}\lVert \delta d\rVert^2.
```

转换损失时须同步转换时间权重。一般矩阵噪声下，
$`\delta s=-L_t^{-\top}\delta\varepsilon`$，欧氏噪声误差对应
Score 的 $`\Sigma_t`$ 加权二次型。

<a name="sec-4-3-5"></a>

### 4.3.5 DDPM 中的 $`v`$-prediction 与 Flow Velocity 的区别

对满足 $`\alpha_t^2+\sigma_t^2=1`$ 的路径，定义扩散监督标签
$`V_t^{\mathrm{diff}}=\alpha_t\varepsilon-\sigma_tD`$。
将两个线性方程合并，可得正交变换及其逆

<a name="eq-fields-vpred-rotation"></a>

```math
\begin{pmatrix}X_t\\V_t^{\mathrm{diff}}\end{pmatrix}
 =\begin{pmatrix}\alpha_t&\sigma_t\\-\sigma_t&\alpha_t\end{pmatrix}
   \begin{pmatrix}D\\\varepsilon\end{pmatrix},\qquad
 D=\alpha_tX_t-\sigma_t V_t^{\mathrm{diff}},\quad
 \varepsilon=\sigma_tX_t+\alpha_t V_t^{\mathrm{diff}}.
\tag{4.14}
```

该参数化见 〔[15](#ref-salimans2022distill)〕。若 $`w_t(x)=\mathbb E[V_t^{\mathrm{diff}}\mid X_t=x]`$，
则 $`d_t^*=\alpha_tx-\sigma_tw_t`$、
$`\varepsilon^*=\sigma_tx+\alpha_tw_t`$。
这些逆变换在单个系数为零时仍可计算。

若 $`\alpha_t=\cos\vartheta(t)`$、$`\sigma_t=\sin\vartheta(t)`$，
对固定的 $`(D,\varepsilon)`$ 求路径导数得
$`\dot X_t=\vartheta'(t)V_t^{\mathrm{diff}}`$。所以

```math
u_t(x)=\vartheta'(t)w_t(x),\qquad
 v_\tau(x)=-\vartheta'(1-\tau)w_{1-\tau}(x).
```

换算包含条件期望、时间尺度与时间方向。
对线性插值 $`X_t=(1-t)D+t\varepsilon`$，两系数的平方和通常随时间变化，
相应的路径导数直接为 $`\varepsilon-D`$，须按该路径推导速度目标。

<a name="sec-4-4"></a>

## 4.4 Velocity Field 的数学推导

<a name="sec-4-4-1"></a>

### 4.4.1 速度场的定义与连续性方程

对于确定性 ODE $`\dot X_t=u_t(X_t)`$，任取紧支撑光滑测试函数 $`\varphi`$，
链式法则给出
$`\frac{\mathrm{d} }{\mathrm{d}  t}\mathbb E\varphi(X_t)=\mathbb E[\nabla\varphi(X_t)^\top u_t(X_t)]`$。
若 $`X_t`$ 的密度为 $`p_t`$，分部积分得到

<a name="eq-fields-continuity"></a>

```math
\partial_t p_t+\nabla\cdot(p_tu_t)=0.
\tag{4.15}
```

因此 $`p_tu_t`$ 是概率流密度，$`u_t`$ 是单位概率质量的输运速度。
将方程展开为 $`\partial_t p_t+u_t\cdot\nabla p_t+p_t\nabla\cdot u_t=0`$，
沿轨迹得到 $`\frac{\mathrm{d} }{\mathrm{d}  t}\log p_t(X_t)=-\nabla\cdot u_t(X_t)`$。
从速度场计算生成分布，还要求 ODE 及连续性方程具有相应存在唯一性，
第 [3.4](chapter-3.md#sec-3-4) 节给出了经典正则情形。

<a name="sec-4-4-2"></a>

### 4.4.2 条件速度与边缘速度的关系

设标签 $`Z\sim\pi`$，每个条件密度满足
$`\partial_t p_t(x\mid z)+\nabla\cdot[p_t(x\mid z)u_t(x\mid z)]=0`$。
对标签积分并交换微分，得到

```math
\partial_t p_t(x)=-\nabla\cdot\int u_t(x\mid z)p_t(x\mid z)\pi(\mathrm{d}  z).
```

故自然的边缘速度为

<a name="eq-fields-conditional-velocity"></a>

```math
u_t(x)=\frac{\int u_t(x\mid z)p_t(x\mid z)\pi(\mathrm{d}  z)}{p_t(x)}
       =\mathbb E[u_t(X_t\mid Z)\mid X_t=x].
\tag{4.16}
```

此公式与条件 Score 恒等式均使用后验加权，但证明分别来自连续性方程与密度求导。
若选择不同条件路径或耦合，后验和条件速度会随之变化。
对于训练时可直接采样的 $`(X_t,Z)`$，平方回归可估计这个条件平均
〔[5](#ref-lipman2023fm)〕。

<a name="sec-4-4-3"></a>

### 4.4.3 高斯条件路径的速度目标 ★

先固定标签 $`z`$，构造 $`X_t=m_t(z)+\sigma_t\varepsilon`$，
$`\varepsilon`$ 沿每条条件轨迹保持固定，$`\sigma_t>0`$。
求导再将 $`\varepsilon=(X_t-m_t)/\sigma_t`$ 代入，得到

<a name="eq-fields-gaussian-conditional-velocity"></a>

```math
u_t(x\mid z)=\dot m_t(z)+\frac{\dot\sigma_t}{\sigma_t}(x-m_t(z)).
\tag{4.17}
```

它输运条件均值和条件协方差。事实上 $`\dot m_t`$ 给出均值变化，
$`2(\dot\sigma_t/\sigma_t)\sigma_t^2=\frac{\mathrm{d} }{\mathrm{d}  t}\sigma_t^2`$
给出协方差变化。取 $`m_t(z)=\alpha_tz`$，标签可写为
$`\dot\alpha_tD+\dot\sigma_t\varepsilon`$。
训练直接使用这两个抽样变量，就可避免在标签中显式除以小 $`\sigma_t`$。

更一般地，若 $`X_t=m_t(z)+L_t\varepsilon`$，$`L_t`$ 可逆，
则 $`u_t(x\mid z)=\dot m_t+\dot L_tL_t^{-1}(x-m_t)`$。
协方差 $`\Sigma_t=L_tL_t^\top`$ 满足
$`\dot\Sigma_t=B_t\Sigma_t+\Sigma_tB_t^\top`$，$`B_t=\dot L_tL_t^{-1}`$。
不同的平方根路径 $`L_t`$ 可以给出相同协方差和不同条件速度，
所以只知道 $`\Sigma_t`$ 的演化还需指定一个速度选择。

<a name="sec-4-4-4"></a>

### 4.4.4 Flow Matching 速度场的条件期望表示

取任意具有可积路径导数的随机插值 $`I_t`$。
只要可交换求导与期望，$`u_t(x)=\mathbb E[\dot I_t\mid I_t=x]`$
满足式 [(4.15)](#eq-fields-continuity) 的弱形式。
对 $`I_t=\alpha_tD+\sigma_tN`$ 及指定端点耦合 $`(D,N)`$，
有 $`u_t=\mathbb E[\dot\alpha_tD+\dot\sigma_tN\mid I_t=x]`$。
若 $`D,N`$ 相关，给定 $`D`$ 的 $`N`$ 分布取决于耦合，
必须从该条件律计算 Score；独立高斯噪声的公式需按其假设使用。

含辅助噪声的插值 $`I_t=\mathcal I(t,D,N)+\gamma(t)\zeta`$
满足 $`\dot I_t=\partial_t\mathcal I+\dot\gamma(t)\zeta`$，
在 $`\gamma(t)>0`$、独立标准高斯 $`\zeta`$ 下又有
$`s_t=-\mathbb E[\zeta\mid I_t=x]/\gamma(t)`$。因此

<a name="eq-fields-si-velocity-score"></a>

```math
u_t(x)=\mathbb E[\partial_t\mathcal I(t,D,N)\mid I_t=x]
                             -\gamma(t)\dot\gamma(t)s_t(x).
\tag{4.18}
```

它允许保留相关端点耦合，同时通过独立辅助高斯推导 Score
〔[8](#ref-albergo2025si)〕。

**例 4.4 · 相关端点对噪声公式的影响**

设 $`(D,N)`$ 为一维联合标准高斯，相关系数为 $`\rho\in(-1,1)`$，
$`X=\alpha D+\sigma N`$、$`\sigma>0`$，记
$`H=\alpha^2+\sigma^2+2\rho\alpha\sigma`$。
直接计算得到 $`s(x)=-x/H`$，而
$`\mathbb E[N\mid X=x]=(\sigma+\rho\alpha)x/H`$。
因而把该噪声条件均值直接除以 $`-\sigma`$ 会多出相关性系数项。
实际的条件核为
$`X\mid D\sim\mathcal N((\alpha+\sigma\rho)D,\sigma^2(1-\rho^2))`$，
其可回归的条件 Score 标签为

```math
-\frac{X-(\alpha+\sigma\rho)D}{\sigma^2(1-\rho^2)}
 =-\frac{N-\rho D}{\sigma(1-\rho^2)}.
```

对这个标签取后验平均即可恢复正确的 $`-x/H`$。

<a name="sec-4-4-5"></a>

### 4.4.5 速度场的非唯一性与规范选择

对给定 $`p_t`$，若两个速度 $`u_t,\widetilde u_t`$ 满足同一连续性方程，
其差 $`w_t=\widetilde u_t-u_t`$ 满足 $`\nabla\cdot(p_tw_t)=0`$。
这给出全部可行速度的一个仿射结构。
例如二维标准高斯静态路径的 $`u=0`$ 与 $`\widetilde u=Jx`$ 都可行，
其中 $`J^\top=-J`$。因为
$`\nabla\cdot(pJx)=p\operatorname{tr}J+(Jx)\cdot\nabla p=0`$。
第一种流固定样本，第二种流旋转样本。

给定 $`\partial_t p`$ 时，可按最小动能选择速度。
若 $`u=\nabla\phi`$ 解加权 Poisson 方程
$`-\nabla\cdot(p\nabla\phi)=\partial_t p`$，且边界条件允许积分，
对任意可行差场 $`w`$ 有
$`\int p\nabla\phi\cdot w=-\int\phi\nabla\cdot(pw)=0`$。
于是

```math
\int p\lVert u+w\rVert^2=\int p\lVert u\rVert^2+\int p\lVert w\rVert^2.
```

故该梯度速度在可行类中最小化动能。方程的解与存在性取决于所给路径和边界。
条件回归、概率流构造、最小动能分别指定速度的选择原则。

<a name="sec-4-5"></a>

## 4.5 Drift Field 与 Score 的关系

<a name="sec-4-5-1"></a>

### 4.5.1 SDE 中 Drift 的定义

考虑前向 Itô SDE
$`\mathrm{d}  X_t=f_t(X_t)\mathrm{d}  t+G_t(X_t)\mathrm{d}  W_t`$，令 $`a_t=G_tG_t^\top`$。
在局部正则条件下

```math
f_t(x)=\lim_{h\downarrow0}\frac1h\mathbb E[X_{t+h}-X_t\mid X_t=x],
 \qquad
 a_t(x)=\lim_{h\downarrow0}\frac1h
       \operatorname{Cov}(X_{t+h}-X_t\mid X_t=x).
```

Drift 描述短时条件均值增量，Diffusion 描述二阶增量。
Brownian 路径的逐点时间导数通常不存在，故随机轨迹的导数标签需通过具体插值
或短时回归定义。
若以 Stratonovich 形式给出漂移 $`\widetilde f`$，
转换到 Itô 形式需加上
$`\frac12\sum_{jk}G_{jk}\partial_jG_{ik}`$ 的第 $`i`$ 分量。
本章的密度方程与反演公式均使用 Itô 漂移。

<a name="sec-4-5-2"></a>

### 4.5.2 Drift、Diffusion 与概率流

Fokker–Planck 方程的守恒形式为 $`\partial_t p_t=-\nabla\cdot j_t`$，其中

<a name="eq-fields-probability-current"></a>

```math
(j_t)_i=f_{i,t}p_t-\tfrac12\sum_j\partial_j(a_{ij,t}p_t),\qquad
 u_t=\frac{j_t}{p_t}.
\tag{4.19}
```

所以同一个漂移配上不同扩散，会产生不同密度演化和概率流速度。
反过来，指定 $`p_t,u_t,a_t`$ 后可选择漂移，使此概率流恰好等于 $`p_tu_t`$。
给定一条概率路径，需要同时说明它是由哪一组漂移与扩散实现的。
当扩散为零，守恒形式直接给出 $`u=f`$。
正扩散下，密度梯度与扩散系数的空间变化共同参与概率流。

<a name="sec-4-5-3"></a>

### 4.5.3 各向同性时 Drift 与 Velocity 的转换 ★

若 $`G_t=g(t)I_d`$，扩散矩阵与空间无关，
$`j_t=f_tp_t-g(t)^2\nabla p_t/2`$。除以 $`p_t`$ 得

<a name="eq-fields-isotropic-conversion"></a>

```math
u_t=f_t-\frac{g(t)^2}{2}s_t,\qquad
 f_t=u_t+\frac{g(t)^2}{2}s_t.
\tag{4.20}
```

当 $`g(t)\ne0`$ 且 $`f_t,u_t`$ 已知，可恢复
$`s_t=2(f_t-u_t)/g(t)^2`$。若只知道 $`u_t`$，仍需给定 $`f_t`$ 或其他路径结构。
在 VP 扩散 $`f_t=-\beta(t)x/2,g^2=\beta(t)`$ 中，
$`u_t=-\beta(t)(x+s_t)/2`$。
若路径为标准高斯静态分布，$`s_t=-x`$，则概率流速度为零。
相应随机漂移仍为 $`-\beta x/2`$，其收缩与扩散共同维持平稳边缘。

<a name="sec-4-5-4"></a>

### 4.5.4 状态依赖各向异性扩散中的修正项 ★

逐坐标展开式 [(4.19)](#eq-fields-probability-current) 中的导数

```math
\sum_j\partial_j(a_{ij,t}p_t)
 =p_t\sum_j\partial_j a_{ij,t}+\sum_j a_{ij,t}\partial_jp_t.
```

定义 $`(\operatorname{div}a)_i=\sum_j\partial_ja_{ij}`$，便得到

<a name="eq-fields-anisotropic-conversion"></a>

```math
\begin{aligned}
 u_t&=f_t-\tfrac12\operatorname{div}a_t-\tfrac12a_ts_t,\\
 f_t&=u_t+\tfrac12\operatorname{div}a_t+\tfrac12a_ts_t.
 \end{aligned}
\tag{4.21}
```

这里 $`a_ts_t`$ 是矩阵乘向量。只有当 $`a_t`$ 可逆时，才可由两个漂移类场恢复
$`s_t=a_t^{-1}(2f_t-2u_t-\operatorname{div}a_t)`$。
若 $`a_t`$ 奇异，该等式确定 $`a_ts_t`$，其核空间内的 Score 分量需额外信息。
即使 $`a_t`$ 空间上常数，各向异性也要求保留整个矩阵。
状态依赖性进一步引入 $`\operatorname{div}a_t`$。

**例 4.5 · 一维乘性扩散中的稳态流**

取 $`p(x)\propto e^{-x^2/2}`$、$`a(x)=1+x^2`$，令概率流速度为零。
由式 [(4.21)](#eq-fields-anisotropic-conversion)，所需 Itô 漂移为

```math
f(x)=\tfrac12a'(x)+\tfrac12a(x)s(x)=\tfrac12(x-x^3).
```

直接计算 $`f p-\frac12(ap)'=0`$，即可核对其静态密度方程。
该漂移超线性，其 SDE 适定性可由局部 Lipschitz 与耗散 Lyapunov 控制建立。
若遗漏 $`a'/2`$，残余概率流为 $`-xp`$，静态方程随之改变。

<a name="sec-4-5-5"></a>

### 4.5.5 反向 SDE 中的 Score 修正

在第 [3.3](chapter-3.md#sec-3-3) 节的时间反演条件下，递增生成时间的漂移为

<a name="eq-fields-reverse-conversion"></a>

```math
b_\tau=-f_t+\operatorname{div}a_t+a_ts_t,\qquad
 v_\tau=-f_t+\tfrac12\operatorname{div}a_t+\tfrac12a_ts_t,
 \quad t=1-\tau.
\tag{4.22}
```

因此 $`b_\tau=v_\tau+\operatorname{div}a_t/2+a_ts_t/2`$。
反向 SDE 与生成 ODE 在精确条件下共享边缘，但二者的漂移与速度各有自己的系数。
例如各向同性 VP 中，$`b_\tau=\beta x/2+\beta s_t`$，
$`v_\tau=\beta x/2+\beta s_t/2`$。
对同一 Score 估计误差 $`e_s`$，反向漂移误差为 $`a_te_s`$，
生成概率流速度误差为 $`a_te_s/2`$。误差的后续放大还取决于动力学稳定性。

<a name="sec-4-6"></a>

## 4.6 Flow Matching 与 Score Matching 的联系

<a name="sec-4-6-1"></a>

### 4.6.1 高斯插值路径的 Score

本节固定 $`X_t=\alpha_tD+\sigma_t\varepsilon`$，其中 $`D`$ 与标准高斯
$`\varepsilon`$ 独立，$`\alpha_t,\sigma_t`$ 为可微的标量系数。
在 $`\sigma_t>0`$ 时，高斯卷积使内部边缘具有密度，且
$`s_t=-\mathbb E[\varepsilon\mid X_t=x]/\sigma_t`$。
若数据 $`D\sim\mathcal N(m,C)`$，边缘密度为
$`\mathcal N(\alpha_tm,H_t)`$、$`H_t=\alpha_t^2C+\sigma_t^2I_d`$，
所以 $`s_t=-H_t^{-1}(x-\alpha_tm)`$。
数据为混合高斯时，各分量均值和协方差分别变成
$`\alpha_tm_j`$ 与 $`\alpha_t^2C_j+\sigma_t^2I_d`$，
再按新的后验责任权重平均分量 Score。

<a name="sec-4-6-2"></a>

### 4.6.2 高斯插值路径的 Velocity

沿固定 $`(D,\varepsilon)`$ 的路径有
$`\dot X_t=\dot\alpha_tD+\dot\sigma_t\varepsilon`$。
因此边缘速度为

<a name="eq-fields-gaussian-marginal-velocity"></a>

```math
u_t(x)=\dot\alpha_t d_t^*(x)+\dot\sigma_t\varepsilon_t^*(x).
\tag{4.23}
```

这一定义对应固定噪声的可微随机插值。
若数据为高斯，代入上一节的两个条件均值可得

```math
u_t(x)=\dot\alpha_tm+
   (\alpha_t\dot\alpha_t C+\sigma_t\dot\sigma_tI_d)
                        H_t^{-1}(x-\alpha_tm).
```

其线性系数为 $`\dot H_t H_t^{-1}/2`$，并且
$`B_tH_t+H_tB_t^\top=\dot H_t`$。
这从协方差演化的角度核对了回归场。

<a name="sec-4-6-3"></a>

### 4.6.3 在非退化条件下由 Score 表示 Velocity ★

当 $`\alpha_t\ne0`$ 时，将 Tweedie 与噪声恒等式代入
式 [(4.23)](#eq-fields-gaussian-marginal-velocity)，逐项整理得到

<a name="eq-fields-score-to-velocity"></a>

```math
\begin{aligned}
 u_t(x)&=\frac{\dot\alpha_t}{\alpha_t}x+C_t s_t(x),\\
 C_t&=\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2
                                     -\sigma_t\dot\sigma_t.
 \end{aligned}
\tag{4.24}
```

第一项来自 $`\dot\alpha_tx/\alpha_t`$，第二项收集条件噪声与条件数据均值中的
Score 系数。对于已知的固定路径，这就是 Score 预测转成 FM 速度预测的公式。
生成速度还需在 $`t=1-\tau`$ 处取负。

对 VP 系数，$`C_t=-\beta(t)/2`$；对加性噪声路径
$`\alpha_t=1`$，有 $`C_t=-\sigma_t\dot\sigma_t`$。
在线性路径 $`\alpha_t=1-t,\sigma_t=t`$ 的内部，

```math
u_t(x)=-\frac{x+t s_t(x)}{1-t},\qquad
 s_t(x)=-\frac{x+(1-t)u_t(x)}t.
```

若使用相关的高斯端点耦合或其他噪声分布，需从其实际条件后验重做这一推导。

对矩阵噪声路径 $`X_t=\alpha_tD+L_t\varepsilon`$，假设 $`L_t`$ 可逆，
记 $`\Sigma_t=L_tL_t^\top`$。此时
$`\mathbb E[\varepsilon\mid X_t=x]=-L_t^\top s_t(x)`$，
将其代入路径导数可得

```math
u_t(x)=\frac{\dot\alpha_t}{\alpha_t}x+\mathcal C_t s_t(x),\qquad
 \mathcal C_t=\frac{\dot\alpha_t}{\alpha_t}\Sigma_t-\dot L_tL_t^\top.
```

矩阵 $`\dot L_tL_t^\top`$ 的次序由路径导数确定。
其对称部分为 $`\dot\Sigma_t/2`$，反对称部分还包含所选平方根路径的信息。
因此在矩阵路径中，速度转换需要保留具体的 $`L_t`$ 演化。

<a name="sec-4-6-4"></a>

### 4.6.4 由 Velocity 恢复 Score 的条件

若 $`C_t\ne0`$，式 [(4.24)](#eq-fields-score-to-velocity) 可反解为
$`s_t=(u_t-\dot\alpha_tx/\alpha_t)/C_t`$。
速度估计误差 $`e_u`$ 因而变为 Score 误差 $`e_s=e_u/C_t`$。
即使 $`C_t`$ 非零，过小的绝对值也会导致数值条件变差。
矩阵噪声情形对应求解 $`\mathcal C_ts_t=u_t-(\dot\alpha_t/\alpha_t)x`$。
可逆时误差满足 $`\lVert e_s\rVert\le\lVert \mathcal C_t^{-1}\rVert\lVert e_u\rVert`$，
最小奇异值很小会放大误差；奇异时只由该方程确定 $`\mathcal C_ts_t`$。
为避免先除以 $`\alpha_t`$，还可直接求解两条线性条件均值方程

<a name="eq-fields-path-determinant"></a>

```math
\begin{pmatrix}x\\u_t(x)\end{pmatrix}
 =\begin{pmatrix}\alpha_t&\sigma_t\\\dot\alpha_t&\dot\sigma_t\end{pmatrix}
   \begin{pmatrix}d_t^*(x)\\\varepsilon_t^*(x)\end{pmatrix},\qquad
 \Delta_t=\alpha_t\dot\sigma_t-\sigma_t\dot\alpha_t.
\tag{4.25}
```

当 $`\Delta_t\ne0`$，有

<a name="eq-fields-velocity-inverse"></a>

```math
d_t^*=\frac{\dot\sigma_t x-\sigma_tu_t}{\Delta_t},\qquad
 \varepsilon_t^*=\frac{-\dot\alpha_t x+\alpha_tu_t}{\Delta_t},\qquad
 s_t=\frac{\dot\alpha_t x-\alpha_tu_t}{\sigma_t\Delta_t}.
\tag{4.26}
```

Score 式另需 $`\sigma_t>0`$。当 $`\alpha_t\ne0`$ 时，
$`C_t=-\sigma_t\Delta_t/\alpha_t`$，两个判别相容。
例如线性路径 $`\Delta_t=1`$，数据端恢复 Score 仍涉及除以 $`t`$。

**例 4.6 · 固定信噪比造成的退化**

取 $`\alpha_t=c_t`$、$`\sigma_t=\lambda c_t`$，其中 $`\lambda>0,c_t>0`$。
此时 $`X_t=c_t(D+\lambda\varepsilon)`$，故
$`u_t=(\dot c_t/c_t)x`$，$`\Delta_t=C_t=0`$。
这个速度场只表达统一缩放，适用于多种不同的数据密度。
所以在此路径族中，单凭该速度无法恢复具体的边缘 Score。

<a name="sec-4-6-5"></a>

### 4.6.5 Flow Matching 与 Diffusion 共享概率路径的构造

令 $`f_t(x)=a_t^{(0)}x`$、扩散为 $`g(t)I_d`$，其中
$`a_t^{(0)}=\dot\alpha_t/\alpha_t`$，$`\alpha_t>0`$。
要使线性 SDE 的条件边缘等于
$`\mathcal N(\alpha_tD,\sigma_t^2I_d)`$，方差方程要求

<a name="eq-fields-gaussian-diffusion-realization"></a>

```math
g(t)^2=\frac{\mathrm{d} }{\mathrm{d}  t}\sigma_t^2
             -2\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2=-2C_t\ge0.
\tag{4.27}
```

结合 $`\alpha_0=1,\sigma_0=0`$，均值与方差方程即可验证全部条件边缘。
该 SDE 的概率流速度为 $`a_t^{(0)}x-g^2s_t/2`$，
恰等于式 [(4.24)](#eq-fields-score-to-velocity)。
若 $`-2C_t<0`$，该指定线性漂移下无法使用实值扩散强度实现所需边缘。

更一般地，给定 $`p_t,u_t,s_t`$，任取合适的半正定扩散协方差 $`A_t(x)`$，
使用 $`f_t=u_t+\operatorname{div}A_t/2+A_ts_t/2`$
也可在适定条件下实现同一边缘。这时漂移通常依赖 Score。
因此同边缘的 SDE 构造可以很广，指定线性前向漂移会进一步约束可选路径。

<a name="sec-4-7"></a>

## 4.7 Schrödinger Bridge 中的 Score 与漂移

<a name="sec-4-7-1"></a>

### 4.7.1 Schrödinger Potentials 的对数梯度

设 Markov 参考过程 $`R`$ 的漂移、扩散协方差和边缘密度分别为
$`f_t,a_t,r_t`$。桥路径律满足
$`\mathrm{d}  P^\star/\mathrm{d}  R=\ell(X_0)h(X_1)`$，定义

```math
\psi_t(x)=\mathbb E_R[h(X_1)\mid X_t=x],\qquad
 \widehat\psi_t(x)=\mathbb E_R[\ell(X_0)\mid X_t=x].
```

在所需正则条件下，Markov 性使过去和未来在给定 $`X_t`$ 后条件独立，因而

<a name="eq-fields-bridge-score"></a>

```math
p_t^\star=r_t\widehat\psi_t\psi_t,\qquad
 s_t^\star=\nabla\log r_t+\nabla\log\widehat\psi_t+\nabla\log\psi_t.
\tag{4.28}
```

这里 $`\widehat\psi_t`$ 采用参考条件期望的定义。
若把 $`r_t\widehat\psi_t`$ 合并定义为另一个势，Score 的分解也需同步改写。
记 $`\eta_t=\nabla\log\psi_t`$、$`\widehat\eta_t=\nabla\log\widehat\psi_t`$，
二者分别来自终点与起点的重加权。
势可按常数互逆缩放而保持乘积和对数梯度不变
〔[13](#ref-leonard2014schrodinger)〕。

<a name="sec-4-7-2"></a>

### 4.7.2 最优控制 Drift 的表示

第 [3.6](chapter-3.md#sec-3-6) 节的 Doob 变换给出

<a name="eq-fields-bridge-drifts"></a>

```math
f_t^\star=f_t+a_t\eta_t,\qquad
 b_\tau^\star=-f_t+\operatorname{div}a_t
                 +a_t\nabla\log r_t+a_t\widehat\eta_t,
 \quad t=1-\tau.
\tag{4.29}
```

前向修正使用终点势梯度，反向参考漂移上的修正使用起点势梯度。
若 $`a=GG^\top`$，前向噪声坐标控制为 $`G^\top\eta_t`$，
实际空间漂移修正是 $`a\eta_t`$。
对可逆 $`a`$，该控制的瞬时二次能量为
$`\eta_t^\top a_t\eta_t/2`$。
实际 DSB 方法也可直接回归前向和反向漂移
〔[14](#ref-debortoli2021dsb)〕，此时需核对网络所用扩散尺度与时间坐标。

<a name="sec-4-7-3"></a>

### 4.7.3 正向与反向漂移的关系

把所有场放在同一空间位置，且令 $`t=1-\tau`$，时间反演给出

<a name="eq-fields-bridge-sum-difference"></a>

```math
\begin{aligned}
 f_t^\star+b_\tau^\star&=\operatorname{div}a_t+a_ts_t^\star,\\
 u_t^\star&=\tfrac12(f_t^\star-b_\tau^\star),\qquad
 v_\tau^\star=\tfrac12(b_\tau^\star-f_t^\star).
 \end{aligned}
\tag{4.30}
```

第一行把漂移之和联系到 Score，第二行给出概率流速度。
若 $`a_t`$ 可逆，$`s_t^\star=a_t^{-1}(f_t^\star+b_\tau^\star-\operatorname{div}a_t)`$。
这里 $`b_\tau^\star`$ 按递增生成时间定义。
如果某篇文献将反向漂移记为原坐标 $`t`$ 下的后向导数，
它与本书 $`b`$ 的符号需要先转换，再比较和差公式。

<a name="sec-4-7-4"></a>

### 4.7.4 桥过程 Score 与边缘密度

式 [(4.28)](#eq-fields-bridge-score) 说明，只掌握某一方向的控制梯度时，
仍需其他势或边缘信息才能恢复桥 Score。
给定两个精确漂移则可用式 [(4.30)](#eq-fields-bridge-sum-difference) 交叉核对。
桥的概率流 ODE 与随机桥共享边缘，具体生成算法见第 3 章。

**例 4.7 · 高斯参考上的指数终点权重**

取一维参考过程 $`\mathrm{d}  X_t=\sqrt\epsilon\mathrm{d}  W_t`$、$`X_0\sim\mathcal N(0,c)`$，
其中 $`c,\epsilon>0`$，记 $`H_t=c+\epsilon t`$。
选 $`h(x)=e^{\lambda x}`$、$`\ell=e^{-\lambda^2H_1/2}`$，使路径重加权归一化。
条件高斯给出

```math
\psi_t(x)=e^{\lambda x+\lambda^2\epsilon(1-t)/2},\qquad
 \widehat\psi_t=\ell,\qquad
 p_t^\star=\mathcal N(\lambda H_t,H_t).
```

因此 $`\eta_t=\lambda,\widehat\eta_t=0`$、
$`s_t^\star=-x/H_t+\lambda`$。两向漂移和概率流速度为

```math
f_t^\star=\epsilon\lambda,\qquad
 b_\tau^\star=-\epsilon x/H_t,\qquad
 u_t^\star=\epsilon\lambda/2+\epsilon x/(2H_t).
```

它们逐项满足式 [(4.30)](#eq-fields-bridge-sum-difference)。
对 $`p_t^\star`$ 的均值、方差求导分别为 $`\epsilon\lambda,\epsilon`$，
与该 ODE 及受控 SDE 的矩方程一致。
这组端点由所选权重确定，给出可直接核对的桥解。

<a name="sec-4-7-5"></a>

### 4.7.5 Score 估计误差对桥接过程的影响

若由两个势网络构造 Score，参考 Score 已知，则
$`e_s=e_\eta+e_{\widehat\eta}`$。若正向漂移网络固定，
通过时间反演公式构造反向漂移时，Score 误差产生 $`a_te_s`$ 的漂移误差。
若直接用反向参考漂移加起点势构造，则其误差为 $`a_te_{\widehat\eta}`$。
两种参数化各自需要检查其实际误差来源。
当三个网络分别学习 $`f_\theta,b_\phi,s_\omega`$，可以计算一致性残差

```math
R_t(x)=f_\theta(x,t)+b_\phi(x,1-t)
                         -\operatorname{div}a_t(x)-a_t(x)s_\omega(x,t).
```

该残差为零是它们与同一扩散边缘相容的必要关系，
完整桥约束还包含密度方程、双端边缘以及相对于参考律的路径最优性。
在同扩散、同初始律及 Girsanov 条件下，令 $`e_f=\widehat f-f^\star`$，有

```math
D_{\mathrm{KL}}(P^\star\|\widehat P)
 =\tfrac12\mathbb E_{P^\star}\int_0^1
            \lVert a_t^{-1/2}e_f(X_t,t)\rVert^2\mathrm{d}  t.
```

终点 KL 再由数据处理不等式控制。
小特征值方向会改变误差的能量权重。

<a name="sec-4-8"></a>

## 4.8 Score 及向量场的计算实现

<a name="sec-4-8-1"></a>

### 4.8.1 已知解析密度时的自动微分

已知可微对数密度 $`\ell(x)=\log p(x)`$ 或未归一化的 $`\log\widetilde p(x)`$，
对输入求梯度即可得到 Score。若密度很小，直接计算 $`\nabla p/p`$
可能出现下溢，使用对数表达式更稳定。
高斯模型使用 Cholesky 分解和线性求解，混合模型使用 log-sum-exp。
固定时间 $`t`$ 时，输入梯度将 $`t`$ 视为常数。

计算流程可写成如下伪代码。单个样本的空间维数为 $`d`$，batch 维独立处理。

```text
x = input_with_gradients()
logp = log_density(x, fixed_time)
score = gradient(sum(logp), x, keep_graph=True)
probe = independent_zero_mean_unit_covariance_vector()
jacobian_transpose_probe = gradient(dot(score, probe), x)
trace_sample = dot(jacobian_transpose_probe, probe)
```

这里将样本对数密度求和，要求每个样本的输出只依赖自身输入。
若模型计算跨样本耦合，梯度会包含跨样本项，应按实际 Jacobian 定义处理。
若用 Score Matching 训练能量网络，还要保留梯度图，以便对输入导数继续求参数梯度。

<a name="sec-4-8-2"></a>

### 4.8.2 未知密度时的神经网络估计

可直接以向量网络 $`s_\theta(x,t)`$ 拟合条件 Score，
或以噪声网络、去噪网络、标量能量网络参数化所需场。
选定参数化后，记录其训练标签、时间权重、数据尺度与用于采样的转换式。
例如噪声输出转 Score 要除以 $`\sigma_t`$，
生成时间速度转加噪时间速度要使用 $`-v_{1-t}`$。
若通过标量势构造 Score，Jacobian 理论上是对称 Hessian；
若直接回归向量场，总体最优值具有目标梯度结构，有限训练输出仍需检验。

训练时每个样本可只采一个时间和一个扰动，不必存储整条路径。
但采样器会在训练边缘以外的状态上调用网络，
分布外误差和轨迹稳定性需要通过生成端验证。
端点排除、噪声下界或输出预条件化都会改变实际数值问题，应与理论目标一同记录。

<a name="sec-4-8-3"></a>

### 4.8.3 条件样本回归与 Monte Carlo 估计

设 $`T`$ 是可采样标签，$`m(x)=\mathbb E[T\mid X=x]`$。
平方回归在联合样本 $`(X,T)`$ 上训练，利用总体条件投影间接估计 $`m`$。
连续变量中，直接等待重复出现完全相同的 $`x`$ 不能提供有效的局部估计。
若条件密度 $`q(x\mid z)`$ 可计算，可从 $`z_i\sim\pi`$ 构造自归一化估计

<a name="eq-fields-monte-carlo"></a>

```math
\widehat m_M(x)=\frac{\sum_{i=1}^Mq(x\mid z_i)T(x,z_i)}
                         {\sum_{i=1}^Mq(x\mid z_i)}.
\tag{4.31}
```

在适当大数定律和分母正的条件下，它收敛到后验平均。
有限样本比值通常有偏，其方差与权重集中程度有关。
令 $`\bar w_i=w_i/\sum_jw_j`$，有效样本量
$`M_{\mathrm{eff}}=1/\sum_i\bar w_i^2`$ 可辅助检查权重退化。
当 $`x`$ 处于先验抽样很少覆盖的区域，权重会集中，高维下尤其需要关注。
改变提议分布时，要将相应重要性比值一并纳入权重。

对高斯扰动，可取 $`T(x,D)=-(x-\alpha_tD)/\sigma_t^2`$
估计边缘 Score；也可取条件速度式 [(4.17)](#eq-fields-gaussian-conditional-velocity)
估计边缘速度。两者的权重来自同一实际条件密度。

<a name="sec-4-8-4"></a>

### 4.8.4 向量场的 Jacobian、散度与 Hessian

对 $`F:\mathbb R^d\to\mathbb R^d`$，采用
$`(J_F)_{ij}=\partial_jF_i`$，散度为 $`\operatorname{tr}J_F`$。
对标量 $`\ell`$，Score 为 $`\nabla\ell`$，其 Jacobian 为 Hessian
$`\nabla^2\ell`$，散度为 Laplacian $`\Delta\ell`$。
完整 Jacobian 包含 $`d^2`$ 个元素，很多算法只需
$`J_Fz`$、$`J_F^\top z`$ 或 $`z^\top J_Fz`$。
固定探针时，$`\nabla_x(z^\top F)=J_F^\top z`$，
再与 $`z`$ 点乘得到所需二次型。
这些恒等式允许用自动微分计算方向导数，并控制存储成本。

状态依赖扩散中的 $`\operatorname{div}a`$ 是一个向量，
其第 $`i`$ 分量是第 $`i`$ 行的散度。
它与标量 $`\operatorname{tr}a`$、向量场散度 $`\nabla\cdot F`$
在类型和求导指标上均需逐项区分。
例如 $`a(x)=\operatorname{diag}(a_1(x),\ldots,a_d(x))`$，
有 $`(\operatorname{div}a)_i=\partial_i a_i`$。

<a name="sec-4-8-5"></a>

### 4.8.5 高维散度估计与 Hutchinson 方法

取与当前 $`x`$ 独立的探针 $`z`$，且 $`\mathbb E[zz^\top]=I_d`$。
由迹的循环性质可得

<a name="eq-fields-hutchinson"></a>

```math
\mathbb E_z[z^\top J_F(x)z]
 =\operatorname{tr}(J_F(x)\mathbb E[zz^\top])
 =\nabla\cdot F(x).
\tag{4.32}
```

平均 $`M`$ 个独立探针可降低方差。
标准高斯探针与独立 Rademacher 探针 $`z_i\in\{-1,1\}`$ 都满足单位协方差。
令 $`S=(J_F+J_F^\top)/2`$，因为反对称部分的二次型为零，
单探针方差分别为

```math
\operatorname{Var}_{\mathrm G}(z^\top J_Fz)=2\lVert S\rVert_{\mathrm F}^2,
 \qquad
 \operatorname{Var}_{\mathrm R}(z^\top J_Fz)=2\sum_{i\ne j}S_{ij}^2.
```

独立探针平均的方差除以 $`M`$。这些公式由探针的二阶与四阶矩展开得到。
对对角 Jacobian，Rademacher 单探针已精确给出迹。
对 $`F(x)=Ax`$，任意反对称矩阵部分都不影响二次型。
随机迹在 CNF 密度计算中的应用见 〔[16](#ref-grathwohl2019ffjord)〕。

若用于 ODE 的对数密度积分，可以在一次轨迹求解中固定探针，
使被积函数在状态与时间上保持平滑。
固定的确定性轨迹上，随机积分估计由 Fubini 给出无偏性；
有限精度自适应求解的轨迹与网格可能依赖探针，完整数值误差还需分析。
若用于优化中的散度项，探针本身视为固定随机输入，并保留网络导数的参数梯度。

<a name="sec-4-8-6"></a>

### 4.8.6 Score 与 Velocity 的数值验证

解析模型可同时比较直接密度导数、条件回归表达和动力学残差。
对 $`p_t=\mathcal N(m_t,H_t)`$，先由密度求 $`s_t`$，
再由条件均值求 $`u_t`$，核对均值与协方差的演化。
对任意可计算密度，可构造归一化连续性残差

<a name="eq-fields-continuity-residual"></a>

```math
\mathcal R_t(x)=\partial_t\log p_t(x)+\nabla\cdot u_t(x)+u_t(x)^\top s_t(x).
\tag{4.33}
```

若场与密度相容，该残差为零。对给定 SDE，检查
$`\partial_t p+\nabla\cdot(fp)-\frac12\sum_{ij}\partial_{ij}(a_{ij}p)`$
可独立核对扩散修正。
直接计算所有解析项适合低维验证，高维则可用测试函数的弱形式。

对未知密度样本，选择有边界衰减的测试函数 $`h`$，
估计式 [(4.4)](#eq-fields-stein) 的样本均值并与 Monte Carlo 标准误比较。
对速度选择 $`\varphi`$，比较
$`\frac{\mathrm{d} }{\mathrm{d}  t}\mathbb E\varphi(X_t)`$
与 $`\mathbb E[\nabla\varphi(X_t)^\top u_t(X_t)]`$。
有限个测试函数只能检查相应方向，还需结合独立样本、多个时刻及生成分布检查。
闭环代数换算可发现系数和符号错误，但若所有式子共享同一错误，
仍需密度残差或独立解析算例来定位。

**例 4.8 · 线性高斯路径的闭环核对**

设 $`D\sim\mathcal N(0,cI_d)`$、$`\varepsilon\sim\mathcal N(0,I_d)`$ 独立，
$`X_t=(1-t)D+t\varepsilon`$，令 $`H_t=c(1-t)^2+t^2`$。则

```math
s_t(x)=-x/H_t,\qquad
 u_t(x)=\frac{t-c(1-t)}{H_t}x.
```

由 $`H_t'=2[t-c(1-t)]`$，有 $`u_t=H_t'x/(2H_t)`$。
直接求得
$`\partial_t\log p_t=-dH_t'/(2H_t)+H_t'\lVert x\rVert^2/(2H_t^2)`$，
加上 $`\nabla\cdot u_t=dH_t'/(2H_t)`$ 和
$`u_t^\top s_t=-H_t'\lVert x\rVert^2/(2H_t^2)`$，各项恰好抵消。
再用式 [(4.26)](#eq-fields-velocity-inverse) 可恢复
$`d_t^*=c(1-t)x/H_t`$、$`\varepsilon_t^*=tx/H_t`$。
这同时核对了密度导数、速度、Score 与两种后验预测。

### 主要转换公式速查

以下各行使用同一时刻和同一概率路径。高斯预测相关行假设
$`X_t=\alpha_tD+\sigma_t\varepsilon`$，且噪声与数据独立。
一般扩散行选用其概率流速度，$`a_t`$ 为扩散协方差。

| 已知量 | 目标量 | 转换与条件 |
| --- | --- | --- |
| 噪声均值 | Score | $`s=-\varepsilon^*/\sigma`$<br> $`\sigma>0`$ |
| Score | 干净均值 | $`d^*=(x+\sigma^2s)/\alpha`$<br> $`\alpha\ne0`$ |
| Score | 路径速度 | $`u=(\dot\alpha/\alpha)x+Cs`$<br> $`\alpha\ne0`$ |
| 路径速度 | Score | $`s=(\dot\alpha x-\alpha u)/(\sigma\Delta)`$<br> $`\sigma\Delta\ne0`$ |
| 漂移与 Score | 概率流速度 | $`u=f-\operatorname{div}a/2-as/2`$ |
| 前向漂移与 Score | 反向漂移 | $`b=-f+\operatorname{div}a+as`$<br> $`t=1-\tau`$ |
| 加噪速度 | 生成速度 | $`v_\tau=-u_{1-\tau}`$ |

其中 $`C=(\dot\alpha/\alpha)\sigma^2-\sigma\dot\sigma`$，
$`\Delta=\alpha\dot\sigma-\sigma\dot\alpha`$。
实际使用时先检查路径、时间、输出含义与分母，再进行数值换算。

**本章小结**

- 条件 Score 与条件速度均按实际后验平均，前者来自密度求导，后者来自概率守恒。
- 高斯扰动下噪声与 Score 通过 $`-1/\sigma_t`$ 联系，Tweedie 公式恢复干净数据后验均值。
- 指定路径后可推导速度转换，反解的可用性由噪声系数和路径系数矩阵的非退化性决定。
- 一般扩散中概率流速度含 $`\operatorname{div}a_t/2`$ 与 $`a_ts_t/2`$ 两项修正。
- 桥的 Score 分解包含参考边缘及两个势梯度，双向漂移之和与之相容。
- 实现时核对输出类型、数据尺度、时间方向、微分指标与 Monte Carlo 误差。

### 本章练习与核对

1. 对 $`X\sim\mathcal N(m,\Sigma)`$，计算 $`\mathbb E[s(X)(X-m)^\top]`$。<br>
 **核对**  $`-I_d`$，可同时由高斯 Score 和 Stein 恒等式得到。
1. 对两分量混合算例，计算 $`s'(0)`$，判断 $`m^2=c`$ 两侧的驻点类型。<br>
 **核对**  $`(m^2-c)/c^2`$，正值给出局部低谷，负值给出局部峰。
1. 设 $`X=D+\sigma\varepsilon`$、$`D\sim\mathcal N(0,c)`$，验证二阶 Tweedie 公式。<br>
 **核对**  $`\operatorname{Var}(D\mid X)=c\sigma^2/(c+\sigma^2)`$。
1. 取 $`\alpha=\cos\vartheta,\sigma=\sin\vartheta`$，
 求 $`\Delta`$ 并将扩散标签转换为加噪边缘速度。<br>
 **核对**  $`\Delta=\vartheta'`$，$`u=\vartheta'\mathbb E[V^{\mathrm{diff}}\mid X]`$。
1. 取 $`\alpha=1-t,\sigma=t`$，求边缘速度、反求 Score 及相应线性扩散强度。<br>
 **核对**  $`u=-(x+ts)/(1-t)`$，$`s=-[x+(1-t)u]/t`$，
 $`g^2=2t/(1-t)`$，各式按内部时刻使用。
1. 对状态依赖扩散算例，验证 $`f=(x-x^3)/2`$ 时静态概率流为零。<br>
 **核对**  $`(ap)'=(x-x^3)p=2fp`$。
1. 给定同一扩散的两个精确桥漂移 $`f_t^\star,b_{1-t}^\star`$，
 在 $`a_t`$ 可逆时恢复 Score 和概率流速度。<br>
 **核对**  $`s=a^{-1}(f^\star+b^\star-\operatorname{div}a)`$，
 $`u=(f^\star-b^\star)/2`$。
1. 对 $`A=\begin{pmatrix}2&1\\3&4\end{pmatrix}`$，
 计算高斯与 Rademacher 单探针迹估计的方差。<br>
 **核对**  对称部分为 $`\begin{pmatrix}2&2\\2&4\end{pmatrix}`$，
 两个方差分别为 $`56,16`$。
1. 对单位球面均匀方向 $`z`$，计算 $`\mathbb E[zz^\top]`$，并调整到单位协方差。<br>
 **核对**  $`I_d/d`$，将方向乘 $`\sqrt d`$。
1. 对最后一个高斯路径算例，使用 $`c=4,t=1/2`$ 计算三个条件场。<br>
 **核对**  $`s=-4x/5`$，$`u=-6x/5`$，$`d^*=8x/5`$。

<a name="references"></a>

## 参考文献

<a name="ref-lipman2023fm"></a>
**[5]** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching for Generative Modeling. ICLR, 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).

<a name="ref-albergo2025si"></a>
**[8]** Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic Interpolants: A Unifying Framework for Flows and Diffusions. Journal of Machine Learning Research, 26(209), 2025. [论文](https://jmlr.org/papers/v26/23-1605.html).

<a name="ref-leonard2014schrodinger"></a>
**[13]** Christian Léonard. A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport. Discrete and Continuous Dynamical Systems, 34(4), 1533–1574, 2014. [arXiv:1308.0215](https://arxiv.org/abs/1308.0215).

<a name="ref-debortoli2021dsb"></a>
**[14]** Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling. NeurIPS, 2021. [论文](https://proceedings.neurips.cc/paper/2021/hash/940392f5f32a7ade1cc201767cf83e31-Abstract.html).

<a name="ref-salimans2022distill"></a>
**[15]** Tim Salimans and Jonathan Ho. Progressive Distillation for Fast Sampling of Diffusion Models. ICLR, 2022. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512).

<a name="ref-grathwohl2019ffjord"></a>
**[16]** Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. ICLR, 2019. [arXiv:1810.01367](https://arxiv.org/abs/1810.01367).

<a name="ref-hyvarinen2005sm"></a>
**[17]** Aapo Hyvärinen. Estimation of Non-Normalized Statistical Models by Score Matching. Journal of Machine Learning Research, 6(24), 695–709, 2005. [论文](https://www.jmlr.org/papers/v6/hyvarinen05a.html).

<a name="ref-vincent2011dsm"></a>
**[18]** Pascal Vincent. A Connection Between Score Matching and Denoising Autoencoders. Neural Computation, 23(7), 1661–1674, 2011. [DOI](https://doi.org/10.1162/NECO_a_00142).

<a name="ref-song2019sliced"></a>
**[21]** Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced Score Matching: A Scalable Approach to Density and Score Estimation. UAI 2019, PMLR 115, 574–584, 2020. [论文](https://proceedings.mlr.press/v115/song20a.html).

<a name="ref-efron2011tweedie"></a>
**[22]** Bradley Efron. Tweedie's Formula and Selection Bias. Journal of the American Statistical Association, 106(496), 1602–1614, 2011. [DOI](https://doi.org/10.1198/jasa.2011.tm11181).
