# Diffusion Model

## 一、前向加噪过程

- 前向传播的过程其实就是不断加噪音的过程，最后变成一个纯噪音
- 每个时刻都要添加高斯噪音，后一个时刻都是由前一个时刻添加噪音得到
- 这个过程可以看成不断构建标签的过程

### 如何得到Xt时刻的分布

$\alpha_t=1-\beta_t$，其中$\beta$要越来越大，论文中0.0001到0.002，从而$\alpha$就越来越小
$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}z_1
$$
一开始加点噪就有效果，越往后面加噪越多才行

现在，知道了后一个时刻的分布是由前一个时刻的分布加噪得到，但是整个序列的分布如果一个一个算效率太低

### Xt由X0直接算到

$$
x_{t-1}=\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}z_2
$$

带入上式得
$$
x_t=\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}z_2)+\sqrt{1-\alpha_t}z_1
\\=\sqrt{a_ta_{t-1}}x_{t-2}+(\sqrt{a_t(1-a_t)}z_2+\sqrt{1-a_t}z_1)\\=\sqrt{a_ta_{t-1}}x_{t-2}+\sqrt{1-a_ta_{t-1}}z_2\\=\sqrt{\bar{a}_t}x_0+\sqrt{1-\bar{a}_t}z_t
$$
其中每次加入的噪声都服从高斯分布$z_1,z_2,...,\sim\N(0,1)$，$\sqrt{a_t(1-a_t)}z_2,\sqrt{1-a_t}z_1$分别服从$\N(0,1-a_t),\N(0,a_t(1-a_{t-1}))$。因此相加后仍然服从高斯分布，可化简$\N(0,\sigma^2_1I)+\N(0,\sigma^2_2I)\sim\N(0,(\sigma_1^2+\sigma_2^2)I)$

这个累乘的公式告诉我们，任何时刻的分布都可以通过X0初始状态算出来，一步到位，这就是diffusion model的核心公式

## 二、去噪生成逆向过程

### 求解公式（贝叶斯公式）

$$
q(x_{t-1}|x_t,x_0)=q(x_t|x_{t-1},x_0)\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}
$$

逆向过程不会求解，使用贝叶斯公式
$$
q(x_{t-1}|x_0):\sqrt{\bar{a}_{t-1}}x_0+\sqrt{1-\bar{a}_{t-1}}z\sim \N(\sqrt{\bar{a}_{t-1}}x_0,1-\bar{a}_{t-1})\\
q(x_{t}|x_0):\sqrt{\bar{a}_{t}}x_0+\sqrt{1-\bar{a}_{t}}z\sim \N(\sqrt{\bar{a}_{t}}x_0,1-\bar{a}_{t})\\
q(x_{t}|x_{t-1},x_0):\sqrt{a_t}x_{t-1}+\sqrt{1-a_t}z\sim \N(\sqrt{a_t}x_{t-1},1-a_t)\\
$$
这三项都可以通过前向过程算出来，分布也列出来：
$$
\propto \exp(-\frac{1}{2}(\frac{(x_t-\sqrt{a_t}x_{t-1})^2}{\beta_t}+\frac{(x_{t-1}-\sqrt{\bar{a}_{t-1}}x_0)^2}{1-\bar{a}_{t-1}}-\frac{(x_t-\sqrt{\bar{a}_t}x_0)^2}{1-\bar{a}_t}))
$$
把标准正太分布展开后，乘法除法相当于幂级数相加减。

### 化简求上一时刻的分布

$$
\propto \exp(-\frac{1}{2}(\frac{(x_t-\sqrt{a_t}x_{t-1})^2}{\beta_t}+\frac{(x_{t-1}-\sqrt{\bar{a}_{t-1}}x_0)^2}{1-\bar{a}_{t-1}}-\frac{(x_t-\sqrt{\bar{a}_t}x_0)^2}{1-\bar{a}_t}))
\\=\exp(-\frac{1}{2}(\frac{x_t^2-2\sqrt{a_t}x_tx_{t-1}+a_tx_{t-1}^2}{\beta_t}+\frac{x^2_{t-1}-2\sqrt{\bar{a}_{t-1}}x_0x_{t-1}+\bar{a}_{t-1}x_0^2}{1-\bar{a}_{t-1}}-\frac{(x_t-\sqrt{\bar{a}_t}x_0)^2}{1-\bar{a}_t}))
\\=\exp(-\frac{1}{2}((\frac{a_t}{\beta_t}+\frac{1}{1-\bar{a}_{t-1}})x^2_{t-1}-(\frac{2\sqrt{a_t}}{\beta_t}x_t+\frac{2\sqrt{\bar{a}_{t-1}}}{1-\bar{a}_{t-1}}x_0)x_{t-1}+c(x_t,x_0)))
$$

C那个是常数项，不影响这个任务，又因为$\exp(-\frac{(x-u)^2}{2\sigma^2})=\exp(-\frac{1}{2}(\frac{1}{\sigma^2}x^2-\frac{2u}{\sigma^2}x+\frac{u^2}{\sigma^2}))$这样就能得到均值和方差了
$$
\bar{u}_t(x_t,x_0)=\frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1-\bar{a}_t}x_t+\frac{\sqrt{\bar{a}_{t-1}}\beta_t}{1-\bar{a}_t}x_0
$$
配完化简后的结果，但是x0现在未知

由前面Xt可以由X0计算得到，现在逆一下
$$
x_0=\frac{1}{\sqrt{\bar{a}_t}}(x_t-\sqrt{1-\bar{a}_t}z_t)
$$
所以，均值为
$$
\bar{u}_t=\frac{1}{\sqrt{a_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{a}_t}}z_t)
$$

### 模型估计噪音

$z_t$其实就是我们要估计的每个时刻的噪音

- 这家伙看起来无法直接求，只能训练一个模型计算
- 论文用的是Unet这种编码器解码器的结构
- 模型的输入参数有两个，分别是当前时刻的分布和时刻t

![image-20230101195411981](D:/Typora/typora-user-images/README/image-20230101195411981.png)



