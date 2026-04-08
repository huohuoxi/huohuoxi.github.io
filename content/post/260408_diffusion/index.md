
---
title: 【扩散模型极简介绍】从2015的Diffusion到2020的DDPM
description: 阅读这两篇论文得到的一些笔记
date: 2026-04-08 14:41:00+0000
math: true
image: cover_1.png
categories:
    - 机器学习
tags:
    - 扩散模型
    - 生成模型
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

声明：部分内容有争议，比如VAE中的隐变量空间是否是高斯的，这里不展开，只给一个最容易理解的版本。

## 1. 背景

生成模型是为了解决下面这个问题：  
如何学习到给定样本的分布 $p(x)$。

如果我们实现了这个目标，也就是说我们的模型能找到一系列样本符合怎样的模式（训练），就可以反过来遵循这种模式，生成出符合这种模式的全新的数据（采样）。

读到这里你可以思考一下，假如让你来设计一个神经网络，你会如何设计来拟合这个采样过程？

下面记我们模型给出的概率分布为 $p(x)$，训练集中样本真实的分布是 $q(x)$。

一个最自然的想法是，在采样过程中，假设我们从一个随机的噪声分布 $p(z)$ 开始，如果能找到一个函数 $p(x\mid z)$（假设这个函数也是高斯的好了，我们用机器学习来学它的均值和方差），利用下面的条件概率：

$$
p(x,z)=p(z)p(x\mid z)
$$

对 $z$ 的全空间进行积分：

$$
p(x)=\int p(z)p(x\mid z)\,dz
$$

把我们建模的 $p(x)$ 和样本的真实分布 $q(x)$ 进行对比拟合，让神经网络不断调整其中函数的参数，就可以学习到样本的分布了。

注：以上两个式子很重要，注意到这里前面一个式子只能得到 $x$ 和 $z$ 的联合分布，只有积分之后才能得到样本分布。

传统的VAE模型就是我们的思路（这里是经过简化的，实际训练需要同时训练一个encoder和一个decoder）。可惜传统的VAE模型被证明，生成效果不是很好。这样我们就自然过渡到了diffusion模型。

## 2. 2015的Deep Unsupervised Learning using Nonequilibrium Thermodynamics

这篇文章主要说了下面一件事:
传统模型之所以有问题，是因为我们假设 $p(x\mid z)$ 是高斯的，但是这个函数离高斯比较远，拟合不够有效。所以，我们把 $z$ 到 $x$ 分解成无数个步骤，而不是通过一步直接完成的。我们把之前的 $z$ 记作 $x_T$（纯粹高斯噪声），而把原来的 $x$ 记作 $x_0$（真实样本）。单步去噪过程就是从 $x_T$ 到 $x_{T-1}$，经过 $T$ 步这样的去噪过程，才得到我们的真实样本 $x_0$。而连接两步之间的映射函数，我们假设是高斯的。

这样一来，我们的条件概率就可以写成：

$$
p(x_0,x_1,\dots,x_T)=p(x_0\mid x_1)p(x_1\mid x_2)\cdots p(x_{T-1}\mid x_T)p(x_T)
$$

这样的一种连乘形式。

再对这个联合分布从 $x_1$ 到 $x_T$ 的全空间积分，就可以得到我们建模的样本分布：

$$
p(x_0)=\int dx_1\,dx_2\cdots dx_T\; p(x_0\mid x_1)p(x_1\mid x_2)\cdots p(x_{T-1}\mid x_T)p(x_T)
$$

也就是说，对于任意时间 $t$，我们只需要调整 $p(x_{t-1}\mid x_t)$ 就可以了。然后我们需要大胆假设 $p(x_{t-1}\mid x_t)$ 是高斯的（这个假设比 $p(x\mid z)$ 是高斯的要更接近事实，也就是说diffusion模型比传统VAE模型的表现要更好）。

下面我们的目标就是拟合高斯函数 $p(x_{t-1}\mid x_t)$ 的分布。对于一个高斯分布来说，只有两个参数需要确定，一个是均值 $\mu$，一个是方差 $\sigma$（这里是多维高斯分布，需要的是协方差矩阵 $\Sigma$）。

我们怎么设定loss呢？我们的loss应该可以让建立的模型 $p(x_0)$ 和样本中真实的分布 $q(x_0)$ 进行对比。

如果是一般的机器学习，我们会使用一些均方根误差之类的误差项。但这里是对比两个分布，在对比两个分布的时候，一般采用最大似然估计作为误差项。

代入我们这里的 $p(x_0)$ 和 $q(x_0)$ 就有：

$$
L=\int dx_0\,q(x_0)\log p(x_0)
=\int dx_0\,q(x_0)\log\int dx_1\,dx_2\cdots dx_T\; p(x_0\mid x_1)p(x_1\mid x_2)\cdots p(x_{T-1}\mid x_T)p(x_T)
$$

其中 $q(x_0)$ 是已知的（训练集中的分布），$p(x_0)$ 是我们经过分布的去噪模型吐出来的。

事实上所有的建模就到此结束了，我们已经有了loss，接下来让我们的loss对所有可学习的参数（就是我们假设的高斯分布里的均值和协方差）自动微分就可以了。

但是注意到我们的log里面是一个多重积分，所以自动微分很困难，需要化简这个式子。这篇文章之所以难读，就是因为文章中主要的篇幅都是在说如何化简这个loss，比较喧宾夺主。为了化简这个式子，我们开始使用一些数学技巧：

刚才我们假设真实的样本是通过纯粹噪声经过 $T$ 步去噪得到的，反过来，我们也可以假设真实的样本分布 $q(x_0)$ 可以通过一个分步的加噪过程，得到纯粹噪声：

$$
q(x_0,x_1,\dots,x_T)=q(x_T\mid x_{T-1})\cdots q(x_1\mid x_0)q(x_0)
$$

$$
q(x_0)=\int dx_1\,dx_2\cdots dx_T\; q(x_T\mid x_{T-1})\cdots q(x_1\mid x_0)q(x_0)
$$

然后通过这个分解，再加上其他的一些数学上的小trick（见2015原文的Appendix B，或者见下面2020那一篇原文的Appendix A，二者是等价的），就可以把上面的loss化简得到：

$$
L_{\mathrm{vlb}}=D_{\mathrm{KL}}\bigl(q(x_T\mid x_0)\,\|\,p(x_T)\bigr)+\sum_{t=2}^{T}D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t,x_0)\,\|\,p(x_{t-1}\mid x_t)\bigr)-\log p(x_0\mid x_1)
$$

不需要知道具体过程，解释一下我们做了什么，最开始的那个多重积分不可解，而化简后，这里loss中的每一项都（几乎）是模型中假设的高斯分布 $p(x_{t-1}\mid x_t)$ 和一个通过真实样本可以计算出的单步加噪分布 $q(x_{t-1}\mid x_t,x_0)$ 之间的KL散度比较。相当于我们只需要对每一步单独做高斯拟合。

然后作者发现，$q(x_{t-1}\mid x_t)$ 仍然是无法计算的，所以在上面化简过的loss中，通过重期望公式，把所有的 $q(x_{t-1}\mid x_t)$ 都改成了含有 $x_0$ 的 $q(x_{t-1}\mid x_t,x_0)$。这就是为什么公式中出现了 $q(x_{t-1}\mid x_t,x_0)$。

https://www.zhihu.com/question/1339105941

以上就是2015年Deep Unsupervised Learning using Nonequilibrium Thermodynamics这篇文章的全部内容。

## 3. 2020年的Denoising Diffusion Probabilistic Models

2015年的那篇文章出现之后，并没有引起很大的轰动。原因是，模型表现仍然有限，没有超过当时SOTA的GAN模型。2020年的这篇文章Denoising Diffusion Probabilistic Models（简称DDPM）才真正在表现上得到了接近SOTA的水平，从而开启了扩散模型的时代。

这篇文章相比2015年的文章更进一步之处在于：
上面的Loss式子中，忽略头尾，中间最重要的 $L_{t-1}$ 部分就是通过一个已知量 $q(x_{t-1}\mid x_t,x_0)$ 来拟合一个高斯分布 $p(x_{t-1}\mid x_t)$：

$$
L_{t-1}=D_{\mathrm{KL}}\bigl(q(x_{t-1}\mid x_t,x_0)\,\|\,p(x_{t-1}\mid x_t)\bigr)
$$

拟合高斯分布，换句话说就是拟合它的均值和协方差。对于协方差，我们的处理是，忽略非对角项，定义它是一个对角矩阵 $\sigma_t^2 I$，并且 $p$ 的协方差通过和 $q$ 的协方差保持一致来得到，从而降低参数量。

因为现在 $p$ 和 $q$ 有相同的协方差了，对于两个高斯项的KL散度，我们就只剩下均值的比较了：

$$
L_{t-1}\propto \left\|\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)\right\|^2
$$

因为我们用来拟合的 $x_t$ 其实是通过训练样本中的 $x_0$ 加噪得到的，所以是样本和噪声的某种混合：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

其中 $\epsilon$ 是高斯噪声，$\alpha_t$ 是超参数。

所以作者通过一通化简计算，告诉我们，$L_{t-1}$ 有一个等价形式，在这个形式中可以不出现 $x_0$，只出现噪声：

$$
L_{t-1}\propto \left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2
$$

也就是说，建模的时候，我们不再建模每一步的 $x$ 是高斯的，然后计算 $q(x_{t-1}\mid x_t,x_0)$ 和 $p(x_{t-1}\mid x_t)$ 分布之间的loss了。我们直接拟合每一步加入的噪声 $\epsilon_\theta(x_t,t)$ 就可以了。作者发现这么做的效果最好。这就是2020这篇文章相比2015文章最大的进步。

至于具体的推导就不在这篇文章中展示了，有兴趣的同学可以自己查阅。

## 参考文献

https://zhuanlan.zhihu.com/p/11263356450  
https://lilianweng.github.io/posts/2021-07-11-diffusion-models/  
https://spaces.ac.cn/archives/9119  
https://spaces.ac.cn/archives/9152  
https://spaces.ac.cn/archives/5253  
以及文章中提到的paper原文


