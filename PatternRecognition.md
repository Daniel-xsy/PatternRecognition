## Lecture2：感知机PLA(Perceptron Learning Algorithm)

**一句话概括：在高维空间中找出某个线性超平面，使得样本能被完全正确的二分类**

由上面概括可大概判断PLA算法的一些特性：

1. 线性分类器
2. 解决二分类问题
3. 迭代至所有训练样本能够被完全正确分类才停止

<img src=".\img\PLA-1.png" alt="image-20210911085548880" style="zoom:67%;" />

从这幅流程图可看出，PLA算法需要具体讨论的点有：

1. 分类器模型具体是什么？
2. 如何定义损失函数（Loss）？
3. 如何寻找超平面，即如何参数优化$\bold W_{t+1} \larr \bold W_t$？
4. 算法收敛性如何？

### 2.1 分类器模型

感知机模型本质上就是将**输入向量$\bold x=(x_1,x_2,...,x_d)^T$与一组权重系数$\bold w=(w_1,w_2,...,w_d)$做内积，加上偏置$b$，判断其正负**。数学表示为：
<img src="http://latex.codecogs.com/gif.latex?\hat y=sgn(\sum^d_{i=1}w_ix_i+b)\\ \hat y=sgn(\bold W_d\bold x_d+b)" />

默认$y=1$表示分类为正，$y=-1$表示分类为负。此时可将$W$和$X$做**增广化**，将常数项并入向量乘法中，使表达式更简洁：
$$
\bold W_{d+1}=(1,w_i,w_2,...,w_d)\\
\bold x_{d+1}=(b,x_1,x_2,...x_d)\\
\hat y =sgn(\bold W_{d+1}\bold x_{d+1})
$$
从模型数学公式可看出，PLA本身仅仅为权重与输入的线性运算，故其本身是一个**线性二分类模型**，只能解决比较简单的**线性二分类问题**。但可以将多个学习到的感知机模型重叠在一起形成多分类模型。

<img src=".\img\PLA-2.png" alt="image-20210911091342676" style="zoom:80%;" />

从数学本质上来看，模型将Weight Vector与Feature Vector作内积，而内积一定程度上可以刻画两个向量之间的**相似度**，所以可以认为Weight Vector就是模型所学习到的**模式**或者称为**特征**

### 2.2 损失函数(Loss)/参数优化

PLA损失函数非常直接清晰，即未被正确分类的训练样本的个数
$$
L = bool(y_n \neq \hat y_n)
$$
优化过程也非常直接：
$$
\bold w_{t+1} = \bold w_t+y_n\bold x_{n(t)}
$$
从内积的角度看，**内积越大表示两个向量之间夹角越小，从而反应两向量更加相似**。模型希望最终学习到的**权重与样本尽可能相似**，因为只有这样，其内积结果才能尽可能大（正数），最终通过符号函数变为1（划分到正样本中）。所以，当$y=1$，$w$会向更接近$x$的方向移动，**使模式与数据特征更匹配或相似**；反之，当$y=-1$，$w$会远离该负样本。

<img src=".\img\PLA-3.png" alt="image-20210911092719447" style="zoom:67%;" />

### 2.3 算法收敛性

算法停止的条件为：所有样本均被正确分类。这自然就引出了一个问题：该模型在怎样的情况下会收敛？显然，对于非线性可分的数据，算法无论如何优化也不能通过一个线性超平面完全正确分类训练样本。那么，当训练样本为线性可分数据时，算法一定收敛吗？

![image-20210911093226343](.\img\PLA-4.png)

可从数学上证明：

![image-20210911093608485](.\img\PLA-5.png)

![image-20210911093630307](.\img\PLA-6.png)

### 2.4 线性不可分情况下的修正 -- Pocket算法

**一句话概括：不要求训练样本被完全正确分类，只需按照与PLA相同的参数优化策略，找到最佳的w，**

![image-20210911093901881](.\img\PLA-7.png)



## Lecture3 : 线性回归

**一句话概括：作为线性回归任务算法，损失函数定义为L2损失，模型输出的y不作任何处理**

### 3.2 线性回归算法

#### 优化目标

$$
\min_w L_{in}(\bold w)=\frac{1}{N}||\bold X\bold w-\bold Y||^2
$$

由于L2损失函数是凸函数，所以$\bold w$取0的时候，损失函数值为0，于是优化该函数即求：
$$
\grad L_{in}(\bold w) = \bold 0^T
$$
可以通过矩阵求导进行推导，如此通过数学方法建立方程—解方程得到的$\bold w$被称为“**解析解**”
$$
\grad L_{in}(\bold w)=\frac{L_{in}(\bold w)}{\part (\bold{Xw-}Y)}\cdot
\frac{\bold{Xw}-Y}{\part \bold{Xw}}\cdot
\frac{\part \bold{Xw}}{\part \bold w}\\
=\frac{2}{N}(\bold{Xw}-Y)^T\cdot\bold 1\cdot \bold X\\
=0
$$
可解得
$$
\bold w^* = (\bold{X}^T\bold X)^{-1}\bold X^TY
$$
在矩阵求导时，先明确链式法则中每一步的形状

### 3.3 梯度下降法

当样本量巨大时，使用解析解的方法计算$\bold w$较为困难，此时多采用迭代的方法求解。而梯度下降算法中最重要的便是学习率$\eta$，此处不加证明的给出“**当梯度较小时，我们希望有更大的学习率；当梯度较大时，我们希望有较小的学习率**”。故在实际应用中，学习率的调整也十分重要，以下分别介绍：

#### a. Adagrad

由上面内容可知，我们可以根据梯度来修正学习率，当$\bold w$是多维向量时，我们也希望在每个维度的学习率有所不同。
$$
w_{i,t+1} \larr w_{i,t}-\frac{\eta}{\sigma_{i,t}}\frac{\part L_{in}}{\part w_{i,t}}\\
\sigma_{i,t} = \sqrt{\frac{1}{t+1}\sum^t_{t=0}(\frac{\part L_{in}}{\part w_{i,t}})^2}+\epsilon
$$
使用该优化方法时，需要输入从开始迭代到目前为止的每一次梯度，$\epsilon$为保证数值稳定性，加在分母上。

优点可以自适应的调整学习率；缺点，随着迭代次数的增加，自适应调整能力减弱，并且所有过去时刻对当前的$\sigma$影响相同，导致不能及时地对当前的梯度做出反应，**反应迟滞**

#### b. RMSProp

只需要把当前梯度和过往梯度进行加权，便可以使最近更新的梯度值有着更大的影响力
$$
\sigma_{i,t}=\sqrt{\alpha(\sigma_{i,t-1})+(1-\alpha)(\frac{\part L_{in}}{\part w_{i,t}})^2}
$$

#### c. Momentum

以上两种方法都会遇到一个问题，当梯度接近0时，学习率都十分低。但是，梯度接近于0并不代表找到了最优解，因为此时有可能是**局部最优(Local optimal)**或者处于**鞍点(Saddle point)**，此时，我们希望优化过程能具有一定的**动量(Momentum)**，在梯度为0的时候仍然可以继续更新跳出局部最优或者快速渡过鞍点。**所谓动量，就是前一步的更新向量**，将其乘上比例系数与本次迭代的更新向量相加
$$
m_t=\lambda  m_{t-1}-\eta\grad L_{in}(w)
$$

#### d. Adam: RMPProp + Momentum

$$
m_t=\lambda  m_{t-1}-\grad L_{in}(w_)\\
\sigma_{i,t}=\sqrt{\alpha(\sigma_{i,t-1})+(1-\alpha)(\frac{\part L_{in}}{\part w_{i,t}})^2}\\
w_{i,t+1} \larr w_{i,t}-\frac{\eta}{\sigma_{i,t}}m_t
$$

除此之外，还需要考虑批量大小对梯度下降法的影响。计算梯度时，每次使用所有样本为**梯度下降法**，若每次只使用一个样本为**随机梯度下降法(SGD)**，若每次使用一个batch，叫做**批量梯度下降法**，其中，batch在每个epoch后会被reshuffle

<img src=".\img\LinearRegression-1.png" alt="image-20210917200513758" style="zoom:67%;" />

实验证明，当使用batch时，一次更新花费的时间和一个epoch花费的时间都是比较小的。同时，batchsize对性能也有一定的影响

<img src=".\img\LinearRegression-2.png" alt="image-20210917200639794" style="zoom:67%;" />

原因可以解释为：基于batch算出来的梯度具有一定的随机性，这样的随机性相当于在训练过程中添加了一定量的噪声，能够帮助优化过程快速**走出鞍点**。同时，由于batch数据分布仅仅为full batch的一个部分，在full batch上梯度较小时在batch上梯度可能较大，便于进一步优化



## Lecture4：Fisher线性判别

**一句话概括：找到某个投影超平面(super plane)，使得两个类别的特征在投影后，同类尽可能靠拢，异类尽可能分开**

使用Fisher判别的需求是：在输入为高维向量时，直接对高维向量进行二分类问题需要很大的计算开销。于是我们希望找到一个超平面对输入数据进行降维，使得降维后我们依然能够较好的分类。

以下仅讨论二分类的Fisher线性判别

<img src=".\img\Fisher-1.png" alt="image-20210919090520896" style="zoom:67%;" />

#### 目标函数

$$
J(w)=\frac{between \space calss \space scatter}{within \space calss \space scatter}\\
w^*={argmax}_w j(w)\\
J(w)=\frac{(E[s|y=1]-E[s|y=-1])^2}{var[s|y=1]+var[s|y=-1]}
$$

#### 数学知识回顾

##### 期望值

可以理解为服从某一概率分布的离散随机变量的加权平均值，其中权重为该变量出现概率的大小

##### 协方差

用于衡量两个随机变量之间**联合变化程度**，计算公式为：
$$
cov(X,Y)=E((X-\mu)(Y-\gamma))=E(X\cdot Y)-\mu \gamma
$$
当X与Y是统计独立时，二者之间的协方差为0，表示X的取值与Y的取值没有任何关系。而若X与Y不是统计独立，其协方差可能取非0值，表示X和Y之间存在某种**内在关联**。

协方差矩阵中
$$
\Sigma_{ij}=\mathrm cov(X_i,X_j)=E[(X_i-\mu_i)(X_j-\mu_j)]
$$
即$X=[X_1,X_2,...,X_n]^T$是由n个随机变量组成的向量

##### 拉格朗日乘子法

在一组约束条件下最优化某一函数值，比如
$$
\mathrm maxf(\bold x) \space \mathrm {subjected}\space \mathrm{to} \space g(\bold x)=0\\
\max f(\bold x,\lambda)=f(\bold x)+\lambda g(\bold x)
$$
即引入新变量，将条件极值问题转化为新函数的自由极值问题

#### 优化过程

根据方差和均值的定义可得

<img src=".\img\Fisher-2.png" alt="image-20210919095909625" style="zoom:33%;" />

<img src=".\img\Fisher-3.png" alt="image-20210919095916127" style="zoom:33%;" />

<img src=".\img\Fisher-4.png" alt="image-20210919095922901" style="zoom:33%;" />

<img src=".\img\Fisher-5.png" alt="image-20210919095930671" style="zoom:33%;" />

即得到投影方向。

<img src=".\img\Fisher-6.png" alt="image-20210919100026362" style="zoom:33%;" />



## Lecture5：逻辑斯蒂回归(Logistic Regression)

### 5.1 逻辑斯蒂回归问题

在分类任务中，我们除了想知道分类的结果，还想知道将数据分为该类的置信度的大小。所以，逻辑斯蒂回归输出的是一个概率值，表示该样本为正样本的概率大小。

由于概率值取$[0,1]$，逻辑斯蒂函数可以将$\bold w^T\bold x+b$映射到$[0,1]$之间

### 5.2 逻辑斯蒂回归损失

由于模型的输入为**概率值**，自然我们希望训练样本的标签值$y$也能是概率值，但实际上为$1$或$-1$，即标签为**含噪标签**。

逻辑斯蒂回归可以理解为**在给定样本下的极大似然估计**，其推导可参考[此处](https://zhuanlan.zhihu.com/p/347782092)。首先考虑能否使用EMS作为损失函数，即
$$
L_{in}(w)=(\theta(w^Tx)-y')^2
$$
<img src=".\img\LogisticRegression-1.png" alt="image-20211006162059179" style="zoom: 50%;" />

$\theta(yw^Tx)(1-\theta(yw^Tx))$项非常容易等于0，造成学习率极低。

接下来分析为什么选择交叉熵，其本质是对数的极大似然
$$
L(w)=argmax_w\prod_{n=1}^N(y_nw^Tx_n)
$$
取对数，再加负号，将极大变为极小
$$
L(w)=argmin_w\frac{1}{N}\sum_{n=1}^N\ln(1+\exp(-y_nw^Tx_n))
$$
交叉熵用于概率分布的拟合

<img src=".\img\LogisticRegression-2.png" alt="image-20211006163255384" style="zoom:50%;" />



## Lecture6：非线性变换(Nonlinear Transformation)

### 6.1 线性不可分问题

**一句话概括：线性不可分的问题都可以通过一定的特征空间变换最终转化为线性可分的问题**

首先考虑特殊情况，原本输入$\bold x$为一个二维特征$\bold x=[x_1,x_2]^T$，可以通过“**升维**”的方式升高到多维，即
$$
\Phi_2(\bold x)=(1,x_1,x_2,x_1^2,x_2^2,x_1x_2)^T
$$
现在需要考虑的问题是：当输入向量维度为$d$，经过非线性变换到$Q$次多项式时，特征空间变为多少维度？

这个问题本质上可以看作是一个**“有放回且无序”**的抽样问题，在$(1,x_1,x_2,...,x_d)$中有放回地拿出$Q$个进行相乘，最终有多少种可能性。不加证明地给出
$$
\widetilde d=C_{Q+d}^Q
$$
证明以后再完善

### 6.2 非线性变换

非线性变换的根本目的在于：**将非线性可分问题转化为线性可分问题，使得线性分类算法能够work**

### **6.3 知识拓展**



## Lecture7：线性支撑向量机(Support Vecter Machine)

### 7.1 最大间隔分类面

在线性分类问题中，当训练样本均为无噪声输入时，我们希望模型能尽可能多的容忍测试样本中的噪声，即——获得一个更加鲁棒的分类面。用数学公式表述：
$$
\max _x \mathrm {margin}(w)\\
\mathrm{subject \space to} \mathrm{\space every}\space y_n\bold w^T \bold x_n \gt 0\\
\mathrm {margin}(w)=\min _{n=1,...,N} \mathrm{distance}(\bold w,\bold x_n)
$$

### 7.2 标准的最大间隔问题

特征空间一向量$\bold x$距离超平面$\bold w$的距离可表述为：
$$
|r|=\frac{\bold w^T \bold x+b}{||\bold w||}
$$
即：
$$
\mathrm {margin}(w)=\min _{n=1,...,N} \frac{y_n(\bold w^T \bold x_n+b)}{||\bold w||}
$$
这一步比较难以理解：由于超平面的法向量$\bold w$和平移系数$b$可以比例缩放，不影响分类面本身以及分类的结果，所以可以令：
$$
\min _{n=1,...,N} y_n(\bold w^T \bold x_n+b)=1
$$
所以问题转化为了：
$$
\max _x \frac{1}{||\bold w||}\\
\mathrm{subject \space to} \mathrm{\space every}\space y_n\bold w^T \bold x_n \gt 0\\
\min _{n=1,...,N} y_n(\bold w^T \bold x_n+b)=1
$$
接下来进行条件松弛，这一步的目的是删去$\min$，使得之后的数学推导更直观。即将上述问题中的约束条件转化为
$$
y_n(\bold w^T \bold x_n+b)\geq1, \forall n\in N
$$
以下用反证法说明条件松弛后不影响该问题的解，即在$y_n(\bold w^T \bold x_n+b)\geq1$条件下解得的最优解$\bold w$和在条件$\min _{n=1,...,N} y_n(\bold w^T \bold x_n+b)=1$下解得的最优解相同

假设$\min _{n=1,...,N} y_n(\bold w^T \bold x_n+b)\geq1.3$可根据分 类面尺度不变性质得$\min _{n=1,...,N} y_n(\frac{\bold w^T}{1.3} \bold x_n+\frac{b}{1.3})\geq1$

所以最终问题可以转化为：
$$
\min _{\bold w} \frac{1}{2}\bold w^T\bold w\\
\mathrm{subject \space to}\space y_n(\bold w^T\bold x_n+b)\geq1,\forall n\in N
$$

#### 7.3 支撑向量机

该算法叫“支撑向量机”的原因是：超平面$\bold w$的取值仅由边界上的样本所决定。

损失函数为Hinge Loss:：$L_{SVM}=\max(0,1-ys)$

最终求解有两种方法：（1）梯度下降；（2）二次规划

首先考虑使用梯度下降求解SVM，损失函数定义为Hinge Loss
$$
L_{SVM}=\max (0,1-ys)
$$
<img src=".\img\SVM-1.png" alt="image-20211007110440794" style="zoom: 50%;" />

然后考虑二次规划求解

二次规划问题的特点是：（1）待优化的目标函数是凸函数 （2）约束条件是线性函数

<img src=".\img\SVM-2.png" alt="image-20211007111014179" style="zoom:80%;" />

此时只需要调用二次规划求解函数，并且将对应的参数匹配即可



## Lecture8：对偶支撑向量机与核支撑向量机(Dual SVM & Kernel SVM)

### 8.1 对偶支撑向量机动机

SVM只能用于解决线性问题，当遇到非线性问题时，需要进行非线性变换，导致维数较高。于是我们希望SVM的求解中（二次规划）可以不依赖于升维后的$\widetilde d$

首先利用拉格朗日乘子将约束条件下的优化问题转化为无约束条件下的优化问题：

<img src=".\img\DualSVM-1.png" alt="image-20211007093500481" style="zoom:80%;" />

由于$1-y_n(w^Tz_n+b)\leq 0$，在$a_n\geq0$时存在最大值，而Lagrange函数又存在最小值

### 8.2 对偶支撑向量机的拉格朗日分析

首先在所有可行的$a_n$中任选一组$a'$，可以得到

<img src=".\img\DualSVM-2.png" alt="image-20211007093949551" style="zoom: 50%;" />

如果此时的$a'$是使拉格朗日函数达到最优的那组值，则：

<img src=".\img\DualSVM-3.png" alt="image-20211007094047909" style="zoom:50%;" />

而二次规划问题又满足强对偶特性，即对等式两边最优化时，得到的最优$(b,w,\alpha)$是相同的，那么原问题的最优化便可以转化为对偶问题的最优化。

于是我们的最优化问题现在变为了：

<img src=".\img\DualSVM-4.png" alt="image-20211007094729162" style="zoom: 50%;" />

首先考虑括号内的$\min$拉格朗日函数问题，分别对$b$和$w$求导，可以得到：

<img src=".\img\DualSVM-5.png" alt="image-20211007094900068" style="zoom: 50%;" />

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20211007094910787.png" alt="image-20211007094910787" style="zoom: 50%;" />

将得到的$w$和$b$带入得到最终的优化问题

<img src=".\img\DualSVM-6.png" alt="image-20211007100430256" style="zoom: 50%;" />

### 8.3 求解对偶支撑向量机最佳值

在8.2中最终得到的优化问题又可以进一步看作是新的二次规划问题，即：

<img src="F:\OneDiveHUST\OneDrive - hust.edu.cn\courses\大三上\模式识别\Code\img\DualSVM-7.png" alt="image-20211007100830854" style="zoom:50%;" />

在求解出$a_n$后，便可以进一步求解$w$和$b$，此时可通过$w$的表达式看出，$a_n\gt0$所对应的样本决定$w$的取值，即这些样本就是支撑向量

### 8.4 对偶支撑向量机讨论

<img src=".\img\DualSVM-8.png" alt="image-20211007101621040" style="zoom:67%;" />

由原问题和对偶问题的比较中可以看出，将原问题转化为对偶问题可以不依赖于非线性变换后的维数$\widetilde {d}$

### 8.5 核函数支撑向量机

我们将原问题转化为对偶问题的原因是因为不想依赖$\widetilde {d}$，但在实际求解二次规划问题时，$Q$是一个稠密矩阵，且每一个元素都需要计算$z_n$的内积：

<img src=".\img\DualSVM-9.png" alt="image-20211007101950126" style="zoom:50%;" />

核函数的基本想法是，提高$z_n^Tz_m=\Phi(x_n)^T\Phi(x_m)$的计算效率。

<img src=".\img\DualSVM-10.png" alt="image-20211007103228484" style="zoom:50%;" />

所以可以定义核函数：

<img src=".\img\DualSVM-11.png" alt="image-20211007103309730" style="zoom:50%;" />

此后的计算便可以借助核函数完成

<img src=".\img\DualSVM-12.png" alt="image-20211007103629311" style="zoom:50%;" />

所以，即使原始数据进行无穷维的变换，核函数依然能够解决问题，其本质是不依赖$\widetilde {d}$
