[TOC]
# 第6天
## 1.神经网络的特点
- 同一层神经元之间没有链接
- 由**输入层, 隐层, 输出层**组成
- 全连接层-> 最后的一层n与n-1层之间的神经元链接-> 决定最后的输出结果的个数

![](/home/zhy/图片/learn_image/t.png)

## 2.浅层人工网络神经模型
### 2.1 SoftMax回归
作用: 计算概率
特点: 所有的类别相加=1
**e是个常数: e=2.71**
公式意思-> s = e^i / (e^1 + e^2 + e^3 ... e^n)

![](/home/zhy/图片/learn_image/2019-03-12 16-55-33 的屏幕截图.png)



<font color=goldenrod>**SoftMax与特征和权重和偏执值的关系示例图**</font>
![](/home/zhy/图片/learn_image/2019-03-12 17-05-55 的屏幕截图.png)

## 3.损失计算-交叉熵损失
通过交叉熵损失公式-> 判断准确性-> 交叉熵损失值越小越准确

公式了解即可
返回的值越小损失的值越小,越准确

![](/home/zhy/图片/learn_image/2019-03-12 17-31-22 的屏幕截图.png)

**<font color=goldenrod>各种算法的策略和优化的不同</font>**
![](/home/zhy/图片/learn_image/2019-03-12 17-35-36 的屏幕截图.png)


### 3.1.全连接-从输入直接到输出->特征+权重的API
- 特征加权重API:就是线性回归的公式接口
tf.matmul(a, b,name=None)+bias
参数说明:
 - a是特征的值
 - b是权重值
 - bias是偏执的值

return:全连接结果-> 供交叉损失运算


### 3.2. SoftMax计算、交叉熵API
- API: tf.nn.softmax_cross_entropy_with_logits(labels=None,logits=None,name=None)
作用: 计算logits和labels之间的交叉损失熵
参数说明:
  - labels:标签值（真实值)-> 最终真实的结果值
  - logits：样本加权之后的值-> softmax转化之前的值
  - 返回值:返回损失值列表--> 需要求出平均的损失值进行优化,见下个api

### 3.3. 损失值列表平均值的计算API
- tf.reduce_mean(input_tensor)
  - 计算张量的尺寸的元素平均值

### 3.4 梯度下降优化API
- 梯度下降优化API:
tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
参数说明:
  - learning_rate:学习率，一般为0-1之间, 一般写0.1就足够大

- 方法: minimize(loss):最小化损失
 - 方法的参数loss是一般是损失的平均值

- 返回值: return:梯度下降op-> 在会话运行不断的优化

### 3.5 准确性的计算
- equal_list = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
 - y 和 y_label 分别为真是的额目标值和预测值,通过这个api来判断是否相等
 - tf.argmax(y_label,1)-> 找出预测结果中分类概率最大值, one_hot编码中的1的标记值是否一致;
 - 返回的是一组数据1的列表, 若相同则标记为1, 不等则标记为0
 - 如下图
![](/home/zhy/图片/learn_image/2019-03-12 20-22-42 的屏幕截图.png)

**返回的equal_list的样式**

![](/home/zhy/图片/learn_image/2019-03-12 20-26-45 的屏幕截图.png)

- accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
 - 所有值求和然后平均, 返回的是准确率
 - 返回的accuracy就是准确率


## 4. 卷积神经网络
神经网络分为: 输入, 隐藏, 输出层

卷积神经网络把隐藏层多分为: 卷积层, 池化层

卷积神经网络->原始图片提取特征-> 生成新特征->再次提取-> 目标值

![](/home/zhy/图片/learn_image/2019-03-13 17-57-10 的屏幕截图.png)

### 4.1 卷积神经网络原理示意图
**<font color=goldenrod size=4>卷积神经本质: 特征值*权重 + 偏执-->得到新的特征值; 特征: 输入值, 权重: 过滤器(数据类型和个数权重与过滤器一致), 偏执:数量等于过滤器输出的数量的值; -->最终经过卷积神经后得到的值还是一个特征的值</font>**

**<font color=goldenrod size=4>(卷积神经网络整个过程)卷积层-->激活函数-->池化层--> 不断的重复-> 全连接层->输出目标值</font>**

**<font color=goldenrod size=4>(卷积神经网络整个过程示意)一个通道一个Filter一步长</font>**

![](/home/zhy/图片/learn_image/2019-03-13 18-00-43 的屏幕截图.png)

**<font color=goldenrod size=4>(卷积神经网络整个过程示意)（步长为2时候)</font>**

![](/home/zhy/图片/learn_image/2019-03-13 18-15-37 的屏幕截图.png)


**<font color=goldenrod size=5>多通道图片-池化过程</font>**

说明:
 - 彩色图片是3通道,黑白是1通道, 当多个图片->多个通道的,
 - 第一次卷积层过滤器设置多少-->输出的值决定第二层卷积时候过滤器的设置
 - 比如第一层卷积设置过滤器是5\*5\*32,-->输出结果
 - 第二层卷积过滤器初始就是对应的32个数-->就是5\*5\*32-->再设置二层过滤器应是 5\*5\*32\*n
 - 以此类推, 权重的shape等同于过滤器的shape, 权重的值由过滤器个数决定

![](/home/zhy/图片/learn_image/卷积网络动态演示)

### 4.2 卷积层的0填充
卷积层在提取特征映射动作称为-> padding(零填充)
由于步长不一定刚好到图片像素的边缘, 有2中方式来处理
- SAME: 越过边界取样, 卷积之后输出大小一样
- VALID: 不越过边界取样, 取样小于图片

**如图: 用SAME越出边界,用0填充,卷积之后输出的大小不变**

![](/home/zhy/图片/learn_image/2019-03-13 18-29-09 的屏幕截图.png)

### 4.3 卷积神经网络的结构
#### 4.3.1 卷积过滤器的输入和输出计算
要记住,会计算
- 输入体积的大小 = H1 \* W1 \* D1
 - 图片的高度像素 * 图片的宽度项目 * 通道数
- 四个超参数
 - Filter数量K-> 过滤器的数量
 - Filter大小F-> 过滤器的大小, 一般设置为1\*1, 3\*3, 5\*5
 - 步长S
 - 零填充的大小P
- 卷积层输出体积的大小: H2 \* W2 \* D2
 - H2 = (H1 - F + 2P) / S + 1
 - W2 = (W1 - F + 2P) / S + 1
 - D2 = K -> 输出深度=过滤器的个数


___
### 4.4 卷积网络的API

卷积层：
- tf.nn.conv2d(input, filter, strides=, padding=, name=None)
 - 计算给定4-D input和filter张量的2维卷积
- 参数说明
 - input：输入值，具有[batch,heigth,width,channel]，类型为float32,64
   - 输入值必须是4D的类型,其他->需要转换
 - filter：过滤器的大小，[filter_height, filter_width,in_channels, out_channels]
   - 过滤器的形状为[过滤器的长, 宽, 输入的通道数, 输出的通道数]
   - 输入的通道数->图片黑白彩色而不同
   - 输出的通道数->取决于过滤器的个数
   - <font color=goldenrod>**过滤器是带权重的值,权重的相撞和过滤器的形状是一致的**</font>
 - strides：strides = [1, stride, stride, 1],步长一般设置为->[1,1,1,1]表示上下左右的步长都是1
 - padding：“SAME”, “VALID”，使用的填充算法的类型，使用“SAME”。其中”VALID”表示滑动超出部分舍弃，“SAME”表示填充，使得变化后height,width一样大
- **卷积处理-->得到值为目标值-->作为新特征供--> 返回值需要加上偏执的值-->才是新特征的值**
- tf.nn.conv2d(input, filter, strides=, padding=, name=None) + bias


### 4.5 新的激活函数Relu
说明: 为什么需要激活函数->更好的分割->通过数学推导获取的结论更好的优化-> 相当于增加网络的非线性的分割能力
- 激活函数为什么用relu, 放弃sigmod
  - 计算量小
  - 对于深层的网络sigmoid函数反向传播时候容易出现梯度爆炸
- 激活函数公式 f(u) = max(0, x)
- sigmoid公式: sigmoid = 1/1+e^-z -> 可见rule计算量小

**激活函数的API**
- tf.nn.relu(features, name=None)
参数说明:
  - features:卷积后加上偏置的结果->就是卷积过滤后输出的结果
  - return:结果

### 4.6 池化层(Pooling)原理和计算API
Pooling池化层的作用-> 减少特征-> 常用的方法Max, Pooling-> 一般取2*2 -> 2个步长

- 池化层的示意图: 左图 -> 实际效果, 右图 -> 原理
 - 获取的输出的值->
 - 2*2步长过滤器->提取过滤器中最大值
 - 过滤器移动步长2 -> 再次提取最大值
 - 获取新的输出值

![](/home/zhy/图片/learn_image/2019-03-13 20-56-02 的屏幕截图.png)

**池化层的API**
- tf.nn.max_pool(value, ksize=, strides=,padding=,name=None)
输入上执行最大池数
参数说明:
 - value:4-D Tensor形状[batch, height, width, channels]->一般是卷积函数的输出值
 - ksize:池化窗口大小，[1, ksize, ksize, 1]-> ksize设置为2
 - strides:步长大小，[1,strides,strides,1]-> 步长也设置为2
 - padding:使用“SAME”填充

### 4.7 卷积神经网络的过程示意图

图片->卷积->激活函数->卷积->激活函数->池化->多次处理后->全连接层-> 目标值

- 前面的卷积工程-> 相当于特征工程
- 后面的全连接相当于特征加权
- 全连接层相当于分类

![](/home/zhy/图片/learn_image/2019-03-13 21-26-57 的屏幕截图.png)

## 5.额外知识点和使用案例
1. 形状不确定时候转化,不确定用-1填充
```
x是个变量-> [28,28,1], 把其转换为4D的
x_reshape = tf.reshape(x,[-1,28,28,1])  # todo:改变形状的时候不知道的形状填-1
```

2. 卷积层的案例->卷积->激活函数->池化(详细键第6天代码)
```
    # 2. 第一个卷积层-> 需要经过卷积, 激活函数.-> 池化
    # 卷积过滤器大小5*5*1 , 32个过滤器, strides=1 -> 激活,池化
    with tf.variable_scope("conv1"):
        # 特征值x的shape固定的要求[None, 784] [None, 28,28,1]->转为4D
        x_reshape = tf.reshape(x,[-1,28,28,1])  # todo:改变形状的时候不知道的形状填-1

        # todo: 初始化随机的权重-> 过滤器是带权重的,
        #  所以权重的形状shape需要确定过滤器大小和数量才能确定,且权重的形状是矩阵的类型
        # todo: 矩阵的特点就是最后一个表示的是个数;这里权重的变量是定义一个函数定义的
        w_conv1 = weight_variables([5,5,1,32])
        # 共有32个偏值
        b_conv1 = bias_variables([32])
        # 调用卷积过滤的api,记得需要加上偏执. 实质就是特征的提取-> 本质原理-> y= xw + b
        # 卷积后:[None, 28,28,1] --> [None, 28,28,32]
        conv_val = tf.nn.conv2d(x_reshape, w_conv1, strides=[1,1,1,1], padding="SAME") + b_conv1
        # 激活函数relu激活
        x_relu1 = tf.nn.relu(conv_val)
        # 池化 2×2 strides=2, [None, 28,28,32]-->[None, 14,14,32]
        x_pool = tf.nn.max_pool(x_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
```



