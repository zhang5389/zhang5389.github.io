

### Python摘录1

From《深度学习入门：基于Python的理论与实现》

#### To study 
tqdm用的真多。

#### Python部分

此外，使用Python不仅可以写出可读性高的代码，还可以写出性能高（处理速度快）的代码。在需要处理大规模数据或者要求快速响应的情况下，使用Python可以稳妥地完成。因此，Python不仅受到初学者的喜爱，同时也受到专业人士的喜爱。实际上，Google、Microsoft、Facebook等战斗在IT行业最前沿的企业也经常使用Python。

cmd里面

pip list 可以列出装过的软件，非常有用。非常方便把软件卸载干净那种特殊的需求。

python终端里面

import xxx as x

x.两下划线version两下划线  可以显示版本号



数据类型:

```python
>>> type(10)
<class 'int'>
>>> type(2.718)
<class 'float'>
>>> type("hello")
<class 'str'>
```

列表或数组：

```python
>>> a = [1, 2, 3, 4, 5] # 生成列表 # 不能用 a = [1 2 3 4 5] 这种Matlab写法
>>> print(a)  # 输出列表的内容
```

切片：

```python
>>> a=[1,2,3,4,5,6,7]
>>> a[1:]
[2, 3, 4, 5, 6, 7]
>>> a[:3]
[1, 2, 3]
>>> a[:-1]
[1, 2, 3, 4, 5, 6]
```

 字典：

```
[1, 2, 3, 4, 5, 6]
>>> me={'height':174}
>>> me['height']
174
>>> me['weight']=80
>>> me
{'height': 174, 'weight': 80}
```

布尔型：

布尔型
Python中有bool型。bool型取True或False中的一个值。针对bool型的运算符包括and、or 和not（针对数值的运算符有+、-、*、/ 等，根据不同的数据类型使用不同的运算符）。

```python
>>> hungry = True     # 饿了？
>>> sleepy = False    # 困了？
>>> type(hungry)
<class 'bool'>
>>> not hungry
False
>>> hungry and sleepy # 饿并且困
False
>>> hungry or sleepy  # 饿或者困
True
```



其他：

```python
>>> for i in [1,2,3]:
	print(i)

1
2
3
```



类：

```python
class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")
    def hello(self):
        print("Hello " + self.name + "!")
    def goodbye(self):
        print("Good-bye " + self.name + "!")
        
m = Man("David")
m.hello()
m.goodbye()        
```



sys.path.append(os.pardir)  # 添加上一级目录的路径

 os.listdir(os.curdir)  # 列出当前目录

os.listdir(os.pardir)  # 列出h上一级目录



这个导入你目录的知识点要记住了：

 观察本书源代码可知，上述代码在mnist_show.py 文件中。mnist_show.py 文件的当前目录是ch03，
但包含load_mnist() 函数的mnist.py 文件在dataset 目录下。因此，mnist_show.py 文件不能跨目
录直接导入mnist.py 文件。sys.path.append(os.pardir) 语句实际上是把父目录deep-learning-
from-scratch 加入到 sys.path（Python 的搜索模块的路径集）中，从而可以导入 deep-learning-
from-scratch 下的任何目录（包括dataset 目录）中的任何文件。——译者注



这种写法不错，学习了。ReLU的class实现。

```Python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()				# 为什么要用 .copy 
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
```

把上面的代码看懂了！

 为什么要用.copy() ??

官方文档里面已经说了，官方文档写的真赞。

**copy --- 浅层 (shallow) 和深层 (deep) 复制操作**

源代码：Lib/copy.py

**Python 中赋值语句不复制对象，而是在目标和对象之间创建绑定 (bindings) 关系。对于自身可变或者包含可变项的集合对象，开发者有时会需要生成其副本用于改变操作，进而避免改变原对象。本模块提供了通用的浅层复制和深层复制操作（如下所述）。**

接口摘要：

- `copy.copy`(*x*)

  返回 *x* 的浅层复制。

- `copy.deepcopy`(*x*[, *memo*])

  返回 *x* 的深层复制。

- *exception* `copy.error`

  针对模块特定错误引发。

浅层复制和深层复制之间的区别仅与复合对象 (即包含其他对象的对象，如列表或类的实例) 相关:

- 一个 *浅层复制* 会构造一个新的复合对象，然后（在可能的范围内）将原对象中找到的 *引用* 插入其中。
- 一个 *深层复制* 会构造一个新的复合对象，然后递归地将原始对象中所找到的对象的 *副本* 插入。

深度复制操作通常存在两个问题, 而浅层复制操作并不存在这些问题：

- 递归对象 (直接或间接包含对自身引用的复合对象) 可能会导致递归循环。
- 由于深层复制会复制所有内容，因此可能会过多复制（例如本应该在副本之间共享的数据）。

The [`deepcopy()`](https://docs.python.org/zh-cn/3.8/library/copy.html?highlight=copy#copy.deepcopy) function avoids these problems by:

- 保留在当前复制过程中已复制的对象的 "备忘录" （`memo`） 字典；以及
- 允许用户定义的类重载复制操作或复制的组件集合。

该模块不复制模块、方法、栈追踪（stack trace）、栈帧（stack frame）、文件、套接字、窗口、数组以及任何类似的类型。它通过不改变地返回原始对象来（浅层或深层地）“复制”函数和类；这与 [`pickle`](https://docs.python.org/zh-cn/3.8/library/pickle.html#module-pickle) 模块处理这类问题的方式是相似的。

制作字典的浅层复制可以使用 [`dict.copy()`](https://docs.python.org/zh-cn/3.8/library/stdtypes.html#dict.copy) 方法，而制作列表的浅层复制可以通过赋值整个列表的切片完成，例如，`copied_list = original_list[:]`。

类可以使用与控制序列化（pickling）操作相同的接口来控制复制操作，关于这些方法的描述信息请参考 [`pickle`](https://docs.python.org/zh-cn/3.8/library/pickle.html#module-pickle) 模块。实际上，[`copy`](https://docs.python.org/zh-cn/3.8/library/copy.html?highlight=copy#module-copy) 模块使用的正是从 [`copyreg`](https://docs.python.org/zh-cn/3.8/library/copyreg.html#module-copyreg) 模块中注册的 pickle 函数。

想要给一个类定义它自己的拷贝操作实现，可以通过定义特殊方法 `__copy__()` 和 `__deepcopy__()`。 调用前者以实现浅层拷贝操作，该方法不用传入额外参数。 调用后者以实现深层拷贝操作；它应传入一个参数即 `memo` 字典。 如果 `__deepcopy__()` 实现需要创建一个组件的深层拷贝，它应当调用 [`deepcopy()`](https://docs.python.org/zh-cn/3.8/library/copy.html?highlight=copy#copy.deepcopy) 函数并以该组件作为第一个参数，而将 memo 字典作为第二个参数。



Sigmoid的Python实现如下：

```Python
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out					# 这句何用意?
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
```

self.out = out，这句有何用意？

ReLU的实现里面，也有一个mask。

**两个疑问了。**

解答：

发现没有，mask和out在两个函数里面都是有用的，相当于一个成员变量。就是这么简单。 



#### self，单独列出来，摘自python官方文档：

**为什么必须在方法定义和调用中显式使用“self”？**

这个想法借鉴了 Modula-3 语言。 出于多种原因它被证明是非常有用的。

首先，更明显的显示出，使用的是方法或实例属性而不是局部变量。 阅读 `self.x` 或 `self.meth()` 可以清楚地表明，即使您不知道类的定义，也会使用实例变量或方法。在 C++ 中，可以通过缺少局部变量声明来判断（假设全局变量很少见或容易识别） —— 但是在 Python 中没有局部变量声明，所以必须查找类定义才能确定。 一些 C++ 和 Java 编码标准要求实例属性具有 `m_` 前缀，因此这种显式性在这些语言中仍然有用。

其次，这意味着如果要显式引用或从特定类调用该方法，不需要特殊语法。 在 C++ 中，如果你想使用在派生类中重写基类中的方法，你必须使用 `::` 运算符 -- 在 Python 中你可以编写 `baseclass.methodname(self, )`。 这对于 [`__init__()`](https://docs.python.org/zh-cn/3.7/reference/datamodel.html#object.__init__) 方法非常有用，特别是在派生类方法想要扩展同名的基类方法，而必须以某种方式调用基类方法时。

最后，它解决了变量赋值的语法问题：为了 Python 中的局部变量（根据定义！）在函数体中赋值的那些变量（并且没有明确声明为全局）赋值，就必须以某种方式告诉解释器一个赋值是为了分配一个实例变量而不是一个局部变量，它最好是通过语法实现的（出于效率原因）。 C++ 通过声明来做到这一点，但是 Python 没有声明，仅仅为了这个目的而引入它们会很可惜。 使用显式的 `self.var` 很好地解决了这个问题。 类似地，对于使用实例变量，必须编写 `self.var` 意味着对方法内部的非限定名称的引用不必搜索实例的目录。 换句话说，局部变量和实例变量存在于两个不同的命名空间中，您需要告诉 Python 使用哪个命名空间。



（看不明白，暂时先记下来吧，加个self就行了。）



#### IDE（从IDE可以看出程序员里面绝大多数都是不懂设计的大傻逼）



Spyder为什么非要用Anaconda来装啊，好蛋疼。

Pycharm或者Spyder吧，两个都很好用。  

暂时先用Spyder吧，后面把 Anaconda也装好，IDE很重要，但也不是决定性的。因为AI实际上Python代码不多。

Spyder设计的真的很优秀，超级好用啊。Rodeo后面也试试吧。如果不能满足要求，后面直接用Pycharm即可 。把Kite一装。应该就可以暂时满足我现在的 需求了。

Spyder有点卡，卡点就卡点，也比wing ide这种明显有设计缺陷的软件要好。

Pycharm也有数据模式，这样的话，spyder就没啥优势了。

感觉Anaconda是必须要用的。



Kite是真好用，我要把Kite跟Spyder配置起来，本来是准备用VS Code的，但是总感觉VS Code差点东西，有时候跳转都不能用，帮助那里好像也有问题，不知道这里能不能解决。能够解决的话，也跟Spyder差的远。但是VS Code里面，Kite是好的，奇了怪了。

人工智能的插件，可以试试。机器学习的。



目前3.3.6有兼容性的问题。

Spyder4.0以后，会兼容Kite了。

 https://www.spyder-ide.org/blog/spyder-kite-funding/   



VS Code的这个问题应该是可以解决的，以后再说。先用Spyder4.0+Kite+官网上面的几个插件，以后把这套东西用熟练就行了。



anaconda装了以后，之前装的numpy之类的库好像全部用不了。真的非常无语。还是不用这个东西了。

其实python版本包括32位64位的版本，其实很好判断的，全部放在python的根目录下面就可以了。不同版本包括各个版本的库，都可以通过文件夹来区分。python自己搞个包管理器即可，anaconda这种东西出来真的就是他妈的添堵的，真的非常烂。



其实，Spyder里面，Kite没有发挥作用，不知道为什么。

VS Code里面，Kite发挥作用了。但是它里面有路径的问题。

VS code  Python   No module named 自定义包    

导入自定义包时，总是出问题。改一改好了，然后open打开绝对路径的又出问题。反正就是解决了一个问题，又出一个新问题，非常麻烦。

还是觉得Spyder更合理一些，因为毕竟可以用。VS Code这里真的是没法用，短时间内也解决不了。

就这样吧。

就这些问题，至少处理了四个小时，也没有找到一个圆满的解决方案，唉。

只能说VS Code 、Spyder、Kite三个软件里面有至少一个有严重的设计问题。现在的这些程序员水平真是他妈的烂，做的东西跟狗屎一样。

路径那里，是VS Code的设计有问题。妈的，任何项目都不要用绝对路径，这么简单的道理都不明白。这群废物大概没有理解文件夹的概念，文件夹是一个整体，可以随意Copy。

Spyder那里，是Kite的设计有问题，Kite检测不出来安装了Spyder软件，所以工作不了。大概就是这样。

其实，Spyder IDE也是蠢，妈的，你整个Exe安装包，安装的过程中向系统路径添加个路径就行了啊，妈的，这些怎么他妈的这么蠢。

这种级别的软件应该就是.exe的安装方法啊，你用pip安装， Kite怎么找到你呢？？

Kite也是傻，不找个配置让用户设置路径，然后检测是否有Spyder软件。这么大的软件，连个设置选项都不给，真是他们的狂妄啊。

anaconda更是傻逼中战斗机，网上大量这种问题：pip安装的包不能在anaconda中使用的问题

尼码的，整成个文件夹而已的事情，有这么难吗？？



这些大傻逼抱成团，圈成一圈了。呵呵。真他妈的。



foobar，连个歌词都搞不定，这不是设计问题是什么问题。



Spyder那里估计是有bug，即使装了4.0.0，跟Kite也没法工作。

还是用VS Code吧，路径的问题有篇回答，但是我没有弄好。

 https://segmentfault.com/a/1190000021046003 



现在是暂时这样，勉强可以用吧。

```Python
def init_network():
    #print(os.curdir)
    with open("ch03\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network
```

在Spyder那里却没有问题，不知道为什么。在Pycharm那里应该也没有问题。

等Spyder那里修复好了再说，我估计是没戏了，不知道出了啥问题。先这样吧。

可能跟文件夹的起始路径有关系？？？

VS Code的目录好像是文件夹的目录。而Spyder好像是执行哪个文件，就以哪个文件为起始目录。

先这样吧，接下来的一段时间内暂时就用VS Code。帮助文档也非常齐全，没啥问题的。VS Code真的很漂亮。后面多装几个Python有关的插件看看。

-------

VS Code 这里呢，用Run Coder确实 方便一些。CTRL+ALT+N 或者对源文件点右键运行，也是可以的。

还有路径的问题，这里再解释一下：

```Python
import sys, os
# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
print("Test 1 "+os.pardir)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
print("Test 2 "+BASE_DIR)

Test 1 ..
Test 2 d:\资料\Linux_当前\AI\【源代码】深度学习入门：基于Python的理论与实现
#  该文件的路径是  d:\资料\Linux_当前\AI\【源代码】深度学习入门：基于Python的理论与实现
```

加上路径的话，问题也是可以解决的。



VS Code里面创建 

创建 Jupyter Notebook
当安装好 Python 插件最新版后，使用快捷键 Ctrl+Shift+P 调出 Command Palette，然后输入 "Python: Create New Blank Jupyter Notebook" ，就能创建一个新的 Jupyter Notebook。

也是非常好用的，可以考虑这个帖子： https://zhuanlan.zhihu.com/p/85445777



到此为止，算是配置好了，可以用一段时间不用修改了。



另外，使用ipython把VS Code里面有时候还要另外打开idle(而idle没有补全功能和清屏功能？？)的问题也全部解决了。

参考这篇文章：

https://cloud.tencent.com/developer/news/19077

感觉整套机制越来越完美了。



ipython清屏：

!clear for Unix-like systems
!CLS for Windows

真他娘的好用。这种一块一块的写法也成为可能了。

![image-20191227163914214](Python.assets/image-20191227163914214.png)

连敲两下Enter，第一次也就是最后一个 ...: （如果想接着写代码，可以接着写的。）第二次也就是运行代码。

IDLE可以退出历史舞台了。



#### conda还是很有用的。

https://blog.csdn.net/yimingsilence/article/details/79388205 
使用这篇帖子里面的方法，修改了默认环境，然后呢，
在pip install 安装whl文件，结果装到电脑本地里面了，非常奇怪。

主要是conda里面的whl版本好多都没有。
这些软件兼容性真的很差，各种傻逼的版本需求。
其实就是在浪费时间，唉。

后面再解决这个问题。

暂时用conda install tensorflow==1.13.1

这个指令很有用的。
conda info --envs 
然后再激活了，再用conda install来安装。
激活 activate **即可。

conda的管理其实还不错。
跟Spyder结合着用吧。主力还是Python3.7.5+Pytorch+Tensorflow2.x。



至于怎么跟VS Code结合，可以CTRL+P，然后再 >select interpreter 选择解释器即可。

这样，差不多就能形成一个闭环了。



#### Numpy部分



```Python
>>> import numpy as np
>>> x = np.array([1.0,2.0,3.0])
>>> 
>>> x
array([1., 2., 3.])
>>> type(x)
<class 'numpy.ndarray'>
```

```python
>>> A = np.array([[1,2],[3,4]])   #注意其中的两个括号，这里很容易输错的。
>>> print(A)
[[1 2]
 [3 4]]
>>> A.shape
(2, 2)
>>> A.dtype
dtype('int32')
>>> A.type
```

点乘：

```Python
>>> A = np.array([[1,2],[3,4]])
>>> B = np.array([[5,6],[7,8]])
>>> A*B
array([[ 5, 12],
       [21, 32]])
```

广播功能：

广播
NumPy中，形状不同的数组之间也可以进行运算。之前的例子中，在2×2的矩阵A和标量10之间进行了乘法运算。在这个过程中，如图1-1所示，标量10 被扩展成了2 × 2的形状，然后再与矩阵A 进行乘法运算。这个巧妙的功能称为广播（broadcast）。

![image-20191225011516129](Python.assets/image-20191225011516129.png)

体会一下这里的广播功能：



```Python
>>> A = np.array([[1, 2], [3, 4]])
>>> B = np.array([10,20])
>>> A * B
array([[10, 40],
       [30, 80]])
>>> A * 10
array([[10, 20],
       [30, 40]])
```

访问元素：

```Python
>>> X = np.array([[51, 55], [14, 19], [0, 4]])
>>> print(X)
[[51 55]
 [14 19]
 [ 0  4]]
>>> X[0]    # 第0行
array([51, 55])
>>> X[0][1] # (0,1)的元素
55
```

除了前面介绍的索引操作，NumPy还可以使用数组访问各个元素。

```Python
>>> X = X.flatten()         # 将X转换为一维数组
>>> print(X)
[51 55 14 19  0  4]
>>> X[np.array([0, 2, 4])] # 获取索引为0、2、4的元素
array([51, 14,  0])

>>> X > 15
array([ True,  True, False,  True, False, False], dtype=bool)
>>> X[X>15]
array([51, 55, 19])
```



numpy的类型转换：

```Python
>>> import numpy as np
>>> x = np.array([-1.0,1.0,2.0])
>>> x
array([-1.,  1.,  2.])
>>> y = x > 0
>>> y = y.astype(np.int)
>>> y
array([0, 1, 1])
```

ndarray.ndim the number of axes (dimensions) of the array.

np.dot(A, B)  矩阵乘法（注意，点乘的话直接用 * 即可）



注意这里的行和列，还是熟悉Numpy里面的行和列比较好。行在后。

```Python
>>> A = np.array([[1,2,3], [4,5,6]])  # A是2行3列，注意了。
>>> A.shape
(2, 3)
>>> B = np.array([[1,2], [3,4], [5,6]])
>>> B.shape
(3, 2)
>>> np.dot(A, B)
array([[22, 28],
       [49, 64]])
```

这个居然可以，不太理解：

```Python
>>> A = np.array([[1,2], [3, 4], [5,6]])
>>> A.shape
(3, 2)
>>> B = np.array([7,8])
>>> B.shape
(2,)						# 这里为什么不直接显示（2，1），不理解
>>> np.dot(A, B)
array([23, 53, 83])
```

这个结果也不理解：  

```Python
>>> X = np.array([1, 2])
>>> X.shape
(2,)
>>> W = np.array([[1, 3, 5], [2, 4, 6]])
>>> print(W)
[[1 3 5]
 [2 4 6]]
>>> W.shape
(2, 3)
>>> Y = np.dot(X, W)
>>> print(Y)
[ 5 11 17]
```

 再加上下面的代码，对Numpy里面的处理完全懵了。

情况一：

```Python
>>> A = np.array([1,2])
>>> A
array([1, 2])
>>> B = np.array([3,4])
>>> C = np.dot(A,B)
>>> C
11
>>> A.shape
(2,)
>>> B.shape
(2,)
>>> D = np.dot(B,A)
>>> D
11
>>> A*B
array([3, 8])
```

情况二：

```Python
>>> a=np.array([[1],[2]])
>>> a
array([[1],
       [2]])
>>> b = np.array([1,2])
>>> np.dot(b,a)
array([5])
>>> np.dot(a,b)   # 报错
```



真他妈的不理解，把np.array([1, 2])设置成1乘2矩阵，把np.array([[1],[2]])设置成2乘1矩阵，这不就好了吗？？为什么设置成现在的这种形式，非常迷惑人。

x[1,2]的shape值(2,)，意思是一维数组，数组中有2个元素。**也就是1乘2，妈的，直接输出1*2不就行了吗？除非给出合理的解释，否则Numpy就是个烂库。**

y[[1],[2]]的shape值是(2,1)，意思是一个二维数组，2行1列。

一维数组。在Numpy中都表现为：（x,）。

```Python
>>> a = np.array([1,2,3])
>>> a
array([1, 2, 3])
>>> a.shape
(3,)
>>> b = np.array([[1,2,3]])
>>> b
array([[1, 2, 3]])
>>> b.shape
(1, 3)
```

这种设计的原因何在？？？我操你妈的。



（x,） 和(1,x)的区别何在？？如果没有合理的解释和没办法的选择（这样设计），那Numpy的作者真是他妈的一头猪。

网上有个帖子讨论这个的，后面再看吧。

 https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r 

 https://stackoverflow.com/questions/12665810/numpy-shape-gives-inconsistent-responses-why/12669291#12669291

 https://stackoverflow.com/questions/27866402/why-numpy-has-dimension-n-instead-of-n-1-only  

还是不理解这种傻逼设计，以后错了，然后把维数互换一下吧，真是没有办法，唉。 

（现实社会的本质就是，明知道是屎，还是要去吃，因为没有其他更好的选择了。就跟生活一样。你能选择不用Numpy吗？你能选择不活着吗？没有选择的。）

就是说出话来，情形一、情形二里面的现象又怎么解释？？

还有广播功能也是个非常蠢的功能。有些自以为很便捷的做法其实破坏了设计的一致性。我承认Numpy是很成功的项目，因为使用者很多。但是这个开发者，真的不太懂设计。

我懂又如何呢，没有项目证明自己，还不是处在新手的范畴？？

就算这里我没有理解，但是这么多人有疑问，难道不是错误的设计了？？？



还有，dot不是会让人联想到点乘吗？？？是不是？？



##### **不要骂Numpy了，把这个例子强制记下来即可。**

```Python
>>> X = np.array([1, 2])			#  先注意这里   (2,x)
>>> X.shape
(2,)
>>> W = np.array([[1, 3, 5], [2, 4, 6]])
>>> print(W)
[[1 3 5]
 [2 4 6]]
>>> W.shape
(2, 3)
>>> Y = np.dot(X, W)				# 再注意这里：  (M,)X(M,N) = (N,)
>>> print(Y)
[ 5  11  17]					    # 再注意这里： 
```

只要不是(x,)形式的乘法，都是满足矩阵乘法的要求的。

但是出现了(x,)的乘法，记忆即可。（有其他更新再补充）

(M,)  元素个数为M的一维数组，也就是1*M矩阵 （代入下式，转了两次，所以说设计不合理）

 **(M,)X(M,N) = (N,)     //  (M,N) X (M,) 则报错！！**

 **(M,)X(M,) = 一个数**



#### matplotlib

绘图：

```Python
import numpy as np
import matplotlib.pyplot as plt
# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)
# 绘制图形
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos") # 用虚线绘制
plt.xlabel("x") # x轴标签
plt.ylabel("y") # y轴标签
plt.title("sin & cos") # 标题
plt.legend()
plt.show()
```

显示图像 ：

```Python
import matplotlib.pyplot as plt
from matplotlib.image import imread
img = imread('lena.png') # 读入图像（设定合适的路径！）
plt.imshow(img)
plt.show()
```

plt.ylim(-0.1, 1.1) # 指定y轴的范围



### 





