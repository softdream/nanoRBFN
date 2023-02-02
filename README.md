# nanoRBFN
RBF Network, implementated by c++
使用C++实现的径向基神经网络分类框架，可以完成简单的分类任务。使用header-only设计，只有头文件，部署使用超级简单。


## 使用
1. rbf.h和k_means_pp.h神经网络的主体部分只依赖Eigen库，可以使用下面命令直接安装：
```
sudo apt-get install libeigen3-dev
```

2. 此工程中提供了一个演示demo用于判断人脸的朝向，代码见face_orientation_detect.h中。
此demo依赖Opencv库。Opencv库的安装见https://github.com/opencv/opencv

3. 编译
```
mkdir build & cd build
cmake ..
make
```
在/bin目录下执行可执行文件即可。

## 人脸朝向识别
1. 数据来源：https://github.com/osama-afifi/RBF-Radial-Basis-Function-Network/tree/master/RadialBasisFunctionNetwork
2. 说明：
2.1 在人脸朝向识别的training数据集中一共有150张人脸不同朝向的图片，每张图片固定分辨率50 * 50；</br>
2.2 在人脸朝向识别的testing数据集中一共有30张人脸不同朝向的图片，每张图片固定分辨率50 * 50；</br>
2.3 图片数据只是简单的序列化为一个2500维的向量，作为神经网络的输入，因此这里神经网络的输入节点个数就是2500个。</br>

## 网络的使用方法
1. 在实例化一个RBF对象时，要显式的指明神经网络的数据类型，输入层节点个数，隐层节点个数，输出层节点个数，这四个参数作为模板参数传入。
```
rbf::RBF<T, InputLayer, HiddenLayer, OutputLayer> rbfn;
```
其中输入节点的个数就是数据向量的维度，隐层节点个数可以自己随便设置，一般越大越精确，但同时计算量会增大，输出层节点个数就是要分的类数，在人脸朝向识别中只分为三类（脸朝左，脸朝前，脸朝右）。
具体的可根据任务自己规划。
2. RBFN的主要api只有两个：
2.1 训练过程，根据已知的数据集学习出网络中的关键参数值：
```
training( const std::vector<RBFDataType>& training_data, // 训练数据集
          const std::vector<int>& traning_labels, // 训练数据集的标签集合
          const RBFValueType learning_rate, // 学习率
          const int num_iterations, // 最大迭代次数
          RBFValueType& mse ) // 均方误差

```
2.2 预测过程，根据实际的输入得到最终的结果：
```
predict( const RBFDataType& input_data,  // 输入数据
         RBFValueType& error ) // 误差值
```
