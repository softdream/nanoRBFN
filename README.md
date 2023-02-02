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
