# ML 实验

## 基本环境
1. opencv库函数编译和配置

## 神经网络
子系统中测试方法：
```bash
g++ Function.h Function.cpp libNet.h libNet.cpp MyNet.cpp -o MyNet  $(pkg-config --cflags --libs opencv4)
./MyNet
```
* 改进策略(按照优先顺序)
  1. 参数改进，可调参数：
     1. 学习率
     2. 训练次数
  2. 网络结构改进
     1. 不同的隐藏层以及每个层的神经元个数
  3. 数据集改进
     1. 数据采样方法
        1. 过采样
        2. 欠采样
     2. 数值正则化
        1. 全部统一正则
        2. 只正则部分
        3. 不正则
        4. 对分类特征数据进行数值类别转换
## 问题
* opencv在windows子系统下配置
  1. 无法编译成功（windows下需要非win32的另一个版本的mingw）
  2. 在子系统中编译需要将opencv编译文件放到ex文件系统（ubuntu目录下）
     3. 测试安装成功：`pkg-config --modversion opencv4`
  3. 配置好opencv后，还可能找不到具体的.so链接库，需要配置文件位置
     ```bash
     安装目录 > `/etc/ld.so.conf
     sudo /sbin/ldconfig -v
     ldconfig
  5. ```
  4. 编译写法：按照创建顺序写c++相关文件，h在前cpp在后，链接库：`$(pkg-config --cflags --libs opencv4)`
  5. 引入4.x版本opencv，头文件写法已经改变`<opencv2/opencv.hpp>`