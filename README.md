

该项目是exFM项目的扩展

## features

在[exFM](https://github.com/tangwang/exFM)的基础上扩展了deep网络结构，新增的特性有：

1. 支持多个特征共享embedding参数。

​	支持多个特征共享embedding：itemID、tagID通常会出现在多个field（出现在item侧、user侧），如果ID特别稀疏配置为共享参数会有好处。但是对于FM如果做参数共享将导致信息丢失（特征是在哪个field），需要适当的修改网络结构。在deepFM中则没有这个问题。

2. 不再支持多种优化器，在fm部分选用ftrl，deep部分用adam 。
3. early stoping
4. 一个项目搞定特征处理、训练、在线的粗排（先对topN进行FM打分）、精排（对粗排的topM补充计算deep网络的打分）。

