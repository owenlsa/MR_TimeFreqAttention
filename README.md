# MR_TimeFreqAttention

```shell
src/ # 存放源码文件
scripts/ # 存放运行脚本
Outputs/ # 输出日志与模型存放
SpectDataset/ # 数据集存放(train, test, val)
```

主程序是`src/SC.py`，跑的话日志会重定向到`Outputs`文件夹里，所以建议这么运行
```shell
# nohup 关掉当前terminal也不会中断
# & 放在后台运行
nohup python src/SC.py &

# 输完上面的命令以后，运行的主要日志去Outputs里看
# tail -f 是查看文件最后且每秒刷新一次的意思
tail -f Outputs/RadioML2016.10a*******.log
```
