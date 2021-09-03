# 2020-05-07
刚开始学RL时嗯混的一个比赛,从网上找的代码用DQN训练吃豆人agent

具体做法是用resnet18直接从游戏画面里抽出来一条特征作为state喂给agent,然后agent选择上下左右四个动作

把resnet18的输入层改成对应像素宽高,输出改成4个

总的来说收获的点有: 

1.pycharm的环境配置(装包也太麻烦了)  

2.linux的基本操作(有华为V100显卡48小时白嫖试用),但是单用命令行显示不了画面其实还是不能训练...

3.本地跑了15W个eps,已经能达到通关需求了,验证了强化学习在离散动作任务中的可行性(即决策能力)

## 游戏画面
![game](https://github.com/iisdd/Competition/blob/main/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E6%AF%94%E8%B5%9B%E8%B5%84%E6%96%99/upload_pic/%E6%B8%B8%E6%88%8F%E7%94%BB%E9%9D%A2.png)
