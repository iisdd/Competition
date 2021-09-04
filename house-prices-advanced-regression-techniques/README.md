# 房价预测(回归)
本次的融合模型结构是这样的:

初级模型(ENet,KRR,GB)->lasso回归->融合模型

融合模型 * 70% + XGB * 15 + LGB * 15% = 最终结果

## 删除离群点

### 删除前
![](https://github.com/iisdd/Competition/blob/main/house-prices-advanced-regression-techniques/upload_pic/Before%20Deleting%20outliers.png)

### 删除后,可以看到面积又大房价又小的点被删除
![](https://github.com/iisdd/Competition/blob/main/house-prices-advanced-regression-techniques/upload_pic/After%20Deleting%20outliers.png)
