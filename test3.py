import pandas as pd
import cal

from sklearn.model_selection import KFold

train = pd.read_csv("train-tree.csv", dtype={'JobLevel': str, 'StockOptionLevel': str, 'JobInvolvement': str})
test = pd.read_csv("test-tree.csv", dtype={'JobLevel': str, 'StockOptionLevel': str, 'JobInvolvement': str})

train_data = train.values.tolist()
test_data = test.values.tolist()

train_attr = train.columns.tolist()
test_attr = test.columns.tolist()

kf = KFold(n_splits=4)
kf.split(train_data, train_attr)
for train_index, test_index in kf.split(train_data):
    # 将属性和标签分为训练集和测试集
    X_train, X_test = attributes[train_index], attributes[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # 在训练集上训练模型
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算评估指标（例如准确率、均方误差等）
    score = your_evaluation_metric(y_test, y_pred)

    # 将评估结果添加到 scores 数组中
    scores.append(score)
print(kf)

tree = cal.dt_init(train_data, train_attr, True)
ans = cal.dt_predict(tree, test_data, test_attr)

a = 0
b = 0
for row in test[1:]:
    if ans[a] == row[-1]:
        b = b + 1
    a = a + 1
print(b / a)



