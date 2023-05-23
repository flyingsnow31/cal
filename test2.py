import pandas as pd
import cal

train = pd.read_csv("train-tree.csv", dtype={'JobLevel': str, 'StockOptionLevel': str, 'JobInvolvement': str})
test = pd.read_csv("test-tree.csv", dtype={'JobLevel': str, 'StockOptionLevel': str, 'JobInvolvement': str})

train_data = train.values.tolist()
test_data = test.values.tolist()

train_attr = train.columns.tolist()
test_attr = test.columns.tolist()

tree = cal.dt_init(train, traint, True)
ans = cal.dt_predict(tree, test, testt)

a = 0
b = 0
for row in test[1:]:
    if ans[a] == row[-1]:
        b = b + 1
    a = a + 1
print(b / a)