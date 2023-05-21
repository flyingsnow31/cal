import pandas as pd
import cal

train_data = pd.read_csv("train-tree.csv", dtype={'JobLevel': str, 'StockOptionLevel': str, 'JobInvolvement': str})
test_data = pd.read_csv("test-tree.csv", dtype={'JobLevel': str, 'StockOptionLevel': str, 'JobInvolvement': str})
train = train_data.values.tolist()
test = test_data.values.tolist()
traint=train_data.columns.tolist()
testt=test_data.columns.tolist()
print(testt)
print(traint)

tree = cal.dt_init(train, traint)
ans = cal.dt_predict(tree, test, testt)
# # ans = cal.dt(train[1:], test[1:], train[0], test[0])
#
a = 0
b = 0
for row in test[1:]:
    if ans[a] == row[-1]:
        b = b + 1
    a = a + 1
print(b / a)
