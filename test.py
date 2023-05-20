import csv
import cal
train = []
test = []
with open('train-tree.csv', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        row_data = []
        for value in row:
            try:
                number = int(value)  # 将值转换为整数
                row_data.append(number)  # 添加整数到子列表
            except ValueError:
                row_data.append(value)  # 添加字符串到子列表
        train.append(row_data)
f.close()
with open('test-tree.csv', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        row_data = []
        for value in row:
            try:
                number = int(value)  # 将值转换为整数
                row_data.append(number)  # 添加整数到子列表
            except ValueError:
                row_data.append(value)  # 添加字符串到子列表
        test.append(row_data)
f.close()
# print(cal.sum([1,2,3]))

# tree = cal.init(train[1:], train[0])
# ans = cal.predict(tree, test[1:], test[0])
ans = cal.dt(train[1:], test[1:], train[0], test[0])

a = 0
b = 0
for row in test[1:]:
    if ans[a] == row[-1]:
        b = b + 1
    a = a + 1
print(b/a)
# cal.gauss(train, test, 'dt_answer1.txt')
