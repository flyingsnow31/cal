import csv
import cal
train = []
test = []
with open('dt_train1.txt', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True, delimiter="\t"):
        train.append(row)
f.close()
with open('dt_test2.txt', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True, delimiter="\t"):
        test.append(row)
f.close()

cal.gauss(train, test, 'dt_answer1.txt')
