import cal

train_file = 'simple_net/data/TrainData_upper.csv'
test_file = 'simple_net/data/TestData_upper.csv'

(train_data, train_attr) = cal.get_data(train_file, 26)
(test_data, test_attr) = cal.get_data(test_file, 26)

net = cal.libNet([26, 50, 2], 0.01, "sigmoid", 20, 1.0, "./output_log.csv", True)

net.train(train_data, train_attr, 1, 500000, False)

net.test(test_data, test_attr)

net.save()

