//
// Created by lizengyi on 2023/5/9.
//

#ifndef DECISIONTREE_LIBDECISIONTREE_H
#define DECISIONTREE_LIBDECISIONTREE_H

#define VERSION 1

#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <map>
#include "variant"


struct DecisionTreeNode {
    std::variant<double, std::string> type;
    std::string attribute;                             // 特征或属性
    std::map<std::variant<int, double, std::string>, DecisionTreeNode*> branches;  // 子节点
};

struct Data {
    std::map<std::string, std::variant<int, double, std::string>> attributes;  // 属性
    std::string class_label;                                                  // 类别
};

DecisionTreeNode *dt_init(const std::vector<std::vector<std::variant<int, double, std::string>>> &data,
                       const std::vector<std::string> &attributes);

std::vector<std::string> dt_predict(const DecisionTreeNode *root,
                                 const std::vector<std::vector<std::variant<int, double, std::string>>> &inputData,
                                 const std::vector<std::string> &attributes);

std::vector<std::string> dt(const std::vector<std::vector<std::variant<int, double, std::string>>> &train_data,
                            const std::vector<std::vector<std::variant<int, double, std::string>>> &test_data,
                            const std::vector<std::string> &train_attributes,
                            const std::vector<std::string> &test_attributes);

double test(const std::vector<std::string>&);


#endif //DECISIONTREE_LIBDECISIONTREE_H
