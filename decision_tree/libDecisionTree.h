//
// Created by lizengyi on 2023/5/9.
//

#ifndef DECISIONTREE_LIBDECISIONTREE_H
#define DECISIONTREE_LIBDECISIONTREE_H

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

DecisionTreeNode *init(const std::vector<std::vector<std::variant<int, double, std::string>>> &data,
                       const std::vector<std::string> &attributes);

std::vector<std::string> predict(const DecisionTreeNode *root,
                                 const std::vector<std::vector<std::variant<int, double, std::string>>> &inputData,
                                 const std::vector<std::string> &attributes);

std::vector<std::string> dt1(const std::vector<std::vector<std::variant<int, double, std::string>>> &data,
                            const std::vector<std::vector<std::variant<int, double, std::string>>> &testData,
                            const std::vector<std::string> &attributes,
                            const std::vector<std::string> &testattributes);

int sum(std::vector<int>);

std::string dt(const std::vector<std::string>&);

std::string dt(const std::vector<std::vector<std::string>>&);

void gauss(const std::vector<std::vector<std::string>> &train,
           const std::vector<std::vector<std::string>> &test, const std::string &filename);

#endif //DECISIONTREE_LIBDECISIONTREE_H
