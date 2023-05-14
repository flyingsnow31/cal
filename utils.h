//
// Created by lizengyi on 2023/5/14.
//

#ifndef DECISIONTREE_UTILS_H
#define DECISIONTREE_UTILS_H

#include "vector"
#include "iostream"
#include "string"
#include "algorithm"
void printMax(std::vector<std::vector<std::string>> m, int row, int col) {
    for(int i = 0; i < row; ++i) {
        for(int j = 0; j < col; ++j) {
            std::cout << m.at(i).at(j) << ' ';
        }
        std::cout << std::endl;
    }
}

#endif //DECISIONTREE_UTILS_H
