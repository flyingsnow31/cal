//
// Created by lizengyi on 2023/5/9.
//

#ifndef DECISIONTREE_LIBDECISIONTREE_H
#define DECISIONTREE_LIBDECISIONTREE_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <map>

int sum(std::vector<int>);
std::string dt(std::vector<std::string>);
std::string dt(std::vector<std::vector<std::string>>);
void gauss(const std::vector<std::vector<std::string>>& train,
          const std::vector<std::vector<std::string>>& test, const std::string& filename);

#endif //DECISIONTREE_LIBDECISIONTREE_H
