#include "libDecisionTree.h"
#include "iostream"
int sum(std::vector<int> l) {
    int num = 0;
    for (auto n : l) {
        num += n;
    }
    return num;
}

std::string dt(std::vector<std::string> strs) {
    std::string a;
    for(auto str : strs) {
        a += str;
    }
    std::cout << a << std::endl;
    return a;
}
std::string dt(std::vector<std::vector<std::string>> strss) {
    std::string a;
    for(auto strs : strss) {
        for(auto str : strs) {
            a += str;
        }
    }
    std::cout << a << std::endl;
    return a;
}