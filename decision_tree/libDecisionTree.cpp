#include "libDecisionTree.h"
#include <cmath>
#include <algorithm>
#include "set"
#include <cassert>

int sum(std::vector<int> l) {
    int num = 0;
    for (auto n: l) {
        num += n;
    }
    return num;
}

std::string dt(const std::vector<std::string> &strs) {
    std::string a;
    for (const auto &str: strs) {
        a += str;
    }
    std::cout << a << std::endl;
    return a;
}

std::string dt(const std::vector<std::vector<std::string>> &strss) {
    std::string a;
    for (const auto &strs: strss) {
        for (const auto &str: strs) {
            a += str;
        }
    }
    std::cout << a << std::endl;
    return a;
}

double findThreshold(const std::vector<Data> &data, const std::string &attribute) {
    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest();

    for (const Data &instance: data) {
        double value = std::get<int>(instance.attributes.at(attribute));
        if (value < minVal) {
            minVal = value;
        }
        if (value > maxVal) {
            maxVal = value;
        }
    }

    // 这里可以根据具体需求选择阈值的计算方式，例如取平均值、中位数等
    double threshold = (minVal + maxVal) / 2;

    // 返回阈值
    return threshold;
}

//计算数据集的熵
double calculateEntropy(const std::vector<Data> &data) {
    std::map<std::string, int> classCounts;
    for (const Data &instance: data) {
        classCounts[instance.class_label]++;
    }

    double entropy = 0.0;
    size_t dataSize = data.size();
    for (const auto &pair: classCounts) {
        double probability = static_cast<double>(pair.second) / static_cast<double>(dataSize);
        entropy -= probability * std::log2(probability);
    }

    return entropy;
}

//根据属性值将数据集划分为子集
std::map<std::string, std::vector<Data>> splitData_str(const std::vector<Data> &data,
                                                       const std::string &attribute) {
    std::map<std::string, std::vector<Data>> subsets;

    for (const Data &instance: data) {
        subsets[std::get<std::string>(instance.attributes.at(attribute))].push_back(instance);
    }

    return subsets;
}

std::map<std::string, std::vector<Data>>
splitData_num(const std::vector<Data> &data, const std::string &attribute, double threshold) {
    std::vector<Data> lessEqualData; // 属性值小于等于阈值的数据集
    std::vector<Data> greaterData;   // 属性值大于阈值的数据集
    std::map<std::string, std::vector<Data>> subsets;
    for (const Data &instance: data) {
        double attributeValue = std::get<int>(instance.attributes.at(attribute));
        if (attributeValue <= threshold) {
            lessEqualData.push_back(instance);
        } else {
            greaterData.push_back(instance);
        }
    }
    subsets["g"] = greaterData;
    subsets["le"] = lessEqualData;
    return subsets;
}

std::map<std::string, std::vector<Data>> splitData(const std::vector<Data> &data,
                                                   const std::string &attribute, double threshold) {
    return std::holds_alternative<std::string>(data[0].attributes.at(attribute)) ? splitData_str(data, attribute)
                                                                                 : splitData_num(data, attribute,
                                                                                                 threshold);
}

void calculateInfoGainForNumericAttribute(const std::vector<Data> &data, const std::string &attribute, double &infoGain,
                                          double &threshold) {
    // 这里是一个示例，你可以根据实际需求选择不同的方法来计算信息增益和选择最佳阈值
    // 例如，使用基于信息熵的方法来计算信息增益和选择最佳阈值
    // 注意：该示例仅处理数值属性，不处理其他类型的属性

    double bestInfoGain = 0.0;
    double bestThreshold = 0.0;

    // 获取属性的所有不同取值
    std::set<double> uniqueValues;
    for (const Data &instance: data) {
        uniqueValues.insert(std::get<double>(instance.attributes.at(attribute)));
    }

    // 尝试每个可能的阈值，并计算信息增益
    for (double value: uniqueValues) {
        std::vector<Data> lessEqualData; // 属性值小于等于阈值的数据集
        std::vector<Data> greaterData;   // 属性值大于阈值的数据集

        for (const Data &instance: data) {
            double attributeValue = std::get<double>(instance.attributes.at(attribute));
            if (attributeValue <= value) {
                lessEqualData.push_back(instance);
            } else {
                greaterData.push_back(instance);
            }
        }

        double pLessEqual = static_cast<double>(lessEqualData.size()) / data.size();
        double pGreater = static_cast<double>(greaterData.size()) / data.size();

        infoGain = calculateEntropy(data) -
                   (pLessEqual * calculateEntropy(lessEqualData) + pGreater * calculateEntropy(greaterData));

        if (infoGain > bestInfoGain) {
            bestInfoGain = infoGain;
            bestThreshold = value;
        }
    }

    infoGain = bestInfoGain;
    threshold = bestThreshold;
}

double calculateInformationGain(const std::vector<Data> &data, const std::string &attribute) {

    double entropy = calculateEntropy(data);
    double informationGain = entropy;
    int totalInstances = data.size();

    // 如果属性是数值类型
    if (std::holds_alternative<int>(data[0].attributes.at(attribute))) {
        std::map<std::string, std::vector<Data>> attributeValues;
        double threshold = findThreshold(data, attribute);

        for (const Data &instance: data) {
            if (std::get<int>(instance.attributes.at(attribute)) > threshold) {
                attributeValues["g"].push_back(instance);
            } else {
                attributeValues["le"].push_back(instance);
            }
        }
        for (const auto &pair: attributeValues) {
            const std::vector<Data> &subset = pair.second;
            double subsetEntropy = calculateEntropy(subset);
            double subsetProbability = static_cast<double>(subset.size()) / totalInstances;
            informationGain -= subsetProbability * subsetEntropy;
        }
    } else if (std::holds_alternative<double>(data[0].attributes.at(attribute))) {

    } else {
        std::map<std::string, std::vector<Data>> attributeValues;
        for (const Data &instance: data) {
            attributeValues[std::get<std::string>(instance.attributes.at(attribute))].push_back(instance);
        }
        for (const auto &pair: attributeValues) {
            const std::vector<Data> &subset = pair.second;
            double subsetEntropy = calculateEntropy(subset);
            double subsetProbability = static_cast<double>(subset.size()) / totalInstances;
            informationGain -= subsetProbability * subsetEntropy;
        }
    }

    return informationGain;
}


std::string selectBestAttribute(const std::vector<Data> &data, const std::set<std::string> &attributes) {
    double maxInformationGain = -std::numeric_limits<double>::infinity();
    std::string bestAttribute;
    for (const std::string &attribute: attributes) {
        double informationGain = calculateInformationGain(data, attribute);
        if (informationGain > maxInformationGain) {
            maxInformationGain = informationGain;
            bestAttribute = attribute;
        }
    }
    return bestAttribute;
}

std::pair<bool, std::string> shouldStop(const std::vector<Data> &data, const std::set<std::string>& attributes) {
//    assert(!data.empty());
    if(data.empty()) {
        return std::make_pair(true, "nullptr");
    }
    std::string classification = data[0].class_label;
    if(attributes.empty()) {
        std::map<std::string, int> attr;
        for (const Data &instance: data) {
            attr[instance.class_label]++;
        }
        int maxnum = -1;
        for(const auto& a: attr) {
            if(a.second > maxnum) {
                maxnum = a.second;
                classification = a.first;
            }
        }
    } else {
        // 如果数据集为空或者最终分类的属性都相同，则停止构建决策树
        for (const Data &instance: data) {
            if (instance.class_label != classification) {
                return std::make_pair(false, "");
            }
        }
//        std::cout << data[0].class_label <<std::endl;
    }

    return std::make_pair(true, classification);
}

//构建决策树
DecisionTreeNode *buildDecisionTree(const std::vector<Data> &data, std::set<std::string> attributes) {
    // 创建节点
    auto *node = new DecisionTreeNode();
    auto [stop, attr] = shouldStop(data, attributes);
    // 检查停止条件
    if (stop) {
        if(attr == "nullptr") {
            return nullptr;
        }
        node->attribute = attr;
//        std::cout << "stop:\t" << node->attribute << std::endl;
        node->type = attr;
        return node;
    }

    // 选择最佳属性
    std::string bestAttribute = selectBestAttribute(data, attributes);
    node->attribute = bestAttribute;
    node->type = "str";

    double threshold = .0;
    if (std::holds_alternative<int>(data[0].attributes.at(bestAttribute))) {
        threshold = findThreshold(data, bestAttribute);
        node->type = threshold;
    }
    attributes.erase(bestAttribute);
    // 划分数据集
    std::map<std::string, std::vector<Data>> subsets = splitData(data, bestAttribute, threshold);

    // 递归构建子节点
    for (const auto &pair: subsets) {
        DecisionTreeNode *childNode = buildDecisionTree(pair.second, attributes);
        if(childNode == nullptr) {
            continue;
        }
        node->branches[pair.first] = childNode;
    }

    return node;
}


std::vector<Data> convertToData(const std::vector<std::vector<std::variant<int, double, std::string>>> &inputData,
                                const std::vector<std::string> &attributes) {
    std::vector<Data> outputData;

    for (const auto &row: inputData) {
        Data data;

        // 遍历每个属性值，并添加到 data.attributes 中
        for (size_t i = 0; i < row.size(); i++) {
            const std::string &attributeName = attributes[i];
            data.attributes[attributeName] = row[i];
        }
        data.class_label = std::get<std::string>(row[row.size()-1]);

        // 添加到最终的 outputData 中
        outputData.push_back(data);
    }
//    for(int i = 0; i < 5; i ++) {
//        std::cout << outputData[i].class_label << std::endl;
//    }
    return outputData;
}

DecisionTreeNode *init(const std::vector<std::vector<std::variant<int, double, std::string>>> &inputData,
                       const std::vector<std::string> &attributes) {
//    printf("init\n");
    std::vector<Data> data = convertToData(inputData, attributes);
    std::set<std::string> set_attributes(attributes.begin(), attributes.end() - 1);
//    std::cout << "build dt\n";
    auto node = buildDecisionTree(data, set_attributes);

    return node;
}
int global_i = 0;
std::string predictImpl(const DecisionTreeNode *root,
                        const std::map<std::string, std::variant<int, double, std::string>> &instance) {
    const DecisionTreeNode *node = root;
    global_i++;
    // 递归遍历决策树节点，直到到达叶子节点
    while (!node->branches.empty()) {
        const std::string &attribute = node->attribute;
//        if(global_i > 75) {
//            std::cout << "impl: " << global_i << "\t" << attribute <<std::endl;
//        }
//        std::cout << "impl\t " << attribute <<std::endl;
        const auto &value = instance.at(attribute);
        if (std::holds_alternative<double>(node->type)) {
            double threshold = std::get<double>(node->type);

//            std::cout << std::get<int>(value) << std::endl;
            if (std::get<int>(value) <= threshold) {
                node = node->branches.find("le")->second;
//                if(global_i > 80) {
//                    std::cout << "node->type: " << global_i << "\t" << threshold << "\t" << std::get<int>(value) <<std::endl;
//                }
            } else {
                node = node->branches.find("g")->second;
            }
        } else {
//            std::cout << std::get<std::string>(value) << std::endl;
            auto it = node->branches.find(value);
            if (it != node->branches.end()) {
                node = it->second;
            } else {
//                if(global_i > 80) {
//                    std::cout << "node->attribute: " << global_i << "\t" << attribute <<std::endl;
//                }
                return node->attribute;
            }
        }
    }

    return node->attribute;
}

std::vector<std::string>
predict(const DecisionTreeNode *root, const std::vector<std::vector<std::variant<int, double, std::string>>> &inputData,
        const std::vector<std::string> &attributes) {
//    std::cout << "start predict\n";
    const DecisionTreeNode *node = root;
    std::vector<Data> testdata = convertToData(inputData, attributes);
    std::vector<std::string> ret;
    for (const auto &data: testdata) {
        ret.emplace_back(predictImpl(node, data.attributes));
    }
//    std::cout << "finish predict\n";
    return ret;
}

std::vector<std::string> dt1(const std::vector<std::vector<std::variant<int, double, std::string>>> &data,
                             const std::vector<std::vector<std::variant<int, double, std::string>>> &testData,
                             const std::vector<std::string> &attributes,
                             const std::vector<std::string> &testattributes) {
    return predict(init(data, attributes), testData, testattributes);
}


