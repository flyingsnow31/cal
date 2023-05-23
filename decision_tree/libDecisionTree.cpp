#include "libDecisionTree.h"
#include <cmath>
#include <algorithm>
#include "set"
//#include "json/json.h"

double test(const std::vector<std::string> &l) {
    std::cout << "cal installed! input string:\n";
    for (const auto &str: l) {
        std::cout << str << " ";
    }
    std::cout << std::endl << "cal version:\t" << VERSION << std::endl;
    return VERSION;
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

std::string theMost(const std::vector<Data> &data) {
    std::string classification;
    std::map<std::string, int> attr;
    for (const Data &instance: data) {
        attr[instance.class_label]++;
    }
    int maxnum = -1;
    for (const auto &a: attr) {
        if (a.second > maxnum) {
            maxnum = a.second;
            classification = a.first;
        }
    }
    return classification;
}

std::pair<bool, std::string> shouldStop(const std::vector<Data> &data, const std::set<std::string> &attributes) {
//    assert(!data.empty());
    if (data.empty()) {
        return std::make_pair(true, "nullptr");
    }
    std::string classification = data[0].class_label;
    if (attributes.empty()) {
        classification = theMost(data);
    } else {
        // 如果数据集为空或者最终分类的属性都相同，则停止构建决策树
        for (const Data &instance: data) {
            if (instance.class_label != classification) {
                return std::make_pair(false, "");
            }
        }
    }

    return std::make_pair(true, classification);
}

double calculateAccuracy(const DecisionTreeNode *node, const std::vector<Data> &validationData);

void countClassOccurrences(const DecisionTreeNode *node, std::map<std::string, int> &classCounts) {
    // 如果节点是叶子节点，则统计目标类别
    if (node->branches.empty()) {
        classCounts[node->attribute]++;
        return;
    }

    // 递归统计子节点的目标类别
    for (const auto &pair: node->branches) {
        countClassOccurrences(pair.second, classCounts);
    }
}


std::string determineMajorityClass(const DecisionTreeNode *node) {
    std::map<std::string, int> classCounts;

    // 统计每个类别的出现次数
    countClassOccurrences(node, classCounts);

    // 找到出现次数最多的类别
    std::string majorityClass;
    int maxCount = -1;

    for (const auto &pair: classCounts) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            majorityClass = pair.first;
        }
    }

    return majorityClass;
}


void postPruning(DecisionTreeNode* &node, const std::vector<Data> &validationData) {
    // 如果节点是叶子节点，直接返回
    if (node->branches.empty()) {
        return;
    }

    // 保存当前节点的子节点
    auto originalBranches = node->branches;

    // 计算当前节点在验证数据集上的准确率
    double originalAccuracy = calculateAccuracy(node, validationData);
    auto tmp = node->attribute;
    // 将当前节点转换为叶子节点，并计算转换后的准确率
    std::string majorityClass = determineMajorityClass(node);
    node->attribute = majorityClass;
    node->branches.clear();
    double prunedAccuracy = calculateAccuracy(node, validationData);

    // 如果转换后的准确率不低于原始准确率，则进行剪枝
    if (prunedAccuracy >= originalAccuracy) {
        for (auto &pair: originalBranches) {
            delete pair.second;
        }
        originalBranches.clear();
    }
        // 否则，还原当前节点的子节点
    else {
        node->branches = originalBranches;
        node->attribute = tmp;
    }
    std::cout << "tt\n";
    // 递归处理子节点
    for (auto &pair: node->branches) {
        postPruning(pair.second, validationData);
    }
}


//构建决策树
DecisionTreeNode *buildDecisionTree(const std::vector<Data> &data, std::set<std::string> attributes) {
    // 创建节点
    auto *node = new DecisionTreeNode();
    auto [stop, attr] = shouldStop(data, attributes);
    // 检查停止条件
    if (stop) {
        if (attr == "nullptr") {
            return nullptr;
        }
        node->attribute = attr;
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
        DecisionTreeNode *childNode;
        if (pair.second.empty()) {
            childNode = new DecisionTreeNode();
            childNode->attribute = theMost(data);
            childNode->type = childNode->attribute;
        } else {
            childNode = buildDecisionTree(pair.second, attributes);
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
        data.class_label = std::get<std::string>(row[row.size() - 1]);

        outputData.push_back(data);
    }
    return outputData;
}

DecisionTreeNode *dt_init(const std::vector<std::vector<std::variant<int, double, std::string>>> &inputData,
                          const std::vector<std::string> &attributes, bool needPostPruning) {
    std::vector<Data> data = convertToData(inputData, attributes);
    std::set<std::string> set_attributes(attributes.begin(), attributes.end() - 1);
    auto node = buildDecisionTree(data, set_attributes);
    if (needPostPruning) {
        std::cout << "start prun\n";
        postPruning(node, data);
    }
    return node;
}

std::string predictImpl(const DecisionTreeNode *root,
                        const std::map<std::string, std::variant<int, double, std::string>> &instance) {
    const DecisionTreeNode *node = root;

    // 递归遍历决策树节点，直到到达叶子节点
    while (!node->branches.empty()) {
        const std::string &attribute = node->attribute;
        const auto &value = instance.at(attribute);
        if (std::holds_alternative<double>(node->type)) {

            double threshold = std::get<double>(node->type);
            if (std::get<int>(value) <= threshold) {
                node = node->branches.find("le")->second;
            } else {
                node = node->branches.find("g")->second;
            }
        } else {
            auto it = node->branches.find(value);
            if (it != node->branches.end()) {
                node = it->second;
            } else {
                return node->attribute;
            }
        }
    }

    return node->attribute;
}

std::vector<std::string>
dt_predict(const DecisionTreeNode *root,
           const std::vector<std::vector<std::variant<int, double, std::string>>> &inputData,
           const std::vector<std::string> &attributes) {
    printf("start predict\n");
    const DecisionTreeNode *node = root;
    std::vector<Data> testdata = convertToData(inputData, attributes);

    printf("start predict1\n");
    std::vector<std::string> ret;
    ret.reserve(testdata.size());
    for (const auto &data: testdata) {
        ret.emplace_back(predictImpl(node, data.attributes));
    }
    return ret;
}

std::vector<std::string> dt(const std::vector<std::vector<std::variant<int, double, std::string>>> &train_data,
                            const std::vector<std::vector<std::variant<int, double, std::string>>> &test_data,
                            const std::vector<std::string> &train_attributes,
                            const std::vector<std::string> &test_attributes) {
    return dt_predict(dt_init(train_data, train_attributes), test_data, test_attributes);
}

double calculateAccuracy(const DecisionTreeNode *node, const std::vector<Data> &validationData) {
    int correctPredictions = 0;

    for (const Data &instance: validationData) {
        if (predictImpl(node, instance.attributes) == instance.class_label) {
            correctPredictions++;
        }
    }

    double accuracy = static_cast<double>(correctPredictions) / validationData.size();
    return accuracy;
}

//void traverseNode(DecisionTreeNode* node, Json::Value& value) {
//
//}
//
//std::string printTree(DecisionTreeNode* node) {
//    Json::Value root(Json::objectValue);
//    traverseNode(node, root);
//}

