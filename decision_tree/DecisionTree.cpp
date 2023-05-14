//
// Created by lizengyi on 2023/5/10.
//
#include <utility>
#include <cassert>
#include "libDecisionTree.h"

using namespace std;

class Table {
public:
    vector<string> attrName;
    vector<vector<string> > data;

    vector<vector<string> > attrValueList;

    void setTable(const std::vector<std::vector<std::string>> &table) {
        assert(table.size() > 0);
        attrName = table.at(0);
        data.resize(table.size() - 1);
        for(size_t i = 1; i < table.size(); ++i) {
            data[i - 1] = table[i];
        }
    }

    void extractAttrValue() {
        attrValueList.resize(attrName.size());
        for (size_t j = 0; j < attrName.size(); j++) {
            map<string, int> value;
            for (auto & i : data) {
                value[i[j]] = 1;
            }

            for (auto & iter : value) {
                attrValueList[j].push_back(iter.first);
            }
        }
    }
};

class Node {
public:
    int criteriaAttrIndex{};
    string attrValue;

    int treeIndex{};
    bool isLeaf;
    string label;

    vector<int> children;

    Node() {
        isLeaf = false;
    }
};

class DecisionTree {
public:
    Table initialTable;
    vector<Node> tree;

    explicit DecisionTree(Table table) {
        initialTable = std::move(table);
        initialTable.extractAttrValue();
        Node root;
        root.treeIndex = 0;
        tree.push_back(root);

        run(initialTable, 0);
        printTree(0, "");

        cout << "<-- finish generating decision tree -->" << endl << endl;
    }

    string guess(vector<string> row) {
        string label;
        int leafNode = dfs(row, 0);
        if (leafNode == -1) {
            return "dfs failed";
        }
        label = tree[leafNode].label;
        return label;
    }

    int dfs(vector<string> &row, int here) {
        if (tree[here].isLeaf) {
            return here;
        }

        int criteriaAttrIndex = tree[here].criteriaAttrIndex;

        for (size_t i = 0; i < tree[here].children.size(); i++) {
            int next = tree[here].children[i];

            if (row[criteriaAttrIndex] == tree[next].attrValue) {
                return dfs(row, next);
            }
        }
        return -1;
    }

    void run(Table table, int nodeIndex) {
        if (isLeafNode(table)) {
            tree[nodeIndex].isLeaf = true;
            tree[nodeIndex].label = table.data.back().back();
            return;
        }

        int selectedAttrIndex = getSelectedAttribute(table);

        map<string, vector<int> > attrValueMap;
        for (size_t i = 0; i < table.data.size(); i++) {
            attrValueMap[table.data[i][selectedAttrIndex]].push_back(i);
        }

        tree[nodeIndex].criteriaAttrIndex = selectedAttrIndex;

        pair<string, int> majority = getMajorityLabel(table);
        if ((double) majority.second / table.data.size() > 0.8) {
            tree[nodeIndex].isLeaf = true;
            tree[nodeIndex].label = majority.first;
            return;
        }

        for (size_t i = 0; i < initialTable.attrValueList[selectedAttrIndex].size(); i++) {
            string attrValue = initialTable.attrValueList[selectedAttrIndex][i];

            Table nextTable;
            vector<int> candi = attrValueMap[attrValue];
            for (size_t j : candi) {
                nextTable.data.push_back(table.data[j]);
            }

            Node nextNode;
            nextNode.attrValue = attrValue;
            nextNode.treeIndex = (int) tree.size();
            tree[nodeIndex].children.push_back(nextNode.treeIndex);
            tree.push_back(nextNode);

            // for empty table
            if (nextTable.data.empty()) {
                nextNode.isLeaf = true;
                nextNode.label = getMajorityLabel(table).first;
                tree[nextNode.treeIndex] = nextNode;
            } else {
                run(nextTable, nextNode.treeIndex);
            }
        }
    }

    static double getEstimatedError(double f, int N) {
        double z = 0.69;
        if (N == 0) {
            cout << ":: getEstimatedError :: N is zero" << endl;
            exit(0);
        }
        return (f + z * z / (2 * N) + z * sqrt(f / N - f * f / N + z * z / (4 * N * N))) / (1 + z * z / N);
    }

    static pair<string, int> getMajorityLabel(Table table) {
        string majorLabel;
        int majorCount = 0;

        map<string, int> labelCount;
        for (auto & i : table.data) {
            labelCount[i.back()]++;

            if (labelCount[i.back()] > majorCount) {
                majorCount = labelCount[i.back()];
                majorLabel = i.back();
            }
        }

        return {majorLabel, majorCount};
    }


    static bool isLeafNode(const Table& table) {
        for (size_t i = 1; i < table.data.size(); i++) {
            if (table.data[0].back() != table.data[i].back()) {
                return false;
            }
        }
        return true;
    }

    int getSelectedAttribute(const Table& table) {
        int maxAttrIndex = -1;
        double maxAttrValue = 0.0;

        // except label
        for (size_t i = 0; i < initialTable.attrName.size() - 1; i++) {
            if (maxAttrValue < getGainRatio(table, i)) {
                maxAttrValue = getGainRatio(table, i);
                maxAttrIndex = i;
            }
        }

        return maxAttrIndex;
    }

    double getGainRatio(const Table& table, int attrIndex) {
        return getGain(table, attrIndex) / getSplitInfoAttrD(table, attrIndex);
    }

    static double getInfoD(Table table) {
        double ret = 0.0;

        int itemCount = (int) table.data.size();
        map<string, int> labelCount;

        for (auto & i : table.data) {
            labelCount[i.back()]++;
        }

        for (auto & iter : labelCount) {
            double p = (double) iter.second / itemCount;

            ret += -1.0 * p * log(p) / log(2);
        }

        return ret;
    }

    static double getInfoAttrD(Table table, int attrIndex) {
        double ret = 0.0;
        int itemCount = (int) table.data.size();

        map<string, vector<int> > attrValueMap;
        for (size_t i = 0; i < table.data.size(); i++) {
            attrValueMap[table.data[i][attrIndex]].push_back(i);
        }

        for (auto & iter : attrValueMap) {
            Table nextTable;
            for (size_t i : iter.second) {
                nextTable.data.push_back(table.data[i]);
            }
            int nextItemCount = (int) nextTable.data.size();

            ret += (double) nextItemCount / itemCount * getInfoD(nextTable);
        }

        return ret;
    }

    static double getGain(const Table& table, int attrIndex) {
        return getInfoD(table) - getInfoAttrD(table, attrIndex);
    }

    static double getSplitInfoAttrD(const Table& table, int attrIndex) {
        double ret = 0.0;

        int itemCount = (int) table.data.size();

        map<string, vector<int> > attrValueMap;
        for (size_t i = 0; i < table.data.size(); i++) {
            attrValueMap[table.data[i][attrIndex]].push_back(i);
        }

        for (auto & iter : attrValueMap) {
            Table nextTable;
            for (size_t i : iter.second) {
                nextTable.data.push_back(table.data[i]);
            }
            int nextItemCount = (int) nextTable.data.size();

            double d = (double) nextItemCount / itemCount;
            ret += -1.0 * d * log(d) / log(2);
        }

        return ret;
    }

    /*
     * Enumerates through all the nodes of the tree and prints all the branches
     */
    void printTree(int nodeIndex, const string &branch) {
        if (tree[nodeIndex].isLeaf)
            cout << branch << "Label: " << tree[nodeIndex].label << "\n";

        for (size_t i = 0; i < tree[nodeIndex].children.size(); i++) {
            int childIndex = tree[nodeIndex].children[i];

            string attributeName = initialTable.attrName[tree[nodeIndex].criteriaAttrIndex];
            string attributeValue = tree[childIndex].attrValue;

            printTree(childIndex, branch + attributeName + " = " + attributeValue + ", ");
        }
    }
};


class InputReader {
private:
    ifstream fin;
    Table table;
public:
    explicit InputReader(const string &filename) {
        fin.open(filename);
        if (!fin) {
            cout << filename << " file could not be opened\n";
            exit(0);
        }
        parse();
    }

    void parse() {
        string str;
        bool isAttrName = true;
        while (!getline(fin, str).eof()) {
            vector<string> row;
            int pre = 0;
            for (size_t i = 0; i < str.size(); i++) {
                if (str[i] == ',') {
                    string col = str.substr(pre, i - pre);

                    row.push_back(col);
                    pre = i + 1;
                }
            }
            string col = str.substr(pre, str.size() - pre - 1);
            row.push_back(col);

            if (isAttrName) {
                table.attrName = row;
                isAttrName = false;
            } else {
                table.data.push_back(row);
            }
        }
    }

    Table getTable() {
        return table;
    }
};

class OutputPrinter {
private:
    ofstream fout;
public:
    explicit OutputPrinter(const string &filename) {
        fout.open(filename);
        if (!fout) {
            cout << filename << " file could not be opened\n";
            exit(0);
        }
    }

    static string joinByTab(vector<string> row) {
        string ret;
        for (size_t i = 0; i < row.size(); i++) {
            ret += row[i];
            if (i != row.size() - 1) {
                ret += ',';
            }
        }
        return ret;
    }

    void addLine(const string &str) {
        fout << str << endl;
    }
};

#include "../utils.h"

void gauss(const std::vector<std::vector<std::string>> &train,
           const std::vector<std::vector<std::string>> &test, const std::string &filename) {
    Table train_table, test_table;

    train_table.setTable(train);
    test_table.setTable(test);

    std::cout << "start build tree\n";

    DecisionTree decisionTree(train_table);

    OutputPrinter outputPrinter(filename);
    outputPrinter.addLine(OutputPrinter::joinByTab(train_table.attrName));

    for (const auto &i: test_table.data) {
        vector<string> result = i;
        result.push_back(decisionTree.guess(i));
        outputPrinter.addLine(OutputPrinter::joinByTab(result));
    }
}

//int main(int argc, const char * argv[]) {
//    if(argc!=4) {
//        cout << "Please follow this format. dt.exe [train.txt] [test.txt] [result.txt]";
//        return 0;
//    }
//
//    string trainFileName = argv[1];
//    InputReader trainInputReader(trainFileName);
//    DecisionTree decisionTree(trainInputReader.getTable());
//
//    string testFileName = argv[2];
//    InputReader testInputReader(testFileName);
//    Table test = testInputReader.getTable();
//
//    string resultFileName = argv[3];
//    OutputPrinter outputPrinter(resultFileName);
//    outputPrinter.addLine(outputPrinter.joinByTab(test.attrName));
//    for(int i=0;i < test.data.size(); i++) {
//        vector<string> result = test.data[i];
//        result.push_back(decisionTree.guess(test.data[i]));
//        outputPrinter.addLine(outputPrinter.joinByTab(result));
//    }
//
//    /* for answer check */
//    /*
//       InputReader answerInputReader("dt_answer1.txt");
//       Table answer = answerInputReader.getTable();
//       int totalCount = (int)answer.data.size();
//       int hitCount = 0;
//       for(int i=0;i < test.data.size(); i++) {
//       if(answer.data[i].back() == decisionTree.guess(test.data[i])) {
//       hitCount++;
//       }
//       }
//       cout << "Accuracy: " << (double)hitCount/totalCount*100 << "%";
//       cout << "(" << hitCount << "/" << totalCount << ")" << endl;
//       */
//    return 0;
//}
