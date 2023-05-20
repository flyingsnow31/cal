//
// Created by lizengyi on 2023/5/20.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "decision_tree/libDecisionTree.h"

namespace py = pybind11;

//namespace pybind11::detail {
//    template<> struct type_caster<DecisionTreeNode> {
//    public:
//        PYBIND11_TYPE_CASTER(DecisionTreeNode, _("DecisionTreeNode"));
//        // 将 C++ 的 DecisionTreeNode 转换为 Python 对象
//        static handle cast(const DecisionTreeNode& node, return_value_policy /* policy */, handle /* parent */) {
//            py::dict dict;
//            dict["type"] = node.type;
//            dict["attribute"] = node.attribute;
//            dict["branches"] = node.branches;
//
//            return dict.release();
//        }
//        // 将 Python 对象转换为 C++ 的 DecisionTreeNode
//        bool load(handle src, bool) {
//            if (!py::isinstance<py::dict>(src))
//                return false;
//
//            py::dict dict = py::cast<py::dict>(src);
//
//            DecisionTreeNode node;  // 声明 DecisionTreeNode 对象
//
//            if (py::handle typeObj = dict["type"]) {
//                if (py::isinstance<double>(typeObj))
//                    node.type = py::cast<double>(typeObj);
//                else if (py::isinstance<std::string>(typeObj))
//                    node.type = py::cast<std::string>(typeObj);
//                else
//                    return false;  // 类型不匹配
//            } else {
//                return false;  // 找不到 "type" 键
//            }
//
//            if (py::handle attrObj = dict["attribute"]) {
//                if (py::isinstance<std::string>(attrObj))
//                    node.attribute = py::cast<std::string>(attrObj);
//                else
//                    return false;  // 类型不匹配
//            } else {
//                return false;  // 找不到 "attribute" 键
//            }
//
//            if (py::handle branchesObj = dict["branches"]) {
//                if (py::isinstance<py::dict>(branchesObj)) {
//                    py::dict branchesDict = py::cast<py::dict>(branchesObj);
//                    for (auto entry : branchesDict) {
//                        std::variant<int, double, std::string> key;
//                        DecisionTreeNode* child = nullptr;
//
//                        // 根据你的数据结构，解析子节点的键和值
//                        // ...
//
//                        // 将解析后的键和值添加到 branches 成员变量中
//                        node.branches[key] = child;
//                    }
//                } else {
//                    return false;  // 类型不匹配
//                }
//            } else {
//                return false;  // 找不到 "branches" 键
//            }
//
//            value = std::move(node);  // 将解析后的 DecisionTreeNode 赋值给 type_caster 的 value
//
//            return true;
//        }
//    };
//}

PYBIND11_MODULE(cal, m) {
//    py::class_<DecisionTreeNode>(m, "DecisionTreeNode")
//        .def(py::init<>())
//        .def_readwrite("type", &DecisionTreeNode::type)
//        .def_readwrite("attribute", &DecisionTreeNode::attribute)
//        .def_readwrite("branches", &DecisionTreeNode::branches);
    m.def("init", &init, "init tree");
    m.def("predict", &predict, "predict");
    m.def("sum", &sum, "sum vec");
    m.def("dt", &dt1, "sum vec");
}
