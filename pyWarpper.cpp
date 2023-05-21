//
// Created by lizengyi on 2023/5/20.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "decision_tree/libDecisionTree.h"

namespace py = pybind11;

PYBIND11_MODULE(cal, m) {
    py::class_<DecisionTreeNode>(m, "DecisionTreeNode")
        .def(py::init<>())
        .def_readwrite("type", &DecisionTreeNode::type)
        .def_readwrite("attribute", &DecisionTreeNode::attribute)
        .def_readwrite("branches", &DecisionTreeNode::branches);
    m.def("dt_init", &dt_init, "init tree");
    m.def("dt_predict", &dt_predict, "predict");
    m.def("test", &test, "sum vec");
    m.def("dt", &dt, "sum vec");
}
