//
// Created by lizengyi on 2023/5/20.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "decision_tree/libDecisionTree.h"
#include "simple_net/libNet.h"

namespace py = pybind11;

using namespace CuiQin;

PYBIND11_MODULE(cal, m) {
    py::class_<libNet>(m, "libNet")
        .def(py::init<std::vector<int>, double, std::string, int, double, std::string>())
// Add bindings for the class methods here
        .def("initNet", &libNet::initNet)
        .def("initWeights", &libNet::initWeights)
        .def("initBias", &libNet::initBias)
        .def("train", &libNet::train2)
        .def("test", &libNet::test)
        .def("save", &libNet::save)
        .def("load", &libNet::load);
    py::class_<DecisionTreeNode>(m, "DecisionTreeNode")
        .def(py::init<>())
        .def_readwrite("type", &DecisionTreeNode::type)
        .def_readwrite("attribute", &DecisionTreeNode::attribute)
        .def_readwrite("branches", &DecisionTreeNode::branches);
    m.def("dt_init", &dt_init, "init tree");
    m.def("dt_predict", &dt_predict, "predict");
    m.def("test", &test, "sum vec");
    m.def("dt", &dt, "dt");
    m.def("get_data", &get_data, "get_data");
}
