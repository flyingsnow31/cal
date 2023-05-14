//
// Created by lizengyi on 2023/5/9.
//
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "decision_tree/libDecisionTree.h"
#include "iostream"
PyObject *Wrappsum(PyObject *self, PyObject *args) {
    PyObject *list;
    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }
    Py_ssize_t size = PyList_Size(list);
    std::vector<int> tmp(size);
    for(int i = 0; i < size; ++i) {
        PyObject *t = PyList_GetItem(list, i);
        tmp.emplace_back(PyLong_AsLong(t));
    }
    return Py_BuildValue("i", sum(tmp));
}

PyObject *Wrappdt(PyObject *self, PyObject *args) {
    PyObject *list;
    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }
    std::cout << "1\n";
    Py_ssize_t size = PyList_Size(list);
    std::vector<std::string> tmp(size);
    for (Py_ssize_t i = 0; i < PyList_Size(list); ++i) {
        PyObject* py_string = PyList_GetItem(list, i);
        std::cout << PyUnicode_Check(py_string) << std::endl;
        PyObject* py_string_utf8 = PyUnicode_AsUTF8String(py_string);
        const char* str = PyBytes_AsString(py_string_utf8);
        tmp.emplace_back(str);
    }
    return Py_BuildValue("s", dt(tmp).c_str());
}
std::string process_string_array(PyObject* pyObject) {
    Py_ssize_t size = PyList_Size(pyObject);
    std::vector<std::vector<std::string>> table(size);
    for (Py_ssize_t i = 0; i < PyList_Size(pyObject); ++i) {
        PyObject* py_strings = PyList_GetItem(pyObject, i);
        std::vector<std::string> row(PyList_Size(py_strings));
        for (Py_ssize_t j = 0; j < PyList_Size(py_strings); ++j) {
            PyObject* py_string = PyList_GetItem(py_strings, j);
            PyObject* py_string_utf8 = PyUnicode_AsUTF8String(py_string);
            const char* str = PyBytes_AsString(py_string_utf8);
            row.emplace_back(str);
        }
        table.emplace_back(row);
    }
    return dt(table);
}
PyObject *Wrappdt2(PyObject *self, PyObject *args) {
    PyObject *list;
    if (!PyArg_ParseTuple(args, "O", &list)) {
        return NULL;
    }

    return Py_BuildValue("s", process_string_array(list).c_str());
}

std::vector<std::string> py2vec(PyObject *list) {
    Py_ssize_t size = PyList_Size(list);
    std::vector<std::string> tmp(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* py_string = PyList_GetItem(list, i);
        PyObject* py_string_utf8 = PyUnicode_AsUTF8String(py_string);
        const char* str = PyBytes_AsString(py_string_utf8);
        tmp[i]=str;
    }
    return tmp;
}

std::vector<std::vector<std::string>> py2vec2(PyObject *pyObject) {
    Py_ssize_t size = PyList_Size(pyObject);
    std::vector<std::vector<std::string>> table(size);
    for (Py_ssize_t i = 0; i < PyList_Size(pyObject); ++i) {
        PyObject* py_strings = PyList_GetItem(pyObject, i);
        std::vector<std::string> row(PyList_Size(py_strings));
        for (Py_ssize_t j = 0; j < PyList_Size(py_strings); ++j) {
            PyObject* py_string = PyList_GetItem(py_strings, j);
            PyObject* py_string_utf8 = PyUnicode_AsUTF8String(py_string);
            const char* str = PyBytes_AsString(py_string_utf8);
            row[j]=str;
        }
        table[i] = row;
    }
    return table;
}

PyObject *WrapGauss(PyObject *self, PyObject *args) {
    PyObject *train, *test;
    const char *filename;
    if (!PyArg_ParseTuple(args, "OOs", &train, &test, &filename)) {
        std::cout << "wrong arg\n";
        return NULL;
    }
    gauss(py2vec2(train), py2vec2(test), filename);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef sum_methods[] = {
        {"sum", Wrappsum, METH_VARARGS, "sum"},
        {"dt", Wrappdt, METH_VARARGS, "dt"},
        {"dt2", Wrappdt2, METH_VARARGS, "dt2"},
        {"gauss", WrapGauss, METH_VARARGS, "gauss"},
        {NULL, NULL}
};
static PyModuleDef summod = {
        PyModuleDef_HEAD_INIT,
        "cal",
        "",
        -1,
        sum_methods
};

PyMODINIT_FUNC PyInit_cal(void) {
    PyObject *m;

    m = PyModule_Create(&summod);
    if (m == NULL)
        return NULL;

    return m;
}