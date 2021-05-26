//
// Created by Reuben Feinman on 5/17/21.
//
#include <pybind11/pybind11.h>
#include "eigsh.cpp"
#include "eigsh_mkl.cpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eigsh", &arpack::eigsh, "eigsh forward",
          py::arg("A"),
          py::arg("largest") = true,
          py::arg("m") = 20,
          py::arg("max_iter") = 10000,
          py::arg("tol") = 1.0e-5);

    m.def("eigsh_mkl", &arpack::eigsh_mkl, "eigsh_mkl forward",
          py::arg("A"),
          py::arg("largest") = true,
          py::arg("m") = 20,
          py::arg("max_iter") = 10000,
          py::arg("tol_dps") = 5);
}