//
//  pybind.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "bpnn.hpp"
#include "utils.hpp"
namespace py = pybind11;

PYBIND11_MODULE(_jacobian, m)
{
    m.doc() = "Fast machine learning in C++"; // optional module docstring
    py::enum_<Regularization>(m, "Regularization")
        .value("L1", Regularization::L1)
        .value("L2", Regularization::L2)
        .export_values();
    py::class_<Layer>(m, "Layer")
        .def(py::init<int, int>());
    py::class_<Network>(m, "Network")
        .def(py::init<char *, int, float, float, Regularization, float,
                  float, bool, float>(),
             py::arg("path"), py::arg("batch"), py::arg("learn_rate"),
             py::arg("bias_rate"), py::arg("regularization"),
             py::arg("lambda"), py::arg("ratio"),
             py::arg("early_exit") = true, py::arg("cutoff") = 0)
        .def("add_layer", &Network::add_layer, py::arg("nodes"),
             py::arg("activation"), py::arg("activation_deriv"))
        .def("initialize", &Network::initialize)
        .def("init_optimizer", &Network::init_optimizer, py::arg("optimizer"))
        .def("init_decay", &Network::init_decay, py::arg("decay"))
        .def("set_activation", &Network::set_activation,
             py::arg("index"), py::arg("custom"),
             py::arg("custom_deriv"))
        .def("feedforward", &Network::feedforward)
        .def("backpropagate", &Network::backpropagate)
        .def("list_net", &Network::list_net)
        .def("cost", &Network::cost)
        .def("accuracy", &Network::accuracy)
        .def("train", &Network::train)
        .def("get_acc", &Network::get_acc)
        .def("get_cost", &Network::get_cost)
        .def("get_val_acc", &Network::get_val_acc)
        .def("get_val_cost", &Network::get_val_cost);
    m.def("linear", &linear, py::arg("x"));
    m.def("linear_deriv", &linear_deriv, py::arg("x"));
    m.def("sigmoid", &sigmoid, py::arg("x"));
    m.def("sigmoid_deriv", &sigmoid_deriv, py::arg("x"));
    m.def("momentum", &optimizers::momentum, py::arg("beta"));
    m.def("demon", &optimizers::demon, py::arg("beta"), py::arg("max_ep"));
    m.def("adam", &optimizers::adam, py::arg("beta1"), py::arg("beta2"), py::arg("epsilon"));
    m.def("adamax", &optimizers::adamax, py::arg("beta1"), py::arg("beta2"), py::arg("epsilon"));
    m.def("step", &decays::step, py::arg("a_0"), py::arg("k"));
    m.def("exponential", &decays::exponential, py::arg("a_0"), py::arg("k"));
    m.def("fractional", &decays::fractional, py::arg("a_0"), py::arg("k"));
    m.def("linear", &decays::linear, py::arg("max_ep"));
}
