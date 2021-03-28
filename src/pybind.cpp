//
//  pybind.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
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
	py::class_<Network>(m, "Network")
		.def(py::init<char *, int, float, float, Regularization, float,
			      float, bool, float>(),
		     py::arg("path"), py::arg("batch"), py::arg("learn_rate"),
		     py::arg("bias_rate"), py::arg("regularization"),
		     py::arg("lambda"), py::arg("ratio"),
		     py::arg("early_exit") = true, py::arg("cutoff") = 0)
		.def("add_layer", &Network::add_layer, py::arg("nodes"),
		     py::arg("name"), py::arg("activation"),
		     py::arg("activation_deriv"))
		.def("initialize", &Network::initialize)
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
}
