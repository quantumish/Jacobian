//
//  pybind.cpp
//  Jacobian
//
//  Created by David Freifeld
//

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "bpnn.hpp"
namespace py = pybind11;

PYBIND11_MODULE(_jacobian, m) {
  m.doc() = "Fast machine learning in C++"; // optional module docstring
  py::enum_<Regularization>(m, "Regularization")
	  .value("L1", Regularization::L1)
	  .value("L2", Regularization::L2)	  
	  .export_values();
  py::class_<Network>(m, "Network")
	  .def(py::init<char*, int, float, float, Regularization, float, float, bool, float>())
	  .def("add_layer", &Network::add_layer, py::arg("nodes"), py::arg("name"), py::arg("activation"), py::arg("activation_deriv"))
	  .def("initialize", &Network::initialize)
      //.def("init_decay", &Network::init_decay, py::arg("type"), py::arg("a_0"), py::arg("k"))
      //.def("set_activation", &Network::set_activation)
	  .def("feedforward", &Network::feedforward)
	  .def("backpropagate", &Network::backpropagate)
	  .def("list_net", &Network::list_net)
	  .def("cost", &Network::cost)
	  .def("accuracy", &Network::accuracy)
      //.def("update_layer", &Network::update_layer, py::arg("vals"), py::arg("len"), py::arg("index"))
	  .def("next_batch", &Network::next_batch)
	  .def("train", &Network::train)
	  .def("get_acc", &Network::get_acc)
	  .def("get_cost", &Network::get_cost)
	  .def("get_val_acc", &Network::get_val_acc)
	  .def("get_val_cost", &Network::get_val_cost);
}
