#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/optimizer.h"
#include "../include/layers.h"
#include "../include/structure.h"
#include "../include/loss.h"
#include "../include/trainer.h"
#include "../include/device.h"

namespace py = pybind11;

PYBIND11_MODULE(_learntorch, m) {
    m.doc() = "Binding de Learntorch C++";

    // --- TENSOR ---
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init([](const std::vector<int>& shape, const std::vector<float>& data) {
            return Tensor::create(shape, data);
        }), py::arg("shape"), py::arg("data") = std::vector<float>{})
        .def_static("zeros", &Tensor::zeros)
        .def_static("ones", &Tensor::ones)
        .def_static("random", &Tensor::random, py::arg("shape"), py::arg("min_val")=-1.0f, py::arg("max_val")=1.0f)
        .def_property_readonly("shape", &Tensor::getShape)
        .def_property_readonly("strides", &Tensor::getStrides)
        .def_property_readonly("grad", &Tensor::getGrad)
        .def("backward", &Tensor::backward)
        .def("item", [](std::shared_ptr<Tensor> t) { return t->getData()[0]; }) 
        .def("data", [](std::shared_ptr<Tensor> t) { 
             return std::vector<float>(t->getData(), t->getData() + t->getSize());
        })
        .def("__add__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return a + b; })
        .def("__add__", [](std::shared_ptr<Tensor> a, float val) { return a + val; }) 
        .def("__sub__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return a - b; })
        .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return a * b; })
        .def("__mul__", [](std::shared_ptr<Tensor> a, float val) { return a * val; })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return a / b; })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, float val) { return a / val; })
        .def("__matmul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return matmul(a, b); })
        .def("slice", &Tensor::slice, py::arg("start"), py::arg("end"))
        .def("__repr__", [](const Tensor &t) {
            std::string s = "Tensor(shape=[";
            auto shape = t.getShape();
            for(size_t i=0; i<shape.size(); i++) s += (i>0?",":"") + std::to_string(shape[i]);
            s += "])"; 
            return s;
        });

    // --- BLOCKS & LAYERS ---
    py::class_<Block, std::shared_ptr<Block>>(m, "Block")
        .def("parameters", &Block::parameters)
        .def("forward", &Block::forward)
        .def("save_weights", &Block::save_weights)
        .def("load_weights", &Block::load_weights);

    py::class_<Linear, Block, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int>(), py::arg("in_features"), py::arg("out_features"))
        .def_readwrite("W", &Linear::W)
        .def_readwrite("B", &Linear::B);
    
    py::class_<ReLU, Block, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>());

    py::class_<Sigmoid, Block, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>());

    py::class_<Tanh, Block, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>());

    py::class_<Softmax, Block, std::shared_ptr<Softmax>>(m,"Softmax")
        .def(py::init<>());

    py::class_<Flatten, Block, std::shared_ptr<Flatten>>(m,"Flatten")
        .def(py::init<>());

    py::class_<Embedding, Block, std::shared_ptr<Embedding>>(m,"Embedding")
        .def(py::init<int, int>(), py::arg("vocab_size"), py::arg("vector_size"));

    py::class_<LayerNorm, Block, std::shared_ptr<LayerNorm>>(m,"LayerNorm")
        .def(py::init<int, float>(), py::arg("embed_dim"), py::arg("epsilon") = 1e-5f);

    py::class_<Conv2D, Block, std::shared_ptr<Conv2D>>(m, "Conv2D")
        .def(py::init<int, int, int, int, int>(), 
             py::arg("in_channels"), py::arg("out_channels"), 
             py::arg("kernel_size"), py::arg("stride"), py::arg("padding"))
        .def_readwrite("W", &Conv2D::W)
        .def_readwrite("B", &Conv2D::B)
        .def_readwrite("stride", &Conv2D::stride)
        .def_readwrite("padding", &Conv2D::padding);

    py::class_<SelfAttention, Block, std::shared_ptr<SelfAttention>>(m, "SelfAttention")
        .def(py::init<int>(), py::arg("dim"))
        .def_readwrite("q_proj", &SelfAttention::q_proj)
        .def_readwrite("k_proj", &SelfAttention::k_proj)
        .def_readwrite("v_proj", &SelfAttention::v_proj)
        .def_readwrite("out_proj", &SelfAttention::out_proj)
        .def_readwrite("embed_dim", &SelfAttention::embed_dim);

    // Serial (Sequential)
    py::class_<Serial, Block, std::shared_ptr<Serial>>(m, "Serial")
        .def(py::init<const std::vector<std::shared_ptr<Block>>&>());

    py::class_<Parallel, Block, std::shared_ptr<Parallel>>(m, "Parallel")
        .def(py::init<const std::vector<std::shared_ptr<Block>>&>());

    py::enum_<JoinMode>(m, "JoinMode")
        .value("SUM", JoinMode::SUM)
        .value("CONCAT", JoinMode::CONCAT)
        .export_values();

    py::class_<Join, Block, std::shared_ptr<Join>>(m, "Join")
        .def(py::init<JoinMode>(), py::arg("mode") = JoinMode::SUM);

    py::class_<Identity, Block, std::shared_ptr<Identity>>(m, "Identity")
        .def(py::init<>());

    // --- OPTIMIZADORES ---
    py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def("step", &Optimizer::step)
        .def("zero_grad", &Optimizer::zero_grad);
    
    py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float>(), 
             py::arg("params"), py::arg("lr"));

    py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float, float, float, float>(), 
             py::arg("params"), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("epsilon") = 1e-8f);

    // --- LOSSES ---
    py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss")
        .def("forward", &Loss::forward);
    
    py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init<>());
    
    py::class_<CrossEntropy, Loss, std::shared_ptr<CrossEntropy>>(m, "CrossEntropy")
        .def(py::init<>());

    // --- TRAINER ---
    py::class_<Trainer>(m, "Trainer")
        .def(py::init<std::shared_ptr<Block>, std::shared_ptr<Optimizer>, std::shared_ptr<Loss>>())
        .def("fit", &Trainer::fit, py::arg("x_train"), py::arg("y_train"), py::arg("x_val"), py::arg("y_val"),  py::arg("epochs"), py::arg("batch_size"), py::arg("print_every")=1);

    // --- FUNCIONES GLOBALES ---
    m.def("matmul", &matmul);
    m.def("relu", static_cast<std::shared_ptr<Tensor> (*)(std::shared_ptr<Tensor>)>(&relu));
    m.def("log", static_cast<std::shared_ptr<Tensor> (*)(std::shared_ptr<Tensor>)>(&log));
    m.def("exp", static_cast<std::shared_ptr<Tensor> (*)(std::shared_ptr<Tensor>)>(&exp));
    m.def("sqrt", static_cast<std::shared_ptr<Tensor> (*)(std::shared_ptr<Tensor>)>(&sqrt));
    m.def("sigmoid", static_cast<std::shared_ptr<Tensor> (*)(std::shared_ptr<Tensor>)>(&sigmoid));
    m.def("tanh", static_cast<std::shared_ptr<Tensor> (*)(std::shared_ptr<Tensor>)>(&tanh));
    m.def("sum", &sum, py::arg("tensor"), py::arg("dim")=0);
    m.def("max", &max, py::arg("tensor"), py::arg("dim")=0);
    m.def("argmax", &argmax, py::arg("tensor"), py::arg("dim")=0);
    m.def("transpose", &transpose_view);

    // --- BACKEND ---
    py::enum_<BackendType>(m, "BackendType")
        .value("CPU", BackendType::CPU)
        .value("CPU_OPTIMIZED", BackendType::CPU_OPTIMIZED)
        .value("GEMM_OPTIMIZED", BackendType::GEMM_OPTIMIZED)
        .export_values();

    m.def("set_backend", &Device::set_backend);
}
