#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include "unit.h"
#include "ops.h"

enum class ActivationFunction {
    ReLu,
    Tanh
};

// Single neuron class
struct Neuron : std::enable_shared_from_this<Neuron> {
    public:
    std::vector<std::shared_ptr<Unit>> weights; // weights of the neuron
    std::shared_ptr<Unit> bias;            // bias of the neuron

    // Constructor
    Neuron(int inputs);

    // Forward function to compute the weights with the inputs
    std::shared_ptr<Unit> forward(std::vector<std::shared_ptr<Unit>>& inputs);

    // Return all the trainable parameters of the neuron
    std::vector<std::shared_ptr<Unit>> parameters();
};