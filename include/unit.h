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

// Represents a structure of a single value which we can operate with and do backpropagation
struct Unit :  std::enable_shared_from_this<Unit>{

    // Data that will be stored within a Unit
    public:
        double data; // The current value
        std::vector<std::shared_ptr<Unit>> children; // Units that will be childrens
        std::string label; // A label just for naming Units
        double grad; // The value of the gradient
        std::function<void()> backward; // A backward function that is called during backpropagation

    // Unit contructor with childrens
    Unit(double d, const std::vector<std::shared_ptr<Unit>>& c = {}, const std::string& l = "");

    // Basic Unit constructor without childrens
    Unit(double d, const std::string& l);

    // This functions prints a clean message with Unit info
    std::string toString() const;

    // The main responsable of propagate the gradient backward from the Unit
    // where the function is called
    void retropropagate();
};