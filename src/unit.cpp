#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include "../include/unit.h"

using namespace std;
// For learning more about the Unit structure, check include/unit.h

// Unit constructor with childrens
Unit::Unit(double d, const vector<shared_ptr<Unit>>& c, const string& l) 
    : data(d),    // Assign the value passed as parameter to the data attribute
    children(c),  // Set the children attribute with the parameter c
    label(l),    // We stablish l as the label of the Unit
    grad(0.0),   // The initial gradient is 0.0
    backward([](){}) // The backward function is empty at first
    {
        // Empty
    }

// Same as the previous constructor but without childrens
Unit::Unit(double d, const string& l)   
: data(d), 
  grad(0.0), 
  children({}), // We set the children list as empty
  label(l), 
  backward([](){}) 
    {
        // Empty
    }

// Basic toString functions to print a clean message of an Unit info
std::string Unit::toString() const {
    std::ostringstream oss;
    oss << "Unit(label=" << label << ", data=" << data << ", grad=" << grad << ")";
    return oss.str();
}

// Propagate the gradient backwards from the Unit where the function is called
void Unit::retropropagate(){
    std::unordered_set<Unit*> visited; // Set of Unit that have been visited
    vector<Unit*> topo;                // Topological order of the Units visited

    // Recursive function responsable of filling the topo variable
    function<void (shared_ptr<Unit>)> build_topo = [&](shared_ptr<Unit> v) {
        // We check if the current Unit have not been visited
        if (visited.find(v.get()) == visited.end()){
            // If so, we insert the Unit to the visited set
            visited.insert(v.get());
            // We iterate the childrens and call recursively the function
            for(auto&child : v->children){
                build_topo(child);
            }
            // Once we check every Unit in the graph, we add all of them one after another
            topo.push_back(v.get());
        };
        
    };

    // We call the build_topo function for creating the topological order
    build_topo(shared_from_this());
    // We reverse the topo order so we have the parents first
    reverse(topo.begin(), topo.end());
    // We set the current Unit (the first one in the reverse topo) grad to 1.0
    // so the chain rule is applied correctly
    this->grad = 1.0;

    // We propagate the gradient, calling each Unit backward function
    for (auto node : topo){
        node->backward();
    }

}

