#include "unit.h"
#include "../include/neuron.h"
#include <cstddef> 

using namespace std;
// Implementation of the functions defined in neuron.h

// Constructor
Neuron::Neuron(int inputs){
    // We create a uniform distribution of random numbers for the weights
    random_device rd;  
    mt19937 gen(rd());  
    uniform_real_distribution<> dist(-1.0, 1.0);
    // Set the bias to 1.0
    bias = make_shared<Unit>(1.0, "b");

    // We create an Unit for each input or weight of the neuron
    for(int i=0;i<inputs;i++){  
        double num = dist(gen);
        auto out = make_shared<Unit>(num,"w"+ to_string(i));
        weights.push_back(out);
    };

};

// Compute the weights and the inputs
shared_ptr<Unit> Neuron::forward(vector<shared_ptr<Unit>>& inputs){
    auto sum = this->bias;

    // Multiply each input with a weight and then sum everything with the bias (input * weights) + bias
    for(size_t i = 0; i < inputs.size(); i++){
        auto mult = inputs.at(i) * weights.at(i);  
        sum = sum + mult;                          
    }

    // We apply the activation function, we use relu
    return operator_tanh(sum);
}

// Return all the trainable parameters
vector<shared_ptr<Unit>> Neuron::parameters(){
    vector<shared_ptr<Unit>> output = {};
    // We add all the weights and the bias to the output
    for(auto& weight : this->weights){
        output.push_back(weight);
    }
    output.push_back(this->bias);

    return output;

}