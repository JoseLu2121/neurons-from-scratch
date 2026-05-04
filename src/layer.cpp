#include "layers.h"

using namespace std;
// Implementation of the functions defined in layer.h

// Constructor
Layer::Layer(int inputs, int out){
    // Initialize 'out' neurons, each expecting 'inputs' connections
    for(int i = 0;i < out;  i++){
        auto neuron = make_shared<Neuron>(inputs);
        neurons.push_back(neuron);
    }
}

// We compute each neuron
vector<shared_ptr<Unit>> Layer::forward(vector<shared_ptr<Unit>>& inputs){
    vector<shared_ptr<Unit>> out {};
    // Call the forward function of every neuron in the layer
    for (auto& neuron : this->neurons){
        auto activated = neuron->forward(inputs);
        out.push_back(activated);
    }
    
    return out;
}

// Return the trainable parameters of the layer
vector<shared_ptr<Unit>> Layer::parameters(){
    vector<shared_ptr<Unit>> out {};
    // The parameters will the weights and bias of each neuron
    for(auto& neuron: this->neurons){
        for(auto& p : neuron->parameters()){
            out.push_back(p);
            
        }
    }

    return out;
}
