#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>

using namespace std;

struct Value :  std::enable_shared_from_this<Value>{

public:
    double data;
    vector<shared_ptr<Value>> children;
    string label;
    double grad;
    std::function<void()> backward;

    Value(double d, const vector<shared_ptr<Value>>& c = {}, const string& l = "") 
    : data(d), children(c), label(l), grad(0.0), backward([](){})  // inicializa data y children directamente
    {
        children.reserve(2);   // reserva capacidad para 2 elementos

    }

    Value(double d, const string& l)
    : data(d), children({}), label(l), backward([](){}) 
    {
        children.reserve(2);
        grad = 0;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Value(label=" << label << ", data=" << data << ", grad=" << grad << ")";
        return oss.str();
    }

    void backward_total(){
        std::unordered_set<Value*> visited;
        vector<Value*> topo;
        function<void (shared_ptr<Value>)> build_topo = [&](shared_ptr<Value> v) {
            if (visited.find(v.get()) == visited.end()){
                visited.insert(v.get());
                for(auto&child : v->children){
                    build_topo(child);
                }
                topo.push_back(v.get());
            };
            
        };
        build_topo(shared_from_this());
        reverse(topo.begin(), topo.end());
        grad = 1.0;

        for (auto node : topo){
            node->backward();
        }


    }


};

shared_ptr<Value> operator+(const shared_ptr<Value>& a, const shared_ptr<Value>& b) {
    auto out = make_shared<Value>(a->data + b->data, vector<shared_ptr<Value>>{a, b});
    out->backward = [a,b,out] () mutable {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
};

shared_ptr<Value> operator*(const shared_ptr<Value>& a, const shared_ptr<Value>& b) {
    auto out = make_shared<Value>(a->data * b->data, vector<shared_ptr<Value>>{a,b});
    out->backward = [a,b,out] () mutable {
        a->grad += out->grad * b->data;
        b->grad += out->grad * a->data;
    };
    return out;
};

shared_ptr<Value> operator-(const shared_ptr<Value>& a, const shared_ptr<Value>& b) {
    auto out = make_shared<Value>(a->data - b->data, vector<shared_ptr<Value>>{a,b});
    out->backward = [a,b,out] () mutable {
        a->grad += out->grad;
        b->grad += -out->grad;
    };
    return out;
};

shared_ptr<Value> operatorpow(const shared_ptr<Value>& a, const shared_ptr<Value>& b){
    auto out = make_shared<Value>(pow(a->data,b->data),
    vector<shared_ptr<Value>>{a,b});
    out->backward = [a,b,out] () mutable {
        a->grad += b->data * pow(a->data,(b->data - 1)) * out->grad;
    };
    return out;
}

shared_ptr<Value> operator/(const shared_ptr<Value>& a, const shared_ptr<Value>& b) {


    return a* operatorpow(b, make_shared<Value>(-1.0,vector<shared_ptr<Value>> {}));
}



shared_ptr<Value> relu(const shared_ptr<Value>& a) {
    double t = a->data < 0 ? 0.0 : a->data;

    auto out = make_shared<Value>(t, vector<shared_ptr<Value>>{a});
    out->backward = [a,out] () mutable {
        a->grad += (out->data > 0) ? out->grad : 0.0;
    };
    return out;
};

shared_ptr<Value> operator_tanh(const shared_ptr<Value>& a) {
    auto output_data = tanh(a->data);
    auto out = make_shared<Value>(output_data, vector<shared_ptr<Value>>{a});
    out->backward = [a,out] () mutable {
        double t2 = tanh(a->data);
        a->grad += (1 - t2*t2) * out->grad;
    };
    return out;
}



struct Neuron : std::enable_shared_from_this<Neuron> {
    public:
    vector<shared_ptr<Value>> weights;
    shared_ptr<Value> bias;
    Neuron(int inputs){
        random_device rd;  
        mt19937 gen(rd());  
        uniform_real_distribution<> dist(-1.0, 1.0);
        bias = make_shared<Value>(1.0, "b");

        
        for(int i=0;i<inputs;i++){
            double num = dist(gen);
            auto out = make_shared<Value>(num,"w"+ to_string(i));
            weights.push_back(out);
        };

    };

    shared_ptr<Value> compute_neuron(vector<shared_ptr<Value>>& inputs){
        auto sum = bias;

        for(size_t i = 0; i < inputs.size(); i++){
            auto mult = inputs.at(i) * weights.at(i);  
            sum = sum + mult;                          
        }

        return operator_tanh(sum);
    }

    vector<shared_ptr<Value>> parameters(){
        vector<shared_ptr<Value>> output = {};
        for(auto& weight : weights){
            output.push_back(weight);
        }
        output.push_back(bias);

        return output;
    
    }
};



struct Layer : std::enable_shared_from_this<Layer> {
    public:
    vector<shared_ptr<Neuron>>  neurons {};
    Layer(int inputs, int out){
        for(int i=0;i < out;  i++){
            auto neuron = make_shared<Neuron>(inputs);
            neurons.push_back(neuron);
        }

    }

    vector<shared_ptr<Value>> compute_layer(vector<shared_ptr<Value>>& inputs){
        vector<shared_ptr<Value>> out {};
        for (auto& neuron : neurons){
            auto activated = neuron->compute_neuron(inputs);
            out.push_back(activated);
        }
        
        return out;

    }

    vector<shared_ptr<Value>> parameters(){
        vector<shared_ptr<Value>> out {};
        for(auto& neuron: neurons){
            for(auto& p : neuron->parameters()){
                out.push_back(p);
                
            }
        }

        return out;
    }

};

struct MLP : std::enable_shared_from_this<MLP> {
    public:
    vector<shared_ptr<Layer>> created_layers {};
    MLP(int inputs, vector<int> outputs){
        vector<int> layers {};
        
        layers.push_back(inputs);
        for(auto& l: outputs){
            layers.push_back(l);
        }
        for(size_t i = 0; i<layers.size()-1;i++){
            auto layer = make_shared<Layer>(layers.at(i),layers.at(i+1));
            created_layers.push_back(layer);
        }
    }

    shared_ptr<Value> compute_mlp(vector<shared_ptr<Value>>& inputs){
        vector<shared_ptr<Value>> layer_output = inputs;
        for (auto& layer : created_layers){
            layer_output = layer->compute_layer(layer_output);
        }
        // asumimos que la última capa tiene al menos 1 neurona y queremos el primer (único) output
        return layer_output.at(0);
    }

    void zero_grad(){
        for(auto& p : this->parameters()){
            p->grad = 0;
        }
    }

    vector<shared_ptr<Value>> parameters(){
        vector<shared_ptr<Value>> out {};
        for(auto& l: created_layers){
            for(auto& p : l->parameters()){
                out.push_back(p); 
            }
        }
        return out;
    }

    vector<shared_ptr<Value>> fit(vector<vector<shared_ptr<Value>>> inputs,
        vector<shared_ptr<Value>> targets, int num_iter, double learning_rate){
        vector<shared_ptr<Value>> output;
        for(int i=0; i < num_iter ; i++){
            output = {};
            shared_ptr<Value> loss_mse_acum = make_shared<Value>(0.0,"loss");
            for(size_t j=0; j<targets.size();j++){
                auto o = this->compute_mlp(inputs.at(j));
                auto target = targets.at(j);
                auto loss = o - target;
                auto loss_mse = loss * loss;
                loss_mse_acum = loss_mse_acum + loss_mse;
                output.push_back(o);
            }
            loss_mse_acum = loss_mse_acum / make_shared<Value>(output.size(),vector<shared_ptr<Value>> {});
            this->zero_grad();
            loss_mse_acum->backward_total();
            for(auto& p : this->parameters()){
                p->data += -learning_rate * p->grad;
                
                
            }

            cout  << "Iteracion: " << i << " loss= " << loss_mse_acum->data << endl;
            for(auto& p:output){
                cout << p->data << endl;
            }
        }
        return output;
    }
};


int main() {
    auto n = Neuron(3);
    vector<vector<shared_ptr<Value>>> inputs = {{make_shared<Value>(3.0,"x1"),
    make_shared<Value>(5.0,"x2"),make_shared<Value>(-2.0,"x3")},
    {make_shared<Value>(5.0,"x1"),
    make_shared<Value>(-1.0,"x2"),make_shared<Value>(-6.0,"x3")}};
    vector<shared_ptr<Value>> targets = {make_shared<Value>(0),make_shared<Value>(1)};
    auto mlp = MLP(3,{4,1});
    
    auto preds = mlp.fit(inputs,targets,20,0.2);

    

}

