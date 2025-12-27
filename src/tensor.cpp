#include "tensor.h"
#include "utils.h" 
using namespace std;


Tensor::Tensor(const vector<int>& str, float* data_param,
       const vector<shared_ptr<Tensor>> childs_param) 
       : data(nullptr), total_size(1), shape(str), childs(childs_param) {
    
    for (const auto& ptr : str) total_size *= ptr;

    if (total_size == 0) return;
    data = shared_ptr<float[]>(new float[total_size]);
    if (data_param != nullptr) {
        for (size_t i = 0; i < total_size; i++) {
            data[i] = data_param[i];
        }
    }

    strides.resize(str.size());
    strides[str.size() - 1] = 1;
    for (int i = str.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * str[i + 1];
    }
}


void Tensor::printElements(int count) const {
    cout << "Elementos del tensor" << endl;
    for (int i = 0; i < count; i++) {

        cout << "Elemento " << i << ": " << getData()[i] << endl;
    }
}

void Tensor::printShape() { 
    cout << "Shape: ("; 
    for (size_t i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i != shape.size() - 1) cout << ", ";
    }
    cout << ")" << endl;
}

void Tensor::printStrides() { 
    cout << "Strides: ("; 
    for (size_t i = 0; i < strides.size(); i++) {
        cout << strides[i];
        if (i != strides.size() - 1) cout << ", ";
    }
    cout << ")" << endl;
}


shared_ptr<Tensor> Tensor::getBatch(int index) {
    assert(getDimension() == 3 && "Tensor must have three dimensions");
    int batch_size = shape.at(0);
    int rows = shape.at(1);
    int cols = shape.at(2);
    if (index < 0 || index >= batch_size)
        throw std::out_of_range("Batch index out of range");

    auto tensor_view = make_shared<Tensor>(*this);
    tensor_view->shape = {rows, cols};
    tensor_view->offset = offset + index * strides[0];
    tensor_view->strides = {cols,1};
    tensor_view->childs = {shared_from_this()};
    return tensor_view;
}

shared_ptr<Tensor> Tensor::view_to_3d() {
    auto view = make_shared<Tensor>(*this); 
    
    if(view->getDimension() == 1){
        view->strides.push_back(0); 
        view->shape.push_back(1);
    }
    if(view->getDimension() == 2){
        view->strides.insert(view->strides.begin(), 0); 
        view->shape.insert(view->shape.begin(), 1);
    }
    return view;
}



shared_ptr<Tensor> Tensor::zeros(const vector<int>& shape) {
    size_t size = 1;
    for (const auto& dim : shape) size *= dim;
    vector<float> zero_data(size, 0.0f);
    return make_shared<Tensor>(shape, zero_data.data());
}

shared_ptr<Tensor> Tensor::ones(const vector<int>& shape) {
    size_t size = 1;
    for (const auto& dim : shape) size *= dim;
    vector<float> one_data(size, 1.0f);
    return make_shared<Tensor>(shape, one_data.data());
}


// backward functions

void build_topo(shared_ptr<Tensor> v, vector<shared_ptr<Tensor>>& topo, unordered_set<Tensor*>& visited){
    if(visited.find(v.get()) == visited.end()){
        visited.insert(v.get());
        for(auto& child : v.get()->getChilds()){
            build_topo(child, topo,visited);
        }
        topo.push_back(v);
    }

}

void Tensor::backward() {
 
    vector<shared_ptr<Tensor>> topo;
    unordered_set<Tensor*> visited;
    

    build_topo(shared_from_this(), topo, visited);
    

    this->grad = Tensor::ones(this->shape);

    cout << "\n--- ORDEN TOPOLOGICO (Backpropagation) ---" << endl;
    cout << "El orden debe ser: Nodo Final -> Intermedios -> Entradas (Hojas)" << endl;
    
 
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        shared_ptr<Tensor> t = *it;
        

        cout << "Procesando nodo: " << t.get() << " | Shape: (";
        for(size_t i=0; i<t->shape.size(); i++) {
            cout << t->shape[i] << (i < t->shape.size()-1 ? "," : "");
        }
        cout << ")" << endl;


        if (t->_backward) {
            t->_backward();
        }
        
    }
    cout << "--- FIN BACKWARD ---\n" << endl;
}

