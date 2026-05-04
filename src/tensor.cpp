#include "tensor.h"
#include "utils.h" 
#include "types.h"
#include "device.h"
#include "backend.h"
#include <fstream>
#include <stdexcept>

using namespace std;

// ====================
// Constructors & Factory
// ====================

shared_ptr<Tensor> Tensor::create(const vector<int>& shape_param, const vector<float>& data_param,
                                  const vector<shared_ptr<Tensor>>& parents_param) {
    return shared_ptr<Tensor>(new Tensor(shape_param, data_param, parents_param));
}

Tensor::Tensor(const vector<int>& shape_param, const vector<float>& data_param,
       const vector<shared_ptr<Tensor>>& parents_param) 
      // Setting the params given to the attributes of the tensor
    : data(nullptr), 
      shape(shape_param), 
      total_size(element_vector_product(shape_param)), 
      parents(parents_param) {
    
    // If the size is 0 we return an empty tensor
    if (total_size == 0) return;

    // The data is a pointer to a float array with tensor size
    this->data = shared_ptr<float[]>(new float[total_size]);
    
    // We initialize the data with the param given
    if (!data_param.empty()) {
        if (data_param.size() != total_size) {
            throw std::invalid_argument("The data given doesn't match the shape of the tensor");
        }

        for (size_t i = 0; i < total_size; i++) {
            data[i] = data_param[i];
        }

    }else{
        // If we have no data given, we set the tensor data to zero
        for (size_t i = 0; i < total_size; i++) {
            data[i] = 0.0f;
        }
    }

    // We initialize the Strides vector safely avoiding segfaults if shape is empty
    if (!shape.empty()) {
        strides.resize(shape.size());
        // The last one is always 1
        strides.back() = 1;
        // Each element is himself times the previous element from the right
        for (int i = (int)shape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
}

// Copy or view constructor
Tensor::Tensor(const Tensor& other) 
    : data(other.data),       
      shape(other.shape),     
      strides(other.strides), 
      total_size(other.total_size),
      grad(nullptr), // We avoid issues with the original grad tensor
      parents({}){}


// ====================
// Utils for debugging
// ====================

void Tensor::printElements(int count) const {
    cout << "Elementos del tensor" << endl;
    for (int i = 0; i < count; i++) {
        cout << "Elemento " << i << ": " << getData()[i] << endl;
    }
}

void Tensor::printShape() const{ 
    cout << "Shape: ("; 
    for (size_t i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i != shape.size() - 1) cout << ", ";
    }
    cout << ")" << endl;
}

void Tensor::printStrides() const { 
    cout << "Strides: ("; 
    for (size_t i = 0; i < strides.size(); i++) {
        cout << strides[i];
        if (i != strides.size() - 1) cout << ", ";
    }
    cout << ")" << endl;
}

void Tensor::info(int max_size) const {
    cout << "Tensor Info:" << endl;
    cout << "  Shape: (";
    for (size_t i = 0; i < shape.size(); i++) 
        cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    cout << ")" << endl;
    
    cout << "  Strides: (";
    for (size_t i = 0; i < strides.size(); i++) 
        cout << strides[i] << (i < strides.size() - 1 ? ", " : "");
    cout << ")" << endl;

    // We only print the elements if the tensor is small 
    if (total_size <= (size_t)max_size) {
        cout << "  Data: [ ";
        for (int i = 0; i < max_size; i++) cout << data[i] << " ";
        cout << "]" << endl;
    } else {
        cout << "  Data: [ ... " << total_size << " elementos ... ]" << endl;
    }
    cout << "-------------------------" << endl;
}

// ===================
// Size and view manipulation
// ===================

shared_ptr<Tensor> Tensor::reshape(const vector<int>& new_shape) {
    auto view = std::shared_ptr<Tensor>(new Tensor(*this));
    size_t new_total_size = element_vector_product(new_shape);
    if (new_total_size != this->total_size) throw std::runtime_error("The new tensor shape needs to have " + 
        std::to_string(this->total_size) + " elements");

    view->shape = new_shape;

    if (new_shape.empty()) {
        view->strides.clear();
        return view;
    }

    // We initialize the Strides vector
    view->strides.resize(new_shape.size());
    // The last one is always 1
    view->strides.back() = 1;
    // Each element is himself times the previous element from the right
    for (int i = (int)view->shape.size() - 2; i >= 0; --i) {
        view->strides[i] = view->strides[i + 1] * view->shape[i + 1];
    }

    return view;
}

shared_ptr<Tensor> Tensor::batch_view(int index, bool keep_dim) {
    if(this->getDimension() < 3) throw std::runtime_error("Tensor must have 3 dimension in order to create a batch view");

    int n_batch = this->shape[0];

    if(index < 0 || index >= n_batch) throw std::runtime_error("Batch index is out of bounds");

    auto view = std::shared_ptr<Tensor>(new Tensor(*this));
    int offset = index * this->strides[0];

    view->data = std::shared_ptr<float[]>(this->data, this->data.get() + offset);
    if(!keep_dim) {
        view->shape.erase(view->shape.begin());
        view->strides.erase(view->strides.begin());
    }    
    else {
        view->shape[0] = 1;
        view->strides[0] = 0;
    }

    view->total_size = element_vector_product(view->shape);
    view->parents = {shared_from_this()};

    return view;
}


std::shared_ptr<Tensor> Tensor::slice(int start_idx, int end_idx) {
    if(this->getDimension() < 1) {
        throw std::runtime_error("Tensor must have at least 1 dimension to slice");
    }

    int n_batch = this->shape[0];
    if(start_idx < 0 || end_idx > n_batch || start_idx >= end_idx) {
        throw std::runtime_error("Slice indices are out of bounds or invalid");
    }

    auto view = std::shared_ptr<Tensor>(new Tensor(*this));
    
    int offset = start_idx * this->strides[0];

    view->data = std::shared_ptr<float[]>(this->data, this->data.get() + offset);

    view->shape[0] = end_idx - start_idx; 
    
    view->total_size = element_vector_product(view->shape);
    
    view->parents = {shared_from_this()};

    return view;
}

shared_ptr<Tensor> Tensor::view_to_3d() {
    // We create a view of the tensor
    auto view = std::shared_ptr<Tensor>(new Tensor(*this));

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

shared_ptr<Tensor> Tensor::view_to_gemm(bool as_b_term) {
    
    auto view = std::shared_ptr<Tensor>(new Tensor(*this));
    // Same logic as view_to_3d, but depends on whether the tensor is the second operand
    if(view->getDimension() == 1){
        if(as_b_term){
            view->shape = {1, view->shape[0], 1};
            view->strides = {0, view->strides[0], 0};
        } else{
            view->shape = {1,1,view->shape[0]};
            view->strides = {0,0, view->strides[0]};
        }
    }
    else if(view->getDimension() == 2){
        view->shape.insert(view->shape.begin(), 1);
        view->strides.insert(view->strides.begin(), 0); 
    }
    else {
        if(view->shape[0] == 1) { view->strides[0] = 0;}
    }
    return view;

}

TensorInfo Tensor::getInfo()  {
    return TensorInfo{
        data.get(),
        shape.data(),
        strides.data(),
        getDimension(),
        total_size
    };
}

// ====================
// View / Broadcasting
// ====================
vector<int> Tensor::broadcast_shapes(const vector<int>& shape_a, const vector<int>& shape_b) {
    int max_dims = std::max(shape_a.size(), shape_b.size());
    vector<int> out_shape(max_dims);
    
    int i_a = (int)shape_a.size() - 1;
    int i_b = (int)shape_b.size() - 1;
    int i_out = max_dims - 1;
    
    while (i_out >= 0) {
        int dim_a = (i_a >= 0) ? shape_a[i_a] : 1;
        int dim_b = (i_b >= 0) ? shape_b[i_b] : 1;
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("Tensors cannot be broadcasted together");
        } else {
            out_shape[i_out] = std::max(dim_a, dim_b);
        }
        
        i_a--; i_b--; i_out--;
    }
    return out_shape;
}

shared_ptr<Tensor> Tensor::broadcast_to(const vector<int>& target_shape) {
    if (this->shape == target_shape) return shared_from_this();

    auto view = std::shared_ptr<Tensor>(new Tensor(*this));
    view->shape = target_shape;
    view->strides.resize(target_shape.size());
    
    int offset = (int)target_shape.size() - (int)this->shape.size();
    
    for (size_t i = 0; i < target_shape.size(); ++i) {
        int original_idx = (int)i - offset;
        
        if (original_idx < 0) {
            // New dimension (broadcasted) -> stride 0
            view->strides[i] = 0;
        } else {
            // Existing dimension
            if (this->shape[original_idx] == target_shape[i]) {
                view->strides[i] = this->strides[original_idx];
            } else {
                // Dimension 1 stretched -> stride 0
                view->strides[i] = 0;
            }
        }
    }
    return view;
}

shared_ptr<Tensor> Tensor::compute_binary_op(shared_ptr<Tensor> b, BinaryOp op){
    // Calculate output shape
    vector<int> out_shape = broadcast_shapes(this->shape, b->shape);
    auto out = Tensor::create(out_shape);

    // Create broadcasted views of inputs
    auto a_view = this->broadcast_to(out_shape);
    auto b_view = b->broadcast_to(out_shape);
    
    // Update output info
    TensorInfo i_out = out->getInfo();
    
    // Call backend with aligned shapes
    Device::get()->binary(a_view->getInfo(), b_view->getInfo(), i_out, op);

    return out;
}

shared_ptr<Tensor> Tensor::compute_unary_op(UnaryOp op) {
    auto out = Tensor::create(this->shape);
    TensorInfo i_out = out->getInfo();
    
    Device::get()->unary(this->getInfo(), i_out, op);

    return out;
}

shared_ptr<Tensor> Tensor::compute_reduce_op(int dim, ReduceOp op) {
    vector<int> out_shape = this->shape;
    out_shape[dim] = 1;
    auto out = Tensor::create(out_shape);
    TensorInfo i_out = out->getInfo();
    
    Device::get()->reduce(this->getInfo(), i_out, dim, op);

    return out;

}

shared_ptr<Tensor> Tensor::compute_matmul(shared_ptr<Tensor> b) {
    auto a_view = this->view_to_gemm(false);
    auto b_view = b->view_to_gemm(true);

    int batch_out = std::max(a_view->shape[0], b_view->shape[0]);

    std::vector<int> out_shape;
    if (this->shape.size() == 2 && b->shape.size() == 2) {
        out_shape = {a_view->shape[1], b_view->shape[2]};
    } else {
        out_shape = {batch_out, a_view->shape[1], b_view->shape[2]};
    }

    auto out = Tensor::create(out_shape);

    shared_ptr<Tensor> out_view;
    if (out_shape.size() == 2) {
       out_view = out->view_to_gemm(false); 
    } else {
       out_view = out; 
    }

    TensorInfo i_a = a_view->getInfo();
    TensorInfo i_b = b_view->getInfo();
    TensorInfo i_out = out_view->getInfo();
    
    Device::get()->gemm(i_a, i_b, i_out);

    return out;
}

shared_ptr<Tensor> Tensor::compute_gather(shared_ptr<Tensor> ind) {
    auto out_shape = ind->shape;
    out_shape.push_back(this->shape.back());

    auto out = Tensor::create(out_shape);
    TensorInfo i_w = this->getInfo();
    TensorInfo i_ind = ind->getInfo();
    TensorInfo i_out = out->getInfo();

    Device::get()->gather(i_w,i_ind,i_out);

    return out;
}

void Tensor::compute_scatter_add(shared_ptr<Tensor> ind, shared_ptr<Tensor> incoming_grad) {
    TensorInfo i_w_grad = this->getInfo();
    TensorInfo i_ind = ind->getInfo();
    TensorInfo i_grad_out = incoming_grad->getInfo();

    Device::get()->scatter_add(i_ind, i_grad_out, i_w_grad);
}

void Tensor::im2col(float* workspace,int out_height, int out_width,
    int kernel_height, int kernel_width, int stride, int padding) {

    for(size_t row = 0; row < (size_t)out_height; row++){
        for(size_t col = 0; col < (size_t)out_width; col++){
            for(size_t channel = 0; channel < (size_t)this->shape[1]; channel++){
                int actual_window = col + row * out_width;
                int window_size = this->shape.at(1)  * kernel_height * kernel_width;
                int channel_offset = (channel * kernel_height * kernel_width);
                float* col_ptr = &workspace[actual_window * window_size + channel_offset];
                for(size_t k_h = 0; k_h < (size_t)kernel_height; k_h++) {
                    int y = row * stride - padding + k_h;
                    for(size_t k_w = 0; k_w < (size_t)kernel_width; k_w++){
                        int x = col * stride - padding + k_w;
                        if(x < 0 || y < 0 || x >= this->shape.at(3) || y >= this->shape.at(2)) *col_ptr = 0.0f;
                        else  {
                            float* pixel_ptr = &this->data[(y * this->strides.at(2)) + (x * this->strides.at(3)) +
                            (channel * this->strides.at(1))];
                            *col_ptr = *pixel_ptr;
                        }

                        col_ptr++;
                    }
                }
            }
        }
    }   

}

void Tensor::col2im (const float* data_col, int out_height, int out_width,
                    int kernel_height, int kernel_width, int stride, int padding) {

    int c_in = this->shape[1];
    int h_in = this->shape[2];
    int w_in = this->shape[3];

    int window_size = c_in * kernel_height * kernel_width;
    for (int row = 0; row < out_height; row++) {
        for (int col = 0; col < out_width; col++) {
            int actual_window = col + row * out_width;
            
            for (int channel = 0; channel < c_in; channel++) {
                int channel_offset = channel * kernel_height * kernel_width;
                const float* col_ptr = &data_col[actual_window * window_size + channel_offset];
                
                for (int k_h = 0; k_h < kernel_height; k_h++) {
                    int y = row * stride - padding + k_h;
                    
                    for (int k_w = 0; k_w < kernel_width; k_w++) {
                        int x = col * stride - padding + k_w;
                        
                        if (x >= 0 && y >= 0 && x < w_in && y < h_in) {
                            int pixel_idx = (y * this->strides[2]) + (x * this->strides[3]) + (channel * this->strides[1]);
                            
                            this->data[pixel_idx] += *col_ptr;
                        }
                        col_ptr++;
                    }
                }
            }
        }
    }
    
}

shared_ptr<Tensor> Tensor::compute_conv2d(std::shared_ptr<Tensor>  w, int stride, int padding) {

        if(this->getDimension() != 4 || w->getDimension() != 4){
            throw std::runtime_error("Input and Weight must have 4 dimensions");
        }
        if (stride <= 0) {
            throw std::runtime_error("stride must be > 0");
        }
        if (padding < 0) {
            throw std::runtime_error("padding must be >= 0");
        }

        int n_batch = this->getShape().at(0);
        int c_in = this->getShape().at(1);
        int h_in = this->getShape().at(2);
        int w_in = this->getShape().at(3);
        int c_out = w->getShape().at(0);
        int k_h = w->getShape().at(2);
        int k_w = w->getShape().at(3);

        int h_out = (h_in + 2 * padding - k_h) / stride + 1;
        int w_out = (w_in + 2 * padding - k_w) / stride + 1;

        if (h_out <= 0 || w_out <= 0) {
            throw std::runtime_error("Invalid output shape for conv2d with current kernel/stride/padding");
        }

        std::vector<int> final_out_shape = {n_batch, c_out, h_out, w_out};
        auto out = Tensor::zeros(final_out_shape);

        int workspace_size = c_in * k_h * k_w * h_out * w_out;
        float* workspace = new float[workspace_size];

        std::vector<float> w_view_data(w->getData(), w->getData() + w->total_size);
        std::vector<int> w_view_shape{c_out, (c_in * k_h * k_w)};
        auto w_view = Tensor::create(w_view_shape, w_view_data)->view_to_gemm(false);

        
        for (int b = 0; b < n_batch; b++) {
            auto x_b = this->batch_view(b,true);
            x_b->im2col(workspace, h_out, w_out, k_h, k_w, stride, padding);


            std::vector<float> workspace_data{workspace, workspace + workspace_size};
            std::vector<int> workspace_tensor_shape = {(c_in * k_h * k_w), (h_out * w_out)};
            auto workspace_tensor = Tensor::create(workspace_tensor_shape, workspace_data);
            workspace_tensor->strides = {1, (c_in * k_h * k_w)};

            shared_ptr<Tensor> im2col_tensor = workspace_tensor->view_to_gemm(true);

            std::vector<int> batch_out_shape = {c_out, (h_out * w_out)};
            auto batch_out = Tensor::create(batch_out_shape)->view_to_3d();
            auto batch_out_info = batch_out->getInfo(); 

            Device::get()->gemm(w_view->getInfo(), im2col_tensor->getInfo(), batch_out_info);
            
            float* temp_data = batch_out->getData();
            float* final_data = out->getData() + b * (c_out * h_out * w_out);
            for(int i = 0; i < (c_out * h_out * w_out); i++) {
                final_data[i] = temp_data[i];
            }
        }

        delete[] workspace;
        return out;

    }

shared_ptr<Tensor> Tensor::compute_flatten() {
    auto out = std::shared_ptr<Tensor>(new Tensor(*this));
    
    int batch_size = this->shape[0];
    int elements_per_batch = this->total_size / batch_size;
    
    out->shape = {batch_size, elements_per_batch};
    out->strides = {elements_per_batch, 1};
    
    return out;
}

// ===================
// Tensor initializers
// ===================
shared_ptr<Tensor> Tensor::zeros(const vector<int>& shape) {
    return Tensor::create(shape, vector<float>(element_vector_product(shape),0.0f));
}

shared_ptr<Tensor> Tensor::ones(const vector<int>& shape) {
    return Tensor::create(shape, vector<float>(element_vector_product(shape),1.0f));
}

shared_ptr<Tensor> Tensor::random(const vector<int>& shape, float min_val, float max_val) {
    size_t size = element_vector_product(shape);
    vector<float> random_data(size);
    
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dis(min_val, max_val);
    for (auto& val : random_data) val = dis(gen);

    return Tensor::create(shape, random_data);
}

// ===================
// Backward functions
// ===================

void Tensor::build_topo(shared_ptr<Tensor> v, vector<shared_ptr<Tensor>>& topo, unordered_set<Tensor*>& visited){
    if(visited.find(v.get()) == visited.end()){
        visited.insert(v.get());
        for(auto& child : v.get()->getParents()){
            build_topo(child, topo,visited);
        }
        topo.push_back(v);
    }

}

// backward function
void Tensor::backward() {
    vector<shared_ptr<Tensor>> topo;
    unordered_set<Tensor*> visited;
    
    // we build out topo graph
    build_topo(shared_from_this(), topo, visited);
    
    // we initialize the current gradient to ones
    this->grad = Tensor::ones(this->shape);
    
    // we go through the graph in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        shared_ptr<Tensor> t = *it;
        if (t->_backward) {
            t->_backward();
        }
    }
}

// ===================
// Load and save weights
// ===================

void Tensor::serialize(std::ofstream& out) const {
    size_t dim = this->shape.size();
    out.write(reinterpret_cast<const char*>(&dim), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(this->shape.data()), dim * sizeof(int));
    out.write(reinterpret_cast<const char*>(this->getData()), this->total_size * sizeof(float));
}

void Tensor::deserialize(std::ifstream& in)  {
    size_t dim;
    in.read(reinterpret_cast<char*>(&dim), sizeof(size_t));

    this->shape.resize(dim);
    in.read(reinterpret_cast<char*>(shape.data()), dim * sizeof(int));

    size_t total_elements = 1;
    for(int dim : this->shape) {
        total_elements *= dim;
    }

    this->data.reset(new float[total_elements]);
    in.read(reinterpret_cast<char*>(this->getData()), total_elements * sizeof(float));

}