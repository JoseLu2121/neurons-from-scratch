#include <iostream>
#include <vector>
#include <memory>
#include <sstream>
#include <functional>
#include <array>

using namespace std;

struct Tensor : std::enable_shared_from_this<Tensor> {
    private:
        shared_ptr<float[]> data;
        size_t total_size;
        vector<Tensor> childs;
        shared_ptr<Tensor> grad;
        std::function<void()> _backward;

    public:
        vector<int> shape;
        vector<int> strides;
        
        Tensor(const vector<int>& str, float* data_param = nullptr, 
        const vector<Tensor> childs_param = {}) : data(nullptr), 
        total_size(1), shape(str), childs(childs_param) {
            for(const auto& ptr : str){
                total_size *= ptr;
            }
            
            if(total_size == 0) return;
            data = shared_ptr<float[]> (new float[total_size]);
            if(data_param != nullptr){
                for(size_t i = 0; i<total_size; i++){
                    data[i] = data_param[i];
                }
            }  

            //strides
            
            strides.resize(str.size());
            strides.back() = 1;

            for (int i = str.size() - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * str[i + 1];
      
            }   
            
        }

        Tensor(const Tensor& other) = default;
        Tensor& operator=(const Tensor& other) = default;

        Tensor(Tensor&& other) = default;
        Tensor& operator=(Tensor&& other) = default;


        void printElements(int count = 1) const {
            cout << "Elementos del tensor" << endl;

            for(int i = 0; i<count ; i++){
                cout << "Elemento " << i << ": " << data[i] << endl;
            }

        }

        size_t getSize() const {

            return total_size;
        }

        vector<int> getStrides() const {
            return strides;
        }

        float* getData() const {
            return data.get();
        }

        vector<int> getShape() const {
            return shape;
        }

        vector<Tensor> getChilds() const {
            return childs;
        }

        int getDimension() const {
            return shape.size();
        }

        void setShape(vector<int> p)  {
            this->shape = p;
        }

        void printShape() {
            cout << "Shape: " << endl;
            cout << "(" << endl;
            for(auto& s:this->shape){
                cout << s << endl;

            }

            cout << ")" << endl;
        }
};

static size_t product(const std::vector<int>& v) {
    size_t p = 1;
    for (int x : v) p *= x;
    return p;
}


void broadcasting_inplace(Tensor& small, const Tensor& large) {
    
    for (size_t d = 0; d < small.shape.size(); ++d) {
        if (small.shape[d] == 1 && large.shape[d] != 1) {
            small.strides[d] = 0;
        }
    }
}

Tensor operator+(Tensor a, Tensor b) { 
    const int wanted_dims = 3; 
    while (a.getDimension() < wanted_dims) { a.shape.push_back(1); a.strides.push_back(0); }
    while (b.getDimension() < wanted_dims) { b.shape.push_back(1); b.strides.push_back(0); }

 
    Tensor *large = &a, *small = &b;
 
    for (int d = 0; d < wanted_dims; ++d) {
        if (a.shape[d] < b.shape[d]) { large = &b; small = &a; break; }
        if (a.shape[d] > b.shape[d]) { large = &a; small = &b; break; }
    }


    broadcasting_inplace(*small, *large);


    vector<int> out_shape(wanted_dims);
    for (int d = 0; d < wanted_dims; ++d) {
        if (a.shape[d] == b.shape[d]) out_shape[d] = a.shape[d];
        else if (a.shape[d] == 1) out_shape[d] = b.shape[d];
        else if (b.shape[d] == 1) out_shape[d] = a.shape[d];
        else throw std::runtime_error("Shapes incompatibles para broadcasting");
    }

    size_t out_size = product(out_shape);

    shared_ptr<float[]> out_data(new float[out_size]);

    vector<int> idx(wanted_dims, 0);

    float* Adata = a.getData();
    float* Bdata = b.getData();
    const vector<int>& Astr = a.getStrides();
    const vector<int>& Bstr = b.getStrides();

    for (size_t linear = 0; linear < out_size; ++linear) {

        size_t offA = 0, offB = 0;
        for (int d = 0; d < wanted_dims; ++d) {
            int ia = (a.shape[d] == 1) ? 0 : idx[d];
            int ib = (b.shape[d] == 1) ? 0 : idx[d];
            offA += static_cast<size_t>(ia) * static_cast<size_t>(Astr[d]);
            offB += static_cast<size_t>(ib) * static_cast<size_t>(Bstr[d]);
        }
        out_data[linear] = Adata[offA] + Bdata[offB];

        for (int d = wanted_dims - 1; d >= 0; --d) {
            idx[d]++;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }

    return Tensor(out_shape, out_data.get(), {a, b});
}


Tensor dot_scalar_product(Tensor& a, Tensor& b){
    int size_output = a.getShape().at(0);
    vector<float> output_data(size_output);
    for(int i=0 ; i<size_output; i++){
        output_data[i] = a.getData()[i] * b.getData()[i];

    }
    return Tensor({size_output}, output_data.data(),{a,b});
}

Tensor vector_matrix_product(Tensor& v, Tensor& m){
    vector<int> v_strides = v.getStrides();
    vector<int> m_strides = m.getStrides();

    int column_v = v.getShape().at(1);
    int column_m = m.getShape().at(1);
    vector<float> output_data(column_m);

    for(int i=0; i<column_m; i++){
        float sum = 0.0;
        for(int j=0; j<column_v; j++){
            sum += v.getData()[j] * m.getData()[i+j*m.strides.at(0)];
        }
        output_data[i] = sum;

    }

    return Tensor({column_m}, output_data.data(), {v,m});
}


Tensor matrix_matrix_product(Tensor& m, Tensor& v){
    vector<int> m_strides = m.getStrides();
    vector<int> v_strides = v.getStrides();

    int row_m = m.shape.at(0);
    int col_v = v.shape.at(1);
    int col_m = m.shape.at(1);
    vector<int> output_shape = {row_m, col_v};

    vector<float> output_data(product(output_shape));
    for(int i=0; i<row_m; i++){
        for(int j=0; j<col_v; j++){
            float sum = 0.0;
            for(int k=0; k<col_m;k++){
                sum += m.getData()[k + i*m_strides.at(0)] * v.getData()[j + 
                k*v_strides.at(0)];
            }

            output_data[i*v_strides.at(0) + j]  = sum;
        }

    }

    return Tensor(output_shape,output_data.data(),{m,v});
}


Tensor matmul(Tensor& a, Tensor& b){
    if (a.getDimension() == 1 && b.getDimension()==1){
        // if 1d * 1d scalar vector product
        return dot_scalar_product(a,b);
    }

    if(a.getDimension() == 1 && b.getDimension() == 2){
        // we transform the 1d to 2d
        a.shape.insert(a.shape.begin(),1);

        a.strides.insert(a.strides.begin(),0);
        // we mult 1D x 2D
        return vector_matrix_product(a,b);
    }

    if(a.getDimension() == 2 && b.getDimension() == 1){
        b.shape.push_back(1);
        b.strides.push_back(0);

        return matrix_matrix_product(a,b);
    }

    if(a.getDimension() == 2 && b.getDimension() == 2){
        return matrix_matrix_product(a,b);

    }

}


int main() {

    float data1[] = {4,3,2,1};
    float data2[] = {1,1,1,1,1,1,1,1,1,1,1,1};
    auto tensor1 = Tensor({4},data1);
 

    auto tensor2 = Tensor({3,4},data2);


    auto tensor3 = matmul(tensor2,tensor1);
    tensor3.printElements(12);
    tensor3.printShape();
}