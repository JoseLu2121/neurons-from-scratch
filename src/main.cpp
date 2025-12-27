#include <iostream>
#include <vector>
#include <memory>
#include "ops.h"
#include "tensor.h"


using namespace std;

int main() {
    auto input = make_shared<Tensor>(vector<int>{1,3}, new float[3]{1,2,3});
    auto w1 = make_shared<Tensor>(vector<int>{3,3}, new float[9]{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9});
    auto b1 = make_shared<Tensor>(vector<int> {1,3}, new float[3]{0.1,0.2,0.3});
    
    auto w2 = make_shared<Tensor>(vector<int>{2,3}, new float[6]{0.2,0.3,0.4,0.5,0.6,0.7});
    auto b2 = make_shared<Tensor>(vector<int> {1,2}, new float[3]{0.1,0.2});
    auto w3 = make_shared<Tensor>(vector<int>{1,2}, new float[2]{0.3,0.4});
    auto b3 = make_shared<Tensor>(vector<int> {1,1}, new float[1]{0.1});

    auto w1_T = transpose_view(w1);
    auto w2_T = transpose_view(w2);
    auto w3_T = transpose_view(w3);

    auto z1 = matmul(input, w1_T) + b1;
    
    auto h1 = relu(z1);
    auto z2 = matmul(h1, w2_T) + b2;
    auto h2 = relu(z2);
    auto z3 = matmul(h2, w3_T) + b3;
    auto y  = relu(z3);
    
    y->printElements(1);
    y->backward();

    cout << "\n=== VERIFICACIÓN FINAL ===" << endl;
    
    cout << "Gradientes de W1 (Capa 1):" << endl;
    if(w1->grad) {
        w1->grad->printElements(9);
    } else {
        cout << "ERROR: W1 no tiene gradiente." << endl;
    }

    cout << "\nGradientes del Input original:" << endl;
    if(input->grad) {
        input->grad->printElements(3);
    }

    return 0;
}