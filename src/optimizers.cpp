#include "optimizer.h"
#include <algorithm>
#include <cmath>

Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>>& params)
    : parameters(params) {}


// Zero Grad
void Optimizer::zero_grad() {
    for(auto& param : parameters){
        if(param->getGrad()){
            std::fill(param->getGrad()->getData(), 
                      param->getGrad()->getData() + param->getGrad()->getSize(), 
                      0.0f);
        }
    }
}

SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& params, float learning_rate)
    : Optimizer(params), lr(learning_rate) {}

// Step function: update parameters using gradients
void SGD::step() {
    for (auto& param : parameters) {
        if (param->getGrad()) {
            float* param_data = param->getData();
            float* grad_data = param->getGrad()->getData();
            size_t size = param->getSize();

            for (size_t i = 0; i < size; ++i) {
                // we add to the parameter the negative gradient scaled by learning rate
                param_data[i] -= lr * grad_data[i]; 
            }
        }
    }
}

Adam::Adam(const std::vector<std::shared_ptr<Tensor>>& params, float learning_rate,
float beta1, float beta2, float epsilon) : Optimizer(params), beta1(beta1), beta2(beta2),
epsilon(epsilon), lr(learning_rate) {
    // Pre-allocate history vectors
    m_history.resize(parameters.size());
    v_history.resize(parameters.size());
};


void Adam::step() {
    
    t++; // increment time

    // we initialize the m and v history
    if(m_history.empty()) {
        m_history.resize(parameters.size());
        v_history.resize(parameters.size());
    }

    for (size_t p_id = 0; p_id < parameters.size(); p_id++) {
        auto& param = parameters[p_id];

        if (param->getGrad()) {
            float* param_data = param->getData();
            float* grad_data = param->getGrad()->getData();
            size_t size = param->getSize();
            // initialize current param momentum to zero
            if(m_history[p_id].empty()) {
                m_history[p_id].assign(size, 0.0f);
                v_history[p_id].assign(size, 0.0f);
            }
            // we get out current vector pointer
            float* m_data = m_history[p_id].data();
            float* v_data = v_history[p_id].data();

            for (size_t i = 0; i < size; ++i) {
                auto current_gradient = grad_data[i];
                // we calculate both momentum and variance vectors
                m_data[i] = beta1 * m_data[i] + (1 - beta1) * current_gradient;
                v_data[i] = beta2 * v_data[i] + (1 - beta2) * (current_gradient * current_gradient);
                // we hat both so we dont divide by < 1
                float m_hat = m_data[i] / (1 - std::pow(beta1,t) );
                float v_hat = v_data[i] / (1 - std::pow(beta2,t) );
                param_data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    }

}





