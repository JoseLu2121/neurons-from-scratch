#include "ops.h"
#include "loss.h"
#include "tensor.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor> prediction, std::shared_ptr<Tensor> target) {
    
    auto diff = prediction - target;
    auto sq_error = diff * diff;
    auto sum_features = sum(sq_error, 1);
    auto total_error = sum(sum_features, 0);
    
    float N = (float)prediction->getSize();
    
    return total_error / N;
}

std::shared_ptr<Tensor> CrossEntropy::forward(std::shared_ptr<Tensor> prediction, 
    std::shared_ptr<Tensor> target) {
    
    int batch_seq = prediction->getSize() / prediction->getShape().back();
    int feat_size = prediction->getShape().back();
    std::vector<float> probs(prediction->getSize());

    for (int b = 0; b < batch_seq; b++) {
        float max_val = -1e9;
        for (int f = 0; f < feat_size; f++) 
            max_val = std::max(max_val, prediction->getData()[b * feat_size + f]);

        float sum_exp = 0;
        for (int f = 0; f < feat_size; f++) {
            probs[b * feat_size + f] = std::exp(prediction->getData()[b * feat_size + f] - max_val);
            sum_exp += probs[b * feat_size + f];
        }
        for (int f = 0; f < feat_size; f++) 
            probs[b * feat_size + f] /= (sum_exp + 1e-9f);
    }

    float sum_val = 0.0f;
    for(size_t i = 0; i < prediction->getSize(); i++) {
        if (target->getData()[i] > 0.0f) {
            sum_val += -target->getData()[i] * std::log(std::max(probs[i], 1e-12f));
        }
    }

    float final_loss = sum_val / batch_seq;
    auto out = Tensor::create({1}, {final_loss});
    out->set_parents({prediction});

    out->set_backward([prediction, target, probs, batch_seq]() {
        prediction->init_grad();
        for(size_t i = 0; i < prediction->getSize(); i++) {
            float grad_val = (probs[i] - target->getData()[i]) / batch_seq;
            prediction->getGrad()->getData()[i] += grad_val;
        }
    });

    return out;
}