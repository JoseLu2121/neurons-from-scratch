#pragma once
#include "tensor.h"
#include "block.h"
#include "optimizer.h"
#include "loss.h"
#include <memory>

class Trainer {
private:
    std::shared_ptr<Block> model;
    std::shared_ptr<Optimizer> optimizer;
    std::shared_ptr<Loss> criterion;

public:
    
    Trainer(std::shared_ptr<Block> m, 
            std::shared_ptr<Optimizer> o, 
            std::shared_ptr<Loss> l);

    void fit(std::shared_ptr<Tensor> x_train, 
             std::shared_ptr<Tensor> y_train, 
             std::shared_ptr<Tensor> x_val, 
             std::shared_ptr<Tensor> y_val, 
             int epochs, 
             int batch_size,
             int print_every = 10);

    float calculate_accuracy(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
};