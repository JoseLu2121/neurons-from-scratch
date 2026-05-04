#pragma once
#include "block.h"
#include "ops.h"       
#include <stdexcept> 
#include "neuron.h"

// A layer of neurons class
struct Layer : std::enable_shared_from_this<Layer> {
    public:
    std::vector<std::shared_ptr<Neuron>> neurons {}; // List of neurons

    // Constructor
    Layer(int inputs, int out);

    // Compute all the neurons
    std::vector<std::shared_ptr<Unit>> forward(std::vector<std::shared_ptr<Unit>>& inputs);

    // Get all the trainable parameters of the layer
    std::vector<std::shared_ptr<Unit>> parameters();
};



// Linear Layer
// Y = X @ W.T + B
struct Linear : public Block {
    TensorPtr W; // Weight (Out, In)
    TensorPtr B; // Bias (1, Out)

    // Constructor
    Linear(int in_size, int out_size) : Block("Linear") {
        // Shape: (Out,In)
        W = Tensor::random({out_size, in_size},-0.01f,0.01f);
        
        // Shape: (1,Out)
        B = Tensor::zeros({1, out_size});
    }

    // Forward function
    TensorList forward(TensorList inputs) override {
        // We expect exactly one input tensor
        if (inputs.size() != 1) {
            throw std::runtime_error("Linear layer expects one input, it received " + std::to_string(inputs.size()));
        }

        auto X = inputs[0];

        // we use the transpose operation to get W.T
        auto W_t = transpose_view(W);

        // Compute Y = X @ W.T + B
        auto Y = matmul(X, W_t) + B;

        return {Y}; // We convert it to TensorList
    }

    // Parameters function (we add Weights and Bias)
    std::vector<TensorPtr> parameters() override {
        return {W, B};
    }
};

// Relu Layer
// Y = max(0, X)
struct ReLU : public Block {
    // Constructor
    ReLU() : Block("ReLU") {}

    TensorList forward(TensorList inputs) override {
        if (inputs.size() != 1) {
             throw std::runtime_error("ReLU waits one input, it received " + std::to_string(inputs.size()));
        }
        
        // We return de operation relu applied to the input tensor
        return { relu(inputs[0]) };
    }

    // No parameters function because we have no Weights or Biases
};


// Sigmoid Layer
struct Sigmoid: public Block {
    // Constructor
    Sigmoid() : Block("Sigmoid") {};

    TensorList forward(TensorList inputs) override {
        if (inputs.size() != 1) {
             throw std::runtime_error("Sigmoid waits one input, it received " + std::to_string(inputs.size()));
        };
        // We return de operation sigmoid applied to the input tensor
        return { sigmoid(inputs[0]) };


    };

    // No parameters function because we have no Weights or Biases

};

struct Conv2D: public Block {
    TensorPtr W;
    TensorPtr B;
    int stride;
    int padding;
    
    Conv2D(int in_channels, int out_channels, int
        kernel_size, int stride, int padding)
        : Block("Conv2D"), stride(stride), padding(padding) {
        
        W = Tensor::random({out_channels,in_channels,kernel_size,kernel_size});
        B = Tensor::zeros({out_channels});
            
    };

    TensorList forward(TensorList inputs) override {
        if (inputs.size() != 1) {
               throw std::runtime_error("Conv2D waits one input, it received " + std::to_string(inputs.size()));
        };

        auto X = inputs[0];

        auto Y = conv2d(X, W, stride, padding) + B;

        return {Y};
    }

    // Parameters function (we add Weights and Bias)
    std::vector<TensorPtr> parameters() override {
        return {W, B};
    }
};

// Flatten layer
class Flatten : public Block {
public:
    Flatten() : Block("Flatten") {};

    TensorList forward(TensorList inputs) override {
        auto out = flatten(inputs[0]);
        return {out};
    }

    TensorList parameters() override {
        return {}; 
    }
};

// Tanh Layer
struct Tanh: public Block {
    // Constructor
    Tanh() : Block("Tanh") {};

    TensorList forward(TensorList inputs) override {
        if (inputs.size() != 1) {
             throw std::runtime_error("Sigmoid waits one input, it received " + std::to_string(inputs.size()));
        };
        // We return de operation sigmoid applied to the input tensor
        return { tanh(inputs[0]) };


    };

    // No parameters function because we have no Weights or Biases

};

// Softmax Layer
struct Softmax: public Block {
    // Constructor
    Softmax() : Block("Softmax") {};

    TensorList forward(TensorList inputs) override {
        if(inputs.size() != 1) {
            throw std::runtime_error("Softmax waits one input, it received " + std::to_string(inputs.size()));

        }
        
        auto x = inputs[0];
        
        // Estabilidad numérica: restar el máximo de cada fila
        auto axis = x->getShape().size() - 1;
        auto max_x = max(x, axis);
        auto x_shifted = x - max_x; 
        
        auto exp_x = exp(x_shifted);
        auto sum_exp_x = sum(exp_x, axis);
        auto out = exp_x / (sum_exp_x + 1e-7f);

        return { out };
    }


};

// Embedding Layer
struct Embedding: public Block {
    TensorPtr W;

    Embedding(int vocab_size, int vector_size) : Block("Embedding") {
        W = Tensor::random({vocab_size,vector_size}, -0.01f, 0.01f);
    };

    TensorList forward(TensorList inputs) override {

        if (inputs.size() != 1) {
            throw std::runtime_error("Embedding expects only one input.");
        }

        auto out = gather(W,inputs[0]);

        return {out};
    }

    // Parameters function (we add Weights and Bias)
    std::vector<TensorPtr> parameters() override {
        return {W};
    }

};

// LayerNorm Layer
struct LayerNorm: public Block {
    TensorPtr gamma;
    TensorPtr beta;
    float epsilon;

    LayerNorm(int embed_dim, float epsilon = 1e-5f): Block("LayerNorm"), epsilon(epsilon) {
        gamma = Tensor::ones({1, 1, embed_dim});
        beta = Tensor::zeros({1, 1, embed_dim});
    };

    TensorList forward(TensorList inputs) override {
        if (inputs.size() != 1) {
            throw std::runtime_error("LayerNorm expects exactly one input");
        }

        auto X = inputs[0];

        int axis = X->getShape().size() - 1;
        float d = static_cast<float>(X->getShape()[axis]);

        auto mean = sum(X, axis) / d;

        auto x_centered = X - mean;
        auto variance = sum(x_centered * x_centered, axis) / d;
        auto desv = sqrt(variance + epsilon);

        auto x_norm = x_centered / desv;

        auto Y = (x_norm * gamma) + beta;

        return {Y};
    }

    std::vector<TensorPtr> parameters() override {
        return {gamma, beta};
    }

};


struct SelfAttention : public Block {
    std::shared_ptr<Linear> q_proj, k_proj, v_proj, out_proj;
    std::shared_ptr<Softmax> softmax_layer;
    int embed_dim;

    SelfAttention(int dim) : Block("SelfAttention"), embed_dim(dim) {
        q_proj = std::make_shared<Linear>(dim, dim);
        k_proj = std::make_shared<Linear>(dim, dim);
        v_proj = std::make_shared<Linear>(dim, dim);
        out_proj = std::make_shared<Linear>(dim, dim);
        
        softmax_layer = std::make_shared<Softmax>();
    }

    TensorList forward(TensorList inputs) override {
        auto X = inputs[0];
        int batch_size = X->getShape()[0];
        int seq_len = X->getShape()[1];

        auto Q = q_proj->forward({X})[0];
        auto K = k_proj->forward({X})[0];
        auto V = v_proj->forward({X})[0];

        auto K_T = transpose_view(K); 
        auto scores = matmul(Q, K_T);

        float scale = std::sqrt(static_cast<float>(embed_dim));
        auto scaled_scores = scores / scale;

        auto mask = Tensor::zeros({batch_size, seq_len, seq_len});
        float* mask_ptr = mask->getData(); // O mask->data
        for (int b = 0; b < batch_size; ++b) {
            for (int r = 0; r < seq_len; ++r) {
                for (int c = r + 1; c < seq_len; ++c) {
                    mask_ptr[b * (seq_len * seq_len) + r * seq_len + c] = -1e9f;
                }
            }
        }

        auto masked_scores = scaled_scores + mask;
        
        auto probs = softmax_layer->forward({masked_scores})[0];
        
        auto context = matmul(probs, V);
        
        return out_proj->forward({context});
    }

    std::vector<TensorPtr> parameters() override {
        std::vector<TensorPtr> p;
        for(auto l : {q_proj, k_proj, v_proj, out_proj}) {
            auto lp = l->parameters();
            p.insert(p.end(), lp.begin(), lp.end());
        }
        return p;
    }
};