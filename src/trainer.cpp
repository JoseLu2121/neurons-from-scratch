#include "trainer.h"
#include <iostream>
#include "ops.h"

// Trainer Constructor
Trainer::Trainer(std::shared_ptr<Block> m, 
                 std::shared_ptr<Optimizer> o, 
                 std::shared_ptr<Loss> l) 
    : model(m), optimizer(o), criterion(l) {}


// Fit Method: Train the model
void Trainer::fit(std::shared_ptr<Tensor> x_train, 
                  std::shared_ptr<Tensor> y_train, 
                  std::shared_ptr<Tensor> x_val, 
                  std::shared_ptr<Tensor> y_val, 
                  int epochs,
                  int batch_size,
                  int print_every) {

    std::cout << "--- Starting training (" << epochs << " epochs) ---" << std::endl;
    
    int total_samples = x_train->getShape()[0];
    int num_batches = (total_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        
        float epoch_loss = 0.0f;

        for ( int b = 0; b < num_batches; b++){

            int start_idx = b * batch_size;
            int end_idx = std::min(start_idx + batch_size, total_samples);

            auto x_batch = x_train->slice(start_idx, end_idx);
            auto y_batch = y_train->slice(start_idx, end_idx);

            // 1. Zero Grad everything
            optimizer->zero_grad();

            // 2. Forward
            auto outputs = model->forward({x_batch});
            auto prediction = outputs[0];

            // 3. Loss
            auto loss = criterion->forward(prediction, y_batch);

            // 4. Backward
            loss->backward();

            // 5. Update
            optimizer->step();

            epoch_loss += loss->getData()[0];
        }

        // 6. Logging
        if (epoch % print_every == 0 || epoch == epochs - 1) {
            float accuracy = calculate_accuracy(x_val, y_val);
            std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / num_batches << " | Accuracy:" <<  accuracy << std::endl;
        }


    }
    
    std::cout << "--- Training finalised---" << std::endl;
}

float  Trainer::calculate_accuracy(std::shared_ptr<Tensor> x_val, std::shared_ptr<Tensor> y_val) {
    std::shared_ptr<Tensor> a_predictions;
    std::shared_ptr<Tensor> b_labels;

    auto prediction = model->forward({x_val})[0];

    if(prediction->getShape().back() == 1) {
        int n = prediction->getShape()[0];
        int count = 0;
        for(int i = 0; i < n; i++) {
            int pred_class = (prediction->getData()[i] > 0.5f) ? 1 : 0;
            int true_class = static_cast<int>(y_val->getData()[i]);
            if(pred_class == true_class) count++;
        }
        return (float)count / n;
    }

    if(prediction->getDimension() == 2){
        a_predictions = argmax(prediction,1);

    } else {
        a_predictions = argmax(prediction,2);
    }

    if(y_val->getDimension() == prediction->getDimension()){
        if(y_val->getDimension() == 2){
            b_labels = argmax(y_val, 1);
        } else {
            b_labels = argmax(y_val, 2);
        }
    } else {
        b_labels = y_val;
    }
    int count_success = 0;
    
    for(size_t i = 0; i < a_predictions->getSize(); i++){
        if(a_predictions->getData()[i] == b_labels->getData()[i]){
            count_success++;
        }

    }

    auto accuracy = (float) count_success / a_predictions->getSize();

    return accuracy;

}