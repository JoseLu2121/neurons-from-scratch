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
                  int epochs,
                  int batch_size,
                  int print_every) {

    std::cout << "--- Starting training (" << epochs << " epochs) ---" << std::endl;
    
    int total_samples = x_train->getShape()[0];
    int num_batches = (total_samples + batch_size - 1) / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;

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

            auto accuracy = calculate_accuracy(prediction,y_batch);

            // 4. Backward
            loss->backward();

            // 5. Update
            optimizer->step();

            epoch_loss += loss->getData()[0];
            epoch_acc += accuracy;
        }

        // 6. Logging
        if (epoch % print_every == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / num_batches << " | Accuracy:" <<  epoch_acc / num_batches << std::endl;
        }


    }
    
    std::cout << "--- Training finalised---" << std::endl;
}

float  Trainer::calculate_accuracy(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> a_predictions;
    std::shared_ptr<Tensor> b_labels;
    if(a->getDimension() == 2){
        a_predictions = argmax(a,1);

    } else {
        a_predictions = argmax(a,2);
    }

    if(b->getDimension() == a->getDimension()){
        if(b->getDimension() == 2){
            b_labels = argmax(b, 1);
        } else {
            b_labels = argmax(b, 2);
        }
    } else {
        b_labels = b;
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