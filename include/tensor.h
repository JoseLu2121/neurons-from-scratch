#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <cassert>
#include <unordered_set>

struct Tensor : std::enable_shared_from_this<Tensor> {
private:
    std::shared_ptr<float[]> data;
    size_t total_size;
 
    size_t offset = 0;

public:
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<std::shared_ptr<Tensor>> childs;
    

    std::shared_ptr<Tensor> grad;
    std::function<void()> _backward;
    void backward();

    Tensor(const std::vector<int>& str, float* data_param = nullptr,
           const std::vector<std::shared_ptr<Tensor>> childs_param = {});

    void printElements(int count = 1) const;
    void printShape();
    void printStrides();
    
    size_t getSize() const { return total_size; }
    std::vector<int> getStrides() const { return strides; }
    float* getData() const { return data.get() + offset; }
    std::vector<int> getShape() const { return shape; }
    std::vector<std::shared_ptr<Tensor>> getChilds() const { return childs; }
    int getDimension() const { return shape.size(); }
    

    std::shared_ptr<Tensor> getBatch(int index);
    std::shared_ptr<Tensor> view_to_3d(); 


    static std::shared_ptr<Tensor> zeros(const std::vector<int>& shape);
    static std::shared_ptr<Tensor> ones(const std::vector<int>& shape);
};