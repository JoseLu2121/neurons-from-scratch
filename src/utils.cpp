#include "utils.h"

size_t product(const std::vector<int>& v) {
    size_t p = 1;
    for (int x : v) p *= x;
    return p;
}