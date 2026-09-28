#include "yolo.hpp"
#include <cmath>

static float sigmoid(float x)
{
    return 1.0f / (std::exp(static_cast<long double>(-x)) + 1);
} 

static void apply_sigmoid(float* arr, std::size_t start, int len)
{
    for (int i{0}; i < len; ++i)
    {
        arr[start + i] = sigmoid(arr[start + i]); 
    }

}

namespace lithium
{
    void yolo(NetworkLayer layer, const Tensor& in, Tensor& out)
    {
        std::copy(in.data, in.data + in.count(), out.data);
        int hw{out.h * out.w};

        for (int i{0}; i < 3; ++i)
        {
            int base{i * (5 + layer.spec.yolo_classes) * hw};

            apply_sigmoid(out.data, static_cast<std::size_t>(base), hw);
            apply_sigmoid(out.data, static_cast<std::size_t>(base + hw), hw);

            for (int j{4}; j < 5 + layer.spec.yolo_classes; ++j)
            {
                apply_sigmoid(out.data, static_cast<std::size_t>(base + j * hw), hw);
            }


            
        }


    }
}