#define STB_IMAGE_IMPLEMENTATION
#include<stb_image.h>
#include "tensor.hpp"
#include "layer.hpp"

#define GRAY_PIXEL 0.5f

static float convert_to_float(int pixel)
{
    return static_cast<float>(pixel / 256); //may be 255 but prolly not ig
}

namespace lithium
{
    Tensor& preprocess(string& path, Netconfig& net) 
    {
        int w, h, channels;
        unsigned char* pixels = stbi_load("dog.jpg", &w, &h, &channels, 3);
        Tensor tensor{};
        tensor.h = h;
        tensor.w = w;
        tensor.c = channels;


        

    }
}



