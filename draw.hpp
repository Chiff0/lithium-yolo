#pragma once

#include "yolo.hpp"

#include <string>
#include <string_view>
#include <vector>


namespace lithium
{
    struct Colour { unsigned char r, g, b; };

    Colour class_colour(std::size_t class_id);

    // reloads the original image, draws every surviving box, writes a binary PPM
    bool draw_detections(std::string_view image_path, std::string_view out_path,
        const std::vector<Decoded>& predictions, int thickness = 3);
}
