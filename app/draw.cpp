#include "stb_image.h"

#include "draw.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <vector>


namespace
{
    constexpr lithium::Colour palette[]
    {
        {230,  25,  75}, { 60, 180,  75}, {255, 225,  25}, {  0, 130, 200},
        {245, 130,  48}, {145,  30, 180}, { 70, 240, 240}, {240,  50, 230},
        {210, 245,  60}, {250, 190, 212}, {  0, 128, 128}, {170, 110,  40},
    };

    void plot(std::vector<unsigned char>& rgb, int w, int h, int x, int y,
        lithium::Colour colour)
    {
        if (x < 0 || y < 0 || x >= w || y >= h)
        {
            return;
        }
        const std::size_t at{(static_cast<std::size_t>(y) * w + x) * 3};
        rgb[at + 0] = colour.r;
        rgb[at + 1] = colour.g;
        rgb[at + 2] = colour.b;
    }

    void rectangle(std::vector<unsigned char>& rgb, int w, int h,
        int left, int top, int right, int bottom, lithium::Colour colour, int thickness)
    {
        for (int t{0}; t < thickness; ++t)
        {
            for (int x{left - t}; x <= right + t; ++x)
            {
                plot(rgb, w, h, x, top - t, colour);
                plot(rgb, w, h, x, bottom + t, colour);
            }
            for (int y{top - t}; y <= bottom + t; ++y)
            {
                plot(rgb, w, h, left - t, y, colour);
                plot(rgb, w, h, right + t, y, colour);
            }
        }
    }
}

namespace lithium
{
    Colour class_colour(std::size_t class_id)
    {
        return palette[class_id % (sizeof(palette) / sizeof(palette[0]))];
    }

    bool draw_detections(std::string_view image_path, std::string_view out_path,
        const std::vector<Decoded>& predictions, int thickness)
    {
        int w{}, h{}, channels{};
        unsigned char* pixels{stbi_load(std::string(image_path).c_str(), &w, &h, &channels, 3)};
        if (pixels == nullptr)
        {
            return false;
        }

        std::vector<unsigned char> rgb(pixels, pixels + static_cast<std::size_t>(w) * h * 3);
        stbi_image_free(pixels);

        for (const Decoded& d : predictions)
        {
            for (std::size_t c{0}; c < d.probs.size(); ++c)
            {
                if (d.probs[c] <= 0.0f)
                {
                    continue;
                }
                const int left{static_cast<int>(d.x - d.w / 2.0f)},
                          top{static_cast<int>(d.y - d.h / 2.0f)},
                          right{static_cast<int>(d.x + d.w / 2.0f)},
                          bottom{static_cast<int>(d.y + d.h / 2.0f)};
                rectangle(rgb, w, h, left, top, right, bottom, class_colour(c), thickness);
            }
        }

        std::ofstream out{std::string(out_path), std::ios::binary};
        if (!out)
        {
            return false;
        }
        out << "P6\n" << w << " " << h << "\n255\n";
        out.write(reinterpret_cast<const char*>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));
        return out.good();
    }
}
