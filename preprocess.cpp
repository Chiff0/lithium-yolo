#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "preprocess.hpp"
#include "letterbox.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>


static constexpr float gray_pixel{0.5f};


static std::vector<float> to_planar(const unsigned char* pixels, int w, int h, int c)
{
    std::vector<float> planar(static_cast<std::size_t>(c) * h * w);
    for (int channel{0}; channel < c; ++channel)
    {
        for (int y{0}; y < h; ++y)
        {
            for (int x{0}; x < w; ++x)
            {
                planar[(static_cast<std::size_t>(channel) * h + y) * w + x] =
                    pixels[(static_cast<std::size_t>(y) * w + x) * c + channel] / 255.0f;
            }
        }
    }
    return planar;
}

static void resize(const std::vector<float>& src, int src_w, int src_h, int c,
    std::vector<float>& dst, int dst_w, int dst_h, std::vector<float>& scratch)
{
    dst.assign(static_cast<std::size_t>(c) * dst_h * dst_w, 0.0f);
    if (src_w == dst_w && src_h == dst_h)
    {
        dst = src;
        return;
    }

    scratch.assign(static_cast<std::size_t>(c) * src_h * dst_w, 0.0f);

    float w_scale{(src_w - 1.0f) / (dst_w - 1.0f)},
          h_scale{(src_h - 1.0f) / (dst_h - 1.0f)};

    for (int channel{0}; channel < c; ++channel)
    {
        const float* in{src.data() + static_cast<std::size_t>(channel) * src_h * src_w};
        float* part{scratch.data() + static_cast<std::size_t>(channel) * src_h * dst_w};

        for (int y{0}; y < src_h; ++y)
        {
            for (int x{0}; x < dst_w; ++x)
            {
                float val{};
                if (x == dst_w - 1 || src_w == 1)
                {
                    val = in[static_cast<std::size_t>(y) * src_w + (src_w - 1)];
                }
                else
                {
                    float sx{x * w_scale};
                    int ix{static_cast<int>(sx)};
                    float dx{sx - ix};
                    val = (1.0f - dx) * in[static_cast<std::size_t>(y) * src_w + ix]
                        + dx * in[static_cast<std::size_t>(y) * src_w + ix + 1];
                }
                part[static_cast<std::size_t>(y) * dst_w + x] = val;
            }
        }

        float* outc{dst.data() + static_cast<std::size_t>(channel) * dst_h * dst_w};
        for (int y{0}; y < dst_h; ++y)
        {
            float sy{y * h_scale};
            int iy{static_cast<int>(sy)};
            float dy{sy - iy};

            for (int x{0}; x < dst_w; ++x)
            {
                outc[static_cast<std::size_t>(y) * dst_w + x] =
                    (1.0f - dy) * part[static_cast<std::size_t>(iy) * dst_w + x];
            }
            if (y == dst_h - 1 || src_h == 1)
            {
                continue;
            }
            for (int x{0}; x < dst_w; ++x)
            {
                outc[static_cast<std::size_t>(y) * dst_w + x] +=
                    dy * part[static_cast<std::size_t>(iy + 1) * dst_w + x];
            }
        }
    }
}

namespace lithium
{
    std::expected<Loaded, preprocess_error> preprocess(std::string_view path, Tensor& out)
    {
        if (out.data == nullptr || out.c <= 0 || out.h <= 0 || out.w <= 0)
        {
            return std::unexpected(preprocess_error::invalid_output);
        }

        int w{}, h{}, channels{};
        unsigned char* pixels{stbi_load(std::string(path).c_str(), &w, &h, &channels, out.c)};
        if (pixels == nullptr)
        {
            return std::unexpected(preprocess_error::file_input_error);
        }

        std::vector<float> planar{to_planar(pixels, w, h, out.c)};
        stbi_image_free(pixels);

        Letterbox box{letterbox_fit(w, h, out.w, out.h)};

        std::vector<float> fitted{}, scratch{};
        resize(planar, w, h, out.c, fitted, box.fit_w, box.fit_h, scratch);

        std::fill(out.data, out.data + out.count(), gray_pixel);

        for (int channel{0}; channel < out.c; ++channel)
        {
            for (int y{0}; y < box.fit_h; ++y)
            {
                const float* row{fitted.data()
                    + (static_cast<std::size_t>(channel) * box.fit_h + y) * box.fit_w};
                std::copy(row, row + box.fit_w,
                    out.data + out.index(channel, box.pad_w, y + box.pad_h));
            }
        }

        return Loaded{w, h};  
    }
}
