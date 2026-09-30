#include "engine.hpp"
#include "draw.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>


static std::vector<std::string> load_names(const std::string& path)
{
    std::vector<std::string> names{};
    std::ifstream file{path};
    for (std::string line; std::getline(file, line);)
    {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            names.push_back(line);
        }
    }
    return names;
}

static const char* describe(lithium::error e)
{
    switch (e)
    {
        case lithium::error::cfg_error:           return "could not parse the cfg";
        case lithium::error::weights_error:       return "could not parse the weights";
        case lithium::error::network_error:       return "could not build the network";
        case lithium::error::preprocessing_error: return "could not load or preprocess the image";
    }
    return "unknown error";
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <cfg> <weights> <image> [names]\n", argv[0]);
        std::printf("  e.g. %s reference/yolov3-tiny-letterbox.cfg "
                    "reference/yolov3-tiny.weights imges/dog.jpg\n", argv[0]);
        return 2;
    }

    const std::string names_path{(argc > 4) ? argv[4] : "reference/coco.names"};
    const std::vector<std::string> names{load_names(names_path)};
    if (names.empty())
    {
        std::printf("could not read class names from %s\n", names_path.c_str());
        return 2;
    }

    auto predictions{lithium::run_inference(argc, argv)};
    if (!predictions)
    {
        std::printf("\n%s\n", describe(predictions.error()));
        return 1;
    }

    int found{0};
    for (const lithium::Decoded& d : *predictions)
    {
        for (std::size_t c{0}; c < d.probs.size(); ++c)
        {
            if (d.probs[c] <= 0.0f)
            {
                continue;
            }
            const char* label{(c < names.size()) ? names[c].c_str() : "?"};
            const lithium::Colour colour{lithium::class_colour(c)};
            std::printf("%-14s %3.0f%%   corners (%4.0f, %4.0f) - (%4.0f, %4.0f)   "
                        "box colour rgb(%3d,%3d,%3d)\n",
                        label, d.probs[c] * 100.0f,
                        d.x - d.w / 2.0f, d.y - d.h / 2.0f,
                        d.x + d.w / 2.0f, d.y + d.h / 2.0f,
                        colour.r, colour.g, colour.b);
            ++found;
        }
    }

    if (found == 0)
    {
        std::printf("no detections above the threshold\n");
    }
    else
    {
        std::printf("\n%d detection%s from %zu surviving box%s\n",
                    found, found == 1 ? "" : "s",
                    predictions->size(), predictions->size() == 1 ? "" : "es");

        if (lithium::draw_detections(argv[3], "detections.ppm", *predictions))
        {
            std::printf("wrote detections.ppm\n");
        }
        else
        {
            std::printf("could not write detections.ppm\n");
        }
    }
    return 0;
}
