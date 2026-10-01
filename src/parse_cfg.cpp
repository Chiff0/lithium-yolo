#include "parsers.hpp"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <version>


namespace lithium
{
    namespace
    {
        // what std::isspace reports in the C locale, minus the locale and minus
        // the runtime call, so trimming survives constant evaluation
        constexpr bool is_space(char c)
        {
            return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
        }

        constexpr std::string_view trim(std::string_view text)
        {
            const auto space = [](char c) { return is_space(c); };

            while (!text.empty() && space(text.front()))
                text.remove_prefix(1);
            while (!text.empty() && space(text.back()))
                text.remove_suffix(1);
            return text;
        }

        template <typename T>
        constexpr std::optional<T> to_number(std::string_view text)
        {
            T value{};
            const char* const last = text.data() + text.size();
            const auto [stop, ec] = std::from_chars(text.data(), last, value);
            if (ec != std::errc{} || stop != last)
                return std::nullopt;
            return value;
        }

        template <typename T>
        constexpr bool assign(T& target, std::string_view text)
        {
            const auto value = to_number<T>(text);
            if (!value)
                return false;
            target = *value;
            return true;
        }

        // "3,4,5" and "10,14,  23,27" both split into their trimmed fields
        constexpr std::vector<std::string_view> split_list(std::string_view text)
        {
            std::vector<std::string_view> fields;
            for (std::size_t start = 0;;)
            {
                const auto comma = text.find(',', start);
                const auto end = (comma == std::string_view::npos) ? text.size() : comma;
                fields.push_back(trim(text.substr(start, end - start)));
                if (comma == std::string_view::npos)
                    return fields;
                start = comma + 1;
            }
        }

        // darknet reads `pad` as a flag ("pad so the output keeps its size") and
        // `padding` as the pixel count itself. Neither becomes a count until the
        // section's `size` is known, and `size` may be written after either of
        // them, so both are parked here until the section closes.
        struct RawPad
        {
            int flag{};
            std::optional<int> padding{};
        };

        constexpr std::optional<LayerSpec::LayerType> to_layer_type(std::string_view name)
        {
            using LayerType = LayerSpec::LayerType;

            if (name == "convolutional")
                return LayerType::Conv;
            if (name == "maxpool")
                return LayerType::Maxpool;
            if (name == "route")
                return LayerType::Route;
            if (name == "upsample")
                return LayerType::Upsample;
            if (name == "yolo")
                return LayerType::Yolo;
            return std::nullopt;
        }

        constexpr bool assign_activation(Activation& target, std::string_view text)
        {
            if (text == "leaky")
                target = Activation::Leaky;
            else if (text == "linear")
                target = Activation::Linear;
            else
                return false;
            return true;
        }

        constexpr bool assign_padding(RawPad& raw, std::string_view text)
        {
            int padding{};
            if (!assign(padding, text))
                return false;
            raw.padding = padding;
            return true;
        }

        constexpr bool assign_ints(std::vector<int>& target, std::string_view text)
        {
            const auto fields = split_list(text);
            std::vector<int> values;
            values.reserve(fields.size());

            for (std::string_view field : fields)
            {
                const auto value = to_number<int>(field);
                if (!value)
                    return false;
                values.push_back(*value);
            }
            target = std::move(values);
            return true;
        }

        constexpr bool assign_anchors(std::vector<std::pair<float, float>>& target, std::string_view text)
        {
            const auto fields = split_list(text);
            if (fields.size() % 2 != 0)
                return false;

            std::vector<std::pair<float, float>> anchors;
            anchors.reserve(fields.size() / 2);

            for (std::size_t i = 0; i < fields.size(); i += 2)
            {
                const auto w = to_number<float>(fields[i]);
                const auto h = to_number<float>(fields[i + 1]);
                if (!w || !h)
                    return false;
                anchors.emplace_back(*w, *h);
            }
            target = std::move(anchors);
            return true;
        }

        // keys NetConfig has no room for (momentum, decay, learning_rate, ...) are skipped
        constexpr bool apply_net(NetConfig& net, std::string_view key, std::string_view value)
        {
            if (key == "letter_box")
                return assign(net.letter_box, value);
            if (key == "batch")
                return assign(net.batch, value);
            if (key == "subdivisions")
                return assign(net.subdivisions, value);
            if (key == "width")
                return assign(net.width, value);
            if (key == "height")
                return assign(net.height, value);
            if (key == "channels")
                return assign(net.channels, value);
            return true;
        }

        constexpr bool apply_conv(LayerSpec& layer, RawPad& raw, std::string_view key, std::string_view value)
        {
            if (key == "batch_normalize")
            {
                int on{};
                if (!assign(on, value))
                    return false;
                layer.batch_norm = (on != 0);
                return true;
            }
            if (key == "filters")
                return assign(layer.filters, value);
            if (key == "size")
                return assign(layer.size, value);
            if (key == "pad")
                return assign(raw.flag, value);
            if (key == "padding")
                return assign_padding(raw, value);
            if (key == "stride")
                return assign(layer.stride, value);
            if (key == "activation")
                return assign_activation(layer.activation, value);
            return false;
        }

        constexpr bool apply_maxpool(LayerSpec& layer, RawPad& raw, std::string_view key, std::string_view value)
        {
            if (key == "size")
                return assign(layer.size, value);
            if (key == "padding")
                return assign_padding(raw, value);
            if (key == "stride")
                return assign(layer.stride, value);
            return false;
        }

        constexpr bool apply_route(LayerSpec& layer, std::string_view key, std::string_view value)
        {
            if (key == "layers")
                return assign_ints(layer.route_layers, value) && !layer.route_layers.empty();
            if (key == "groups")
                return assign(layer.route_groups, value);
            if (key == "group_id")
                return assign(layer.route_group_id, value);
            return false;
        }

        constexpr bool apply_upsample(LayerSpec& layer, std::string_view key, std::string_view value)
        {
            if (key == "stride")
                return assign(layer.stride, value);
            return false;
        }

        // keys LayerSpec has no room for (num, jitter, random, ...) are skipped
        constexpr bool apply_yolo(LayerSpec& layer, std::string_view key, std::string_view value)
        {
            if (key == "mask")
                return assign_ints(layer.yolo_mask, value);
            if (key == "anchors")
                return assign_anchors(layer.yolo_anchors, value);
            if (key == "classes")
                return assign(layer.yolo_classes, value);
            return true;
        }

        constexpr bool apply_layer(LayerSpec& layer, RawPad& raw, std::string_view key, std::string_view value)
        {
            using LayerType = LayerSpec::LayerType;

            switch (layer.type)
            {
            case LayerType::Conv:
                return apply_conv(layer, raw, key, value);
            case LayerType::Maxpool:
                return apply_maxpool(layer, raw, key, value);
            case LayerType::Route:
                return apply_route(layer, key, value);
            case LayerType::Upsample:
                return apply_upsample(layer, key, value);
            case LayerType::Yolo:
                return apply_yolo(layer, key, value);
            }
            return false;
        }

        // `pad` asks for "keep the output the same size", which costs size/2 pixels
        // a side and so comes out at 0 for a 1x1 kernel; otherwise `padding` is the
        // count itself and defaults to none
        constexpr int conv_padding(int flag, std::optional<int> padding, int size)
        {
            return (flag != 0) ? size / 2 : padding.value_or(0);
        }

        // darknet pads a pooling window by size-1 unless the cfg says otherwise
        constexpr int pool_padding(std::optional<int> padding, int size)
        {
            return padding.value_or(size - 1);
        }

        // darknet resolves a negative route entry relative to the routing layer
        // itself, and a route may only ever look backwards
        constexpr std::optional<int> resolve_route(int entry, std::size_t self)
        {
            const long long index = (entry < 0)
                ? static_cast<long long>(self) + entry
                : entry;

            if (index < 0 || static_cast<unsigned long long>(index) >= self)
                return std::nullopt;
            return static_cast<int>(index);
        }

        // everything a section cannot settle while it is being read: padding needs
        // the kernel size, and a route entry needs its own position in the network.
        // Past this point layer.pad is a pixel count and route_layers are absolute.
        constexpr bool finalize(ParsedCfg& cfg, const std::vector<RawPad>& raw_pads)
        {
            using LayerType = LayerSpec::LayerType;

            for (std::size_t i = 0; i < cfg.layers.size(); ++i)
            {
                LayerSpec& layer = cfg.layers[i];
                const RawPad& raw = raw_pads[i];

                switch (layer.type)
                {
                case LayerType::Conv:
                    if (layer.size <= 0)
                        return false;
                    layer.pad = conv_padding(raw.flag, raw.padding, layer.size);
                    break;

                case LayerType::Maxpool:
                    if (layer.size <= 0)
                        return false;
                    layer.pad = pool_padding(raw.padding, layer.size);
                    break;

                case LayerType::Route:
                    for (int& entry : layer.route_layers)
                    {
                        const auto source = resolve_route(entry, i);
                        if (!source)
                            return false;
                        entry = *source;
                    }
                    break;

                case LayerType::Upsample:
                case LayerType::Yolo:
                    break;
                }
            }
            return true;
        }

        // The rules above are pure, so the compiler can check them. Anchors are the
        // one thing that cannot be constant-evaluated: std::from_chars is constexpr
        // for integers but not for floats.
        //
        // libstdc++ only made the integer overloads constexpr in version 13, so on
        // an older standard library the checks compile out instead of failing the
        // build. JetPack 6 ships libstdc++ 12.
#if defined(__cpp_lib_constexpr_charconv) && __cpp_lib_constexpr_charconv >= 202207L
        namespace checks
        {
            static_assert(trim("  416  ") == "416");
            static_assert(trim("\t-1, 8\r\n") == "-1, 8");
            static_assert(trim("   ").empty());

            static_assert(to_number<int>("416") == 416);
            static_assert(to_number<int>("-4") == -4);
            static_assert(!to_number<int>("4x"));
            static_assert(!to_number<int>(""));

            static_assert(to_layer_type("convolutional") == LayerSpec::LayerType::Conv);
            static_assert(to_layer_type("maxpool") == LayerSpec::LayerType::Maxpool);
            static_assert(!to_layer_type("shortcut"));

            constexpr bool splits_the_way_cfgs_are_written()
            {
                return split_list("3,4,5") == std::vector<std::string_view>{"3", "4", "5"}
                    && split_list("-1, 8") == std::vector<std::string_view>{"-1", "8"}
                    && split_list("10,14,  23,27") == std::vector<std::string_view>{"10", "14", "23", "27"};
            }
            static_assert(splits_the_way_cfgs_are_written());

            // `pad` is a flag: a 1x1 kernel with pad=1 is *unpadded*
            static_assert(conv_padding(1, std::nullopt, 3) == 1);
            static_assert(conv_padding(1, std::nullopt, 1) == 0);
            static_assert(conv_padding(0, std::nullopt, 3) == 0);
            static_assert(conv_padding(0, 2, 3) == 2);
            static_assert(conv_padding(1, 7, 3) == 1);  // `pad` overrides `padding`

            static_assert(pool_padding(std::nullopt, 2) == 1);
            static_assert(pool_padding(std::nullopt, 3) == 2);
            static_assert(pool_padding(0, 2) == 0);

            // yolov3-tiny's two routes: `-4` at layer 17, then `-1, 8` at layer 20
            static_assert(resolve_route(-4, 17) == 13);
            static_assert(resolve_route(-1, 20) == 19);
            static_assert(resolve_route(8, 20) == 8);
            static_assert(!resolve_route(-1, 0));   // nothing before the first layer
            static_assert(!resolve_route(-5, 1));   // reaches back past the start
            static_assert(!resolve_route(1, 1));    // forward
            static_assert(!resolve_route(0, 0));    // itself

            // and the whole finalize pass over a hand-built stand-in for layers 12-17
            constexpr bool finalizes_a_network()
            {
                ParsedCfg cfg{};
                using LayerType = LayerSpec::LayerType;
                cfg.layers.resize(5);
                cfg.layers[0] = {.type = LayerType::Conv, .size = 3};
                cfg.layers[1] = {.type = LayerType::Conv, .size = 1};
                cfg.layers[2] = {.type = LayerType::Maxpool, .size = 2};
                cfg.layers[3] = {.type = LayerType::Upsample};
                cfg.layers[4] = {.type = LayerType::Route, .route_layers = {-4, 1}};

                std::vector<RawPad> raw(5);
                raw[0].flag = 1;  // pad=1 size=3 -> 1
                raw[1].flag = 1;  // pad=1 size=1 -> 0

                return finalize(cfg, raw)
                    && cfg.layers[0].pad == 1
                    && cfg.layers[1].pad == 0
                    && cfg.layers[2].pad == 1
                    && cfg.layers[4].route_layers == std::vector<int>{0, 1};
            }
            static_assert(finalizes_a_network());
        }
#endif
    }

    std::expected<ParsedCfg, parse_error> parse_cfg(std::string_view path)
    {
        std::ifstream in{std::filesystem::path(path)};
        if (!in)
            return std::unexpected(parse_error::file_input_error);

        ParsedCfg cfg{};
        std::vector<RawPad> raw_pads;  // parallel to cfg.layers, discarded by finalize()
        bool in_net = false;
        bool in_section = false;

        std::string line;
        while (std::getline(in, line))
        {
            const std::string_view text = trim(line);
            if (text.empty() || text.front() == '#' || text.front() == ';')
                continue;

            if (text.front() == '[')
            {
                const auto close = text.find(']');
                if (close == std::string_view::npos || close + 1 != text.size())
                    return std::unexpected(parse_error::invalid_input);

                const std::string_view name = text.substr(1, close - 1);
                if (name == "net")
                {
                    if (in_section) // [net] only makes sense as the opening section
                        return std::unexpected(parse_error::invalid_input);
                    in_net = true;
                }
                else
                {
                    const auto type = to_layer_type(name);
                    if (!type)
                        return std::unexpected(parse_error::invalid_input);
                    in_net = false;
                    cfg.layers.emplace_back().type = *type;
                    raw_pads.emplace_back();
                }
                in_section = true;
                continue;
            }

            if (!in_section)
                return std::unexpected(parse_error::invalid_input);

            const auto equals = text.find('=');
            if (equals == std::string_view::npos)
                return std::unexpected(parse_error::invalid_input);

            const std::string_view key = trim(text.substr(0, equals));
            const std::string_view value = trim(text.substr(equals + 1));
            if (key.empty() || value.empty())
                return std::unexpected(parse_error::invalid_input);

            const bool applied = in_net ? apply_net(cfg.net, key, value)
                                        : apply_layer(cfg.layers.back(), raw_pads.back(), key, value);
            if (!applied)
                return std::unexpected(parse_error::invalid_input);
        }

        if (in.bad() || cfg.layers.empty())
            return std::unexpected(parse_error::invalid_input);

        if (!finalize(cfg, raw_pads))
            return std::unexpected(parse_error::invalid_input);

        return cfg;
    }
}