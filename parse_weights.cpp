#include "parsers.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <vector>


namespace lithium
{
    namespace
    {
        template <typename T>
        bool read_raw(std::istream& in, T* dst, std::size_t count)
        {
            const auto bytes = static_cast<std::streamsize>(count * sizeof(T));
            in.read(reinterpret_cast<char*>(dst), bytes);
            return in.gcount() == bytes;
        }

        // parse_cfg has already turned every route entry into an absolute index
        // pointing strictly backwards; anything else is a hand-built cfg
        bool routes_backwards(int entry, std::size_t self)
        {
            return entry >= 0 && static_cast<std::size_t>(entry) < self;
        }
    }

    std::expected<ParsedWeights, parse_error> parse_weights(
        std::string_view path, const ParsedCfg& cfg, std::pmr::memory_resource* resource)
    {
        if (resource == nullptr || cfg.layers.empty() || cfg.net.channels <= 0)
            return std::unexpected(parse_error::invalid_input);

        std::ifstream in{std::filesystem::path(path), std::ios::binary};
        if (!in)
            return std::unexpected(parse_error::file_input_error);

        in.seekg(0, std::ios::end);
        const std::streamoff file_size = in.tellg();
        in.seekg(0, std::ios::beg);
        if (file_size < 0 || !in)
            return std::unexpected(parse_error::file_input_error);

        ParsedWeights out{resource};

        std::int32_t version[3]{};
        if (!read_raw(in, version, 3))
            return std::unexpected(parse_error::invalid_input);
        out.major = version[0];
        out.minor = version[1];
        out.revision = version[2];
        if (out.major < 0 || out.minor < 0)
            return std::unexpected(parse_error::invalid_input);

        // 0.2 widened the "images seen" counter from 32 to 64 bits
        std::uint64_t header_bytes = 3 * sizeof(std::int32_t);
        if (out.major > 0 || out.minor >= 2)
        {
            std::uint64_t seen{};
            if (!read_raw(in, &seen, 1))
                return std::unexpected(parse_error::invalid_input);
            out.seen = seen;
            header_bytes += sizeof(seen);
        }
        else
        {
            std::uint32_t seen{};
            if (!read_raw(in, &seen, 1))
                return std::unexpected(parse_error::invalid_input);
            out.seen = seen;
            header_bytes += sizeof(seen);
        }

        const auto total_bytes = static_cast<std::uint64_t>(file_size);
        if (total_bytes < header_bytes)
            return std::unexpected(parse_error::invalid_input);
        const std::uint64_t payload_bytes = total_bytes - header_bytes;
        if (payload_bytes % sizeof(float) != 0)
            return std::unexpected(parse_error::invalid_input);

        // every read is bounded by what the file still holds, so a bogus cfg
        // cannot ask us to allocate more than the blob could possibly contain
        std::uint64_t remaining = payload_bytes / sizeof(float);
        const auto take = [&](std::pmr::vector<float>& dst, std::uint64_t count)
        {
            if (count > remaining)
                return false;
            dst.resize(static_cast<std::size_t>(count));
            if (count != 0 && !read_raw(in, dst.data(), static_cast<std::size_t>(count)))
                return false;
            remaining -= count;
            return true;
        };

        out.layers.resize(cfg.layers.size());
        std::vector<int> out_channels(cfg.layers.size(), 0);
        int in_channels = cfg.net.channels;

        for (std::size_t i = 0; i < cfg.layers.size(); ++i)
        {
            const LayerSpec& layer = cfg.layers[i];
            int produced = in_channels;  // most layers leave the channel count alone

            switch (layer.type)
            {
            case LayerSpec::LayerType::Conv:
            {
                if (layer.filters <= 0 || layer.size <= 0 || in_channels <= 0)
                    return std::unexpected(parse_error::invalid_input);

                const auto filters = static_cast<std::uint64_t>(layer.filters);
                const auto kernel = static_cast<std::uint64_t>(layer.size);
                const auto count = filters * static_cast<std::uint64_t>(in_channels) * kernel * kernel;

                // darknet's order: biases, then the batch-norm triple, then the kernels
                LayerWeights& weights = out.layers[i];
                if (!take(weights.biases, filters))
                    return std::unexpected(parse_error::invalid_input);
                if (layer.batch_norm
                    && !(take(weights.scales, filters)
                         && take(weights.rolling_mean, filters)
                         && take(weights.rolling_variance, filters)))
                {
                    return std::unexpected(parse_error::invalid_input);
                }
                if (!take(weights.weights, count))
                    return std::unexpected(parse_error::invalid_input);

                produced = layer.filters;
                break;
            }
            case LayerSpec::LayerType::Route:
            {
                if (layer.route_layers.empty() || layer.route_groups <= 0)
                    return std::unexpected(parse_error::invalid_input);

                long long concatenated = 0;
                for (int entry : layer.route_layers)
                {
                    if (!routes_backwards(entry, i))
                        return std::unexpected(parse_error::invalid_input);
                    concatenated += out_channels[entry];
                }
                if (concatenated <= 0 || concatenated % layer.route_groups != 0)
                    return std::unexpected(parse_error::invalid_input);

                produced = static_cast<int>(concatenated / layer.route_groups);
                break;
            }
            case LayerSpec::LayerType::Maxpool:
            case LayerSpec::LayerType::Upsample:
            case LayerSpec::LayerType::Yolo:
                break;
            }

            out_channels[i] = produced;
            in_channels = produced;
        }

        // leftover floats mean the cfg and the blob describe different networks
        if (remaining != 0)
            return std::unexpected(parse_error::invalid_input);

        return out;
    }
}
