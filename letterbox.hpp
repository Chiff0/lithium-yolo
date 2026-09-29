#pragma once

namespace lithium
{
    struct Letterbox { int fit_w, fit_h, pad_w, pad_h; };
    Letterbox letterbox_fit(int image_w, int image_h, int net_w, int net_h);
}