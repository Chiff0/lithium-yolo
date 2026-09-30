#include "letterbox.hpp"


namespace lithium
{
    Letterbox letterbox_fit(int image_w, int image_h, int net_w, int net_h)
    {
        Letterbox letterbox{};

        if (static_cast<float>(image_w) / net_w > static_cast<float>(image_h) / net_h)
        {
            letterbox.fit_w = net_w;
            letterbox.fit_h = image_h * net_w / image_w ;
        }
        else
        {
            letterbox.fit_h = net_h;
            letterbox.fit_w = image_w * net_h / image_h ;
        }
        letterbox.pad_w = (net_w - letterbox.fit_w) / 2;
        letterbox.pad_h = (net_h - letterbox.fit_h) / 2;

        return letterbox;
        
    }
}