// 1. **Pre-emphasis Filter:** Apply a first-order high-pass filter ($y[n] = x[n] - \alpha x[n-1]$, where $\alpha \approx 0.95 - 0.97$) to compensate for the $-6\text{ dB/octave}$ glottal roll-off.
#pragma once
#include <vector>

inline std::vector<float> pre_emphasis_filter(const std::vector<float>& buffer)
{
    if (buffer.empty()) return {};

    std::vector<float> out(buffer.size());
    const float alpha = 0.95f;

    // The first sample has no x[n-1], so we keep it as is or use a historical sample
    out[0] = buffer[0]; 

    // Start loop from index 1 to avoid buffer[-1] out-of-bounds crash
    for (size_t i = 1; i < buffer.size(); i++)
    {
        out[i] = buffer[i] - (alpha * buffer[i - 1]); // Multiplied by alpha
    }

    return out; // Added missing return statement
}