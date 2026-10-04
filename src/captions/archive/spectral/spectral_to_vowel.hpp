#pragma once
#ifndef SPECTRAL_TO_VOWEL_HPP
#define SPECTRAL_TO_VOWEL_HPP

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "spectral_analysis.hpp"

struct SpectralVowelEntry {
    std::string key;           // "IY", "AE", etc.
    std::string ipa;           // "i", "æ"
    std::string example;       // "beet", "bat"
    float centroid_hz;         // Expected spectral centroid in Hz
    float pitch_range_low;     // Typical F0 range (male speaker)
    float pitch_range_high;
    float spectral_tilt_db;    // Expected tilt in dB/octave (negative = rolloff)
    float tilt_tolerance;      // Acceptable variation in tilt
};

class SpectralVowelDB {
private:
    std::vector<SpectralVowelEntry> db;

public:
    SpectralVowelDB() {
        // Initialize spectral vowel dataset
        // Centroid values based on frequency distributions of formants
        // Tilt values: vowels have -12 to -6 dB/octave rolloff; high consonants near 0 or positive
        db = {
            // Front Vowels (high F2, rising spectral content)
            {"IY", "i",  "beet",   1850.0f,  100.0f, 130.0f,  -9.0f,  2.0f},
            {"IH", "ɪ",  "bit",    1700.0f,   95.0f, 125.0f,  -9.0f,  2.0f},
            {"EY", "eɪ", "bait",   1600.0f,   90.0f, 120.0f, -10.0f,  2.0f},
            {"EH", "ɛ",  "bet",    1500.0f,   85.0f, 115.0f, -10.0f,  2.0f},
            {"AE", "æ",  "bat",    1400.0f,   85.0f, 115.0f, -11.0f,  2.0f},

            // Central Vowels
            {"AH", "ʌ",  "butt",   1200.0f,   85.0f, 115.0f, -11.0f,  2.0f},
            {"ER", "ɝ",  "bird",   1150.0f,   90.0f, 120.0f, -11.0f,  2.0f}, // Rhotic
            {"AX", "ə",  "about",  1300.0f,   80.0f, 110.0f, -11.0f,  2.0f},

            // Back Vowels (lower centroid, strong rolloff)
            {"AA", "ɑ",  "father",  900.0f,   80.0f, 110.0f, -12.0f,  2.0f},
            {"AO", "ɔ",  "bought",  950.0f,   75.0f, 105.0f, -12.0f,  2.0f},
            {"OW", "oʊ", "boat",   1050.0f,   75.0f, 105.0f, -12.0f,  2.0f},
            {"UH", "ʊ",  "book",   1100.0f,   70.0f, 100.0f, -12.0f,  2.0f},
            {"UW", "u",  "boot",   1150.0f,   70.0f, 100.0f, -12.0f,  2.0f},
        };
    }

    struct MatchResult {
        std::string key;
        std::string ipa;
        float distance;
        bool in_bounds;
    };

    // Find closest vowel using spectral features
    MatchResult find_nearest_phoneme(const SpectralFeatures& features) const {
        if (!features.is_voiced) {
            return {"SIL", "", 0.0f, false}; // Unvoiced / Silence
        }

        float min_dist = 1.0e30f;
        const SpectralVowelEntry* best_match = nullptr;

        for (const auto& entry : db) {
            // Distance components: normalized L2 norm in feature space
            
            // 1. Centroid distance (Hz scale, normalized by typical range ~500-2000 Hz)
            float centroid_diff = features.centroid_hz - entry.centroid_hz;
            float centroid_dist = (centroid_diff * centroid_diff) / (400.0f * 400.0f); // ~400 Hz tolerance

            // 2. Pitch distance (Hz scale, normalized by typical range ~70-400 Hz for male speaker)
            float pitch_tolerance = (entry.pitch_range_high - entry.pitch_range_low) / 2.0f + 20.0f;
            float pitch_mid = (entry.pitch_range_low + entry.pitch_range_high) / 2.0f;
            float pitch_diff = features.pitch_f0_hz - pitch_mid;
            float pitch_dist = (pitch_diff * pitch_diff) / (pitch_tolerance * pitch_tolerance);

            // 3. Spectral tilt distance (dB/octave scale, normalized by ~2 dB tolerance)
            float tilt_diff = features.spectral_tilt - entry.spectral_tilt_db;
            float tilt_dist = (tilt_diff * tilt_diff) / (entry.tilt_tolerance * entry.tilt_tolerance);

            // Weighted combination (centroid most important, tilt less so)
            float distance = std::sqrt(centroid_dist * 2.0f + pitch_dist * 0.5f + tilt_dist * 0.3f);

            if (distance < min_dist) {
                min_dist = distance;
                best_match = &entry;
            }
        }

        if (best_match) {
            // Check bounds: centroid within ±500 Hz, pitch within typical range
            float centroid_err = std::abs(features.centroid_hz - best_match->centroid_hz);
            float pitch_range = best_match->pitch_range_high - best_match->pitch_range_low;
            float pitch_err = std::abs(features.pitch_f0_hz - (best_match->pitch_range_low + best_match->pitch_range_high) / 2.0f);
            
            bool within_bounds = (centroid_err <= 500.0f) && (pitch_err <= pitch_range);

            return {best_match->key, best_match->ipa, min_dist, within_bounds};
        }

        return {"UNK", "", min_dist, false};
    }
};

#endif // SPECTRAL_TO_VOWEL_HPP