#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

struct FormantEntry {
    std::string key;          // "IY", "AE", etc.
    std::string ipa;          // "i", "æ"
    std::string example;      // "beet", "bat"
    float f1;                 // Target F1 in Hz
    float f2;                 // Target F2 in Hz
    float f3;                 // Target F3 in Hz
    float tolerance;          // Radius / Max confidence boundary
};

// Convert Linear Hz to Bark Scale (Traunmüller formula) for perceptual distance
inline float hz_to_bark(float f) {
    return (26.81f * f) / (1960.0f + f) - 0.53f;
}

class FormantVectorDB {
private:
    std::vector<FormantEntry> db;

public:
    FormantVectorDB() {
        // Initialize the vector dataset
        db = {
            // Front Vowels
            {"IY", "i",  "beet",   270.0f, 2290.0f, 3010.0f, 80.0f},
            {"IH", "ɪ",  "bit",    390.0f, 1990.0f, 2550.0f, 90.0f},
            {"EY", "eɪ", "bait",   530.0f, 1840.0f, 2480.0f, 90.0f},
            {"EH", "ɛ",  "bet",    660.0f, 1720.0f, 2410.0f, 100.0f},
            {"AE", "æ",  "bat",    730.0f, 1090.0f, 2440.0f, 110.0f},

            // Central Vowels
            {"AH", "ʌ",  "butt",   640.0f, 1190.0f, 2390.0f, 100.0f},
            {"ER", "ɝ",  "bird",   490.0f, 1350.0f, 1690.0f, 90.0f}, // Low F3
            {"AX", "ə",  "about",  500.0f, 1500.0f, 2500.0f, 100.0f},

            // Back Vowels
            {"AA", "ɑ",  "father", 730.0f, 1090.0f, 2440.0f, 110.0f},
            {"AO", "ɔ",  "bought", 570.0f,  840.0f, 2410.0f, 100.0f},
            {"OW", "oʊ", "boat",   500.0f, 1000.0f, 2350.0f, 90.0f},
            {"UH", "ʊ",  "book",   440.0f, 1020.0f, 2240.0f, 90.0f},
            {"UW", "u",  "boot",   300.0f,  870.0f, 2240.0f, 80.0f}
        };
    }

    struct MatchResult {
        std::string key;
        std::string ipa;
        float distance;
        bool in_bounds;
    };

    // Finds the closest phoneme key using Bark-scaled acoustic distance
    MatchResult find_nearest_phoneme(float measured_f1, float measured_f2, float measured_f3) const {
        if (measured_f1 <= 0.0f || measured_f2 <= 0.0f) {
            return {"SIL", "", 0.0f, false}; // Silence / Invalid frame
        }

        float min_dist = std::numeric_limits<float>::max();
        const FormantEntry* best_match = nullptr;

        // Convert inputs to Bark scale
        float b1 = hz_to_bark(measured_f1);
        float b2 = hz_to_bark(measured_f2);
        float b3 = (measured_f3 > 0.0f) ? hz_to_bark(measured_f3) : 0.0f;

        for (const auto& entry : db) {
            float target_b1 = hz_to_bark(entry.f1);
            float target_b2 = hz_to_bark(entry.f2);
            float target_b3 = hz_to_bark(entry.f3);

            // Perceptual Bark distance: F1 and F2 drive the vowel identity most strongly
            float d1 = b1 - target_b1;
            float d2 = b2 - target_b2;
            
            // F3 weighting is reduced unless it's rhotic (/ɝ/)
            float d3 = (measured_f3 > 0.0f) ? (b3 - target_b3) * 0.35f : 0.0f;

            // Weighted Bark Distance
            float distance = std::sqrt(d1 * d1 * 1.5f + d2 * d2 * 1.0f + d3 * d3);

            if (distance < min_dist) {
                min_dist = distance;
                best_match = &entry;
            }
        }

        if (best_match) {
            // Check if within acceptable tolerance bounds in standard Hz space
            float f1_err = std::abs(measured_f1 - best_match->f1);
            float f2_err = std::abs(measured_f2 - best_match->f2);
            bool within_bounds = (f1_err <= best_match->tolerance * 1.5f) && 
                                 (f2_err <= best_match->tolerance * 2.0f);

            return {best_match->key, best_match->ipa, min_dist, within_bounds};
        }

        return {"UNK", "", min_dist, false};
    }
};