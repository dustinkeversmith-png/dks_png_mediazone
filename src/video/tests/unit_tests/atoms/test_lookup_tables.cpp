#include "test_harness.hpp"
#include "lookup_tables/geometry_lut.hpp"

#include <sstream>

class LutUnitTest {
public:
    std::vector<LoadedSample> samples;
    vision::GeometryLUT lut;
    AccuracyReport report{"lookup_tables / aspect-compactness"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_luts");
        samples = load_png_dataset(vision::dataset_dir(root, "unit_luts"));
        std::cout << "loaded " << samples.size() << " LUT probes\n";
        return !samples.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        std::vector<vision::GeometryLUT::Stats> stats;
        for (const auto& sample : samples) {
            stats.push_back(vision::GeometryLUT::measure(sample.image));
            lut.insert(stats.back(), sample.row.label);
        }
        for (size_t i = 0; i < samples.size(); ++i) {
            const std::string pred = lut.query(stats[i]);
            const bool ok = pred == samples[i].row.label;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            std::cout << "  " << samples[i].row.file << "  aspect=" << stats[i].aspect
                      << "  compact=" << stats[i].compactness << "  pred=" << pred
                      << (ok ? "  PASS\n" : "  FAIL\n");
            artifacts << samples[i].row.file << "\t" << stats[i].aspect << "\t"
                      << stats[i].compactness << "\t" << pred << "\n";
        }
        report.notes.push_back("self-insert then query: LUT must recover the stored bucket label");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "lut_hits.tsv"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    LutUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("lookup_tables"));
    return acc >= 0.8f ? 0 : 2;
}
