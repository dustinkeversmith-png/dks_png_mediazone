#include "test_harness.hpp"
#include "spatial_trees/2d_r_tree/r_tree_2d.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>

class RTreeUnitTest {
public:
    vision::RTree2D tree;
    std::vector<vision::RTree2D::Item> items;
    AccuracyReport report{"2d_r_tree / AABB overlap"};
    std::ostringstream artifacts;

    bool load_corresponding_dataset(const std::string& root) {
        print_banner("load_corresponding_dataset: unit_rtree");
        const auto path = vision::join_path(vision::dataset_dir(root, "unit_rtree"), "index.tsv");
        std::ifstream in(path);
        std::string line;
        bool header = true;
        while (std::getline(in, line)) {
            if (header) {
                header = false;
                continue;
            }
            std::istringstream ss(line);
            vision::RTree2D::Item it;
            if (!(ss >> it.id >> it.box.x)) {
                continue;
            }
            // TSV: id cluster x y w h  — skip cluster if the second token is int cluster id
            float cluster_or_x = it.box.x;
            float y, w, h;
            if (ss >> y >> w >> h) {
                // parsed cluster as x, so shift
                it.box = {cluster_or_x, y, w, h};
                float extra;
                if (ss >> extra) {
                    it.box = {y, w, h, extra};
                }
            }
            items.push_back(it);
        }
        // Robust parse via fields
        items.clear();
        for (const auto& row : vision::load_tsv(path)) {
            if (row.fields.size() < 6) {
                continue;
            }
            vision::RTree2D::Item it;
            it.id = std::atoi(row.fields[0].c_str());
            it.box.x = std::strtof(row.fields[2].c_str(), nullptr);
            it.box.y = std::strtof(row.fields[3].c_str(), nullptr);
            it.box.w = std::strtof(row.fields[4].c_str(), nullptr);
            it.box.h = std::strtof(row.fields[5].c_str(), nullptr);
            items.push_back(it);
        }
        std::cout << "loaded " << items.size() << " bounding boxes\n";
        return !items.empty();
    }

    void run_analysis() {
        print_banner("run_analysis");
        ScopedTimer timer(&report.elapsed_ms);
        for (const auto& it : items) {
            tree.insert(it.id, it.box);
        }
        tree.build();
        const std::vector<vision::Rect> queries = {
            {0, 0, 200, 200}, {400, 100, 250, 250}, {50, 300, 400, 80}, {700, 0, 180, 500},
        };
        for (const auto& q : queries) {
            auto a = tree.query_intersects(q);
            auto b = tree.brute_intersects(q);
            std::sort(a.begin(), a.end());
            const bool ok = a == b;
            ++report.total;
            if (ok) {
                ++report.passed;
            }
            std::cout << "  query [" << q.x << "," << q.y << " " << q.w << "x" << q.h << "]  tree="
                      << a.size() << " brute=" << b.size() << (ok ? "  PASS\n" : "  FAIL\n");
            artifacts << q.x << "," << q.y << " hits=" << a.size() << " match=" << ok << "\n";
        }
        report.notes.push_back("R-tree overlap results must equal brute-force");
    }

    float evaluate_accuracy() {
        report.finalize();
        report.print();
        return static_cast<float>(report.accuracy);
    }

    void output_artifacts(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "rtree_queries.txt"), artifacts.str());
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : vision::find_data_root(argv[0]);
    std::cout << "data root: " << root << "\n";
    RTreeUnitTest test;
    if (!test.load_corresponding_dataset(root)) {
        return 1;
    }
    test.run_analysis();
    const float acc = test.evaluate_accuracy();
    test.output_artifacts(make_artifact_dir("2d_r_tree"));
    return acc >= 0.99f ? 0 : 2;
}
