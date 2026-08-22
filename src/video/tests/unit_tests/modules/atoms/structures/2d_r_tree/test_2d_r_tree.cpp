#include "test_harness.hpp"
#include "mission_helpers.hpp"
#include "structures/2d_r_tree/r_tree_2d.hpp"
#include "datasets/synthetic/synthetic_provider.hpp"

#include <sstream>

class RTreeAtom {
public:
    std::vector<vision::RTree2D::Item> items;
    AtomDemoReport report{"2d_r_tree"};
    std::ostringstream boxes_tsv;
    std::ostringstream queries_tsv;
    std::vector<std::string> written;

    bool load(const AtomCli& cli, int argc, char** argv) {
        print_banner("load mission samples");
        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, 1, 0);
        boxes_tsv << "id\tfile\tlabel\tx\ty\tw\th\n";
        const auto vision_root = fs::path(vision::find_vision_root(argc > 0 ? argv[0] : nullptr));
        datasets::SyntheticProvider provider(vision_root / "synthetic", datasets::SyntheticKind::Boxes);
        provider.initialize();
        const auto boxes = provider.load_boxes(0);
        int id = 0;
        for (const auto& box : boxes) {
            vision::RTree2D::Item it;
            it.id = id++;
            it.box = box;
            items.push_back(it);
            boxes_tsv << it.id << "\tbox_" << it.id << "\tsynthetic\t" << it.box.x << '\t' << it.box.y
                      << '\t' << it.box.w << '\t' << it.box.h << '\n';
        }
        if (items.empty()) {
            const auto path = vision::join_path(vision::dataset_dir(cli.data_root, cli.dataset), "index.tsv");
            for (const auto& row : vision::load_tsv(path)) {
                if (row.fields.size() < 6) {
                    continue;
                }
                vision::RTree2D::Item it;
                it.id = id++;
                it.box.x = std::strtof(row.fields[2].c_str(), nullptr);
                it.box.y = std::strtof(row.fields[3].c_str(), nullptr);
                it.box.w = std::strtof(row.fields[4].c_str(), nullptr);
                it.box.h = std::strtof(row.fields[5].c_str(), nullptr);
                items.push_back(it);
            }
        }
        std::cout << "loaded " << items.size() << " boxes via " << mission.provider_name << "\n";
        report.n_inputs = static_cast<int>(items.size());
        return !items.empty();
    }

    void run(const std::string& art_dir) {
        print_banner("run R-tree → query hit lists");
        ScopedTimer timer(&report.elapsed_ms);
        vision::RTree2D tree;
        for (const auto& it : items) {
            tree.insert(it.id, it.box);
        }
        tree.build();
        queries_tsv << "qx\tqy\tqw\tqh\thit_ids\tbrute_match\n";
        const std::vector<vision::Rect> queries = {
            {0, 0, 200, 200}, {400, 100, 250, 250}, {50, 300, 400, 80}, {700, 0, 180, 500},
        };
        bool all_match = true;
        for (const auto& q : queries) {
            auto hits = tree.query_intersects(q);
            const auto brute = tree.brute_intersects(q);
            const bool match = hits == brute;
            all_match = all_match && match;
            std::cout << "  query [" << q.x << "," << q.y << " " << q.w << "x" << q.h
                      << "]  hits=" << hits.size() << "  exact=" << match << "\n";
            queries_tsv << q.x << '\t' << q.y << '\t' << q.w << '\t' << q.h << '\t';
            for (size_t i = 0; i < hits.size(); ++i) {
                if (i) {
                    queries_tsv << ',';
                }
                queries_tsv << hits[i];
            }
            queries_tsv << '\t' << (match ? 1 : 0) << '\n';
            ++report.n_outputs;
        }
        mission::write_rtree_json(vision::join_path(art_dir, "rtree_hierarchy.json"), items.size(),
                                  queries.size(), all_match);
        written.push_back("rtree_hierarchy.json");
        report.notes.push_back("artifacts: query_bench.tsv, rtree_hierarchy.json, boxes.tsv");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "boxes.tsv"), boxes_tsv.str());
        vision::write_text_file(vision::join_path(dir, "query_bench.tsv"), queries_tsv.str());
        written.insert(written.begin(), "query_bench.tsv");
        written.insert(written.begin(), "boxes.tsv");
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    return run_atom_main(argc, argv, "synthetic_boxes", [&](const AtomCli& cli) -> int {
        RTreeAtom atom;
        if (!atom.load(cli, argc, argv)) {
            std::cerr << "no box inputs for r-tree atom\n";
            return 1;
        }
        if (cli.list_only) {
            std::cout << "  boxes=" << atom.items.size() << "\n";
            return 0;
        }
        const std::string art = make_artifact_dir(cli.artifact_dir);
        atom.run(art);
        atom.write(art);
        return 0;
    });
}
