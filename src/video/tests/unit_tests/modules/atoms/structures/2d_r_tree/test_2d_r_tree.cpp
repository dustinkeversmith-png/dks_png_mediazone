#include "test_harness.hpp"
#include "spatial_trees/2d_r_tree/r_tree_2d.hpp"

#include <sstream>

// Atom demo: AABB list in → tree query hit lists out.
class RTreeAtom {
public:
    std::vector<vision::RTree2D::Item> items;
    AtomDemoReport report{"2d_r_tree"};
    std::ostringstream boxes_tsv;
    std::ostringstream queries_tsv;
    std::vector<std::string> written;

    bool load(const std::string& root, const std::string& dataset, const std::string& sample_filter) {
        print_banner("load inputs: " + dataset);
        const auto path = vision::join_path(vision::dataset_dir(root, dataset), "index.tsv");
        int id = 0;
        boxes_tsv << "id\tfile\tlabel\tx\ty\tw\th\n";
        for (const auto& row : vision::load_tsv(path)) {
            if (row.fields.size() < 6) {
                continue;
            }
            if (!sample_filter.empty() && row.file.find(sample_filter) == std::string::npos &&
                row.label.find(sample_filter) == std::string::npos) {
                continue;
            }
            vision::RTree2D::Item it;
            it.id = id++;
            it.box.x = std::strtof(row.fields[2].c_str(), nullptr);
            it.box.y = std::strtof(row.fields[3].c_str(), nullptr);
            it.box.w = std::strtof(row.fields[4].c_str(), nullptr);
            it.box.h = std::strtof(row.fields[5].c_str(), nullptr);
            items.push_back(it);
            boxes_tsv << it.id << '\t' << row.file << '\t' << row.label << '\t' << it.box.x << '\t'
                      << it.box.y << '\t' << it.box.w << '\t' << it.box.h << '\n';
        }
        std::cout << "loaded " << items.size() << " boxes\n";
        report.n_inputs = static_cast<int>(items.size());
        return !items.empty();
    }

    void run(const std::string& /*art_dir*/) {
        print_banner("run R-tree → query hit lists");
        ScopedTimer timer(&report.elapsed_ms);
        vision::RTree2D tree;
        for (const auto& it : items) {
            tree.insert(it.id, it.box);
        }
        tree.build();
        queries_tsv << "qx\tqy\tqw\tqh\thit_ids\n";
        const std::vector<vision::Rect> queries = {
            {0, 0, 200, 200}, {400, 100, 250, 250}, {50, 300, 400, 80}, {700, 0, 180, 500},
        };
        for (const auto& q : queries) {
            auto hits = tree.query_intersects(q);
            std::cout << "  query [" << q.x << "," << q.y << " " << q.w << "x" << q.h
                      << "]  hits=" << hits.size() << "\n";
            queries_tsv << q.x << '\t' << q.y << '\t' << q.w << '\t' << q.h << '\t';
            for (size_t i = 0; i < hits.size(); ++i) {
                if (i) {
                    queries_tsv << ',';
                }
                queries_tsv << hits[i];
            }
            queries_tsv << '\n';
            ++report.n_outputs;
        }
        report.notes.push_back("outputs: boxes.tsv, query_hits.tsv");
    }

    void write(const std::string& dir) {
        vision::write_text_file(vision::join_path(dir, "boxes.tsv"), boxes_tsv.str());
        vision::write_text_file(vision::join_path(dir, "query_hits.tsv"), queries_tsv.str());
        written = {"boxes.tsv", "query_hits.tsv"};
        write_atom_manifest(dir, report, written);
        report.print();
        std::cout << "artifacts -> " << dir << "\n";
    }
};

int main(int argc, char** argv) {
    constexpr const char* kDataset = "unit_rtree";
    return run_atom_main(argc, argv, kDataset, [&](const AtomCli& cli) -> int {
        RTreeAtom atom;
        if (!atom.load(cli.data_root, cli.dataset, cli.sample_filter)) {
            std::cerr << "no inputs for " << cli.dataset << " under " << cli.data_root << "\n";
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
