#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vocal::cli {
class Options {
public:
    bool help{};
    Options(int argc, char** argv, std::initializer_list<std::string> allowed) {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") { help = true; continue; }
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
                throw std::invalid_argument("unknown option: " + key);
            if (++i == argc || std::string(argv[i]).rfind("--", 0) == 0)
                throw std::invalid_argument("missing value for " + key);
            if (!values_.emplace(key, argv[i]).second) throw std::invalid_argument("duplicate option: " + key);
        }
    }
    [[nodiscard]] bool has(const std::string& key) const { return values_.contains(key); }
    [[nodiscard]] std::string get(const std::string& key, std::string fallback = {}) const {
        const auto value = values_.find(key);
        return value == values_.end() ? fallback : value->second;
    }
    [[nodiscard]] std::string required(const std::string& key) const {
        const auto value = get(key);
        if (value.empty()) throw std::invalid_argument("required option: " + key);
        return value;
    }
    template<class T> [[nodiscard]] T number(const std::string& key, T fallback) const {
        return has(key) ? parse<T>(get(key)) : fallback;
    }
    template<class T> static T parse(const std::string& value) {
        T result{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
            throw std::invalid_argument("invalid numeric value: " + value);
        return result;
    }
private:
    std::map<std::string, std::string> values_;
};

template<class T> std::vector<T> read_values(const std::filesystem::path& path, std::size_t limit) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read control file: " + path.string());
    std::vector<T> values;
    for (std::string line; std::getline(input, line);) {
        line.resize(line.find('#') == std::string::npos ? line.size() : line.find('#'));
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        for (std::string value; row >> value;) {
            if (values.size() == limit) throw std::invalid_argument("control file exceeds value limit");
            values.push_back(Options::parse<T>(value));
        }
    }
    return values;
}

inline std::ofstream report_file(std::filesystem::path output) {
    output.replace_extension(".diagnostics.json");
    std::ofstream report(output);
    if (!report) throw std::runtime_error("cannot write diagnostics: " + output.string());
    return report;
}

template<class T> void array(std::ostream& out, const std::vector<T>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}
} // namespace vocal::cli
