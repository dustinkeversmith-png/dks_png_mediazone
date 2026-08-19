#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct IgnoreRule {
    std::string pattern;
    bool negated = false;
    bool directory_only = false;
    bool has_slash = false;
    std::regex expression{std::regex("$a")};
};

struct Options {
    bool use_defaults = true;
    std::vector<std::string> extra_patterns;
};

static std::string trim(std::string value) {
    while (!value.empty() && (value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
        value.pop_back();
    }
    size_t first = 0;
    while (first < value.size() && (value[first] == ' ' || value[first] == '\t')) {
        ++first;
    }
    return value.substr(first);
}

static std::string glob_to_regex(std::string_view pattern) {
    std::string result = "^";
    for (size_t i = 0; i < pattern.size(); ++i) {
        const char character = pattern[i];
        if (character == '*') {
            if (i + 1 < pattern.size() && pattern[i + 1] == '*') {
                ++i;
                if (i + 1 < pattern.size() && pattern[i + 1] == '/') {
                    ++i;
                    result += "(?:.*/)?";
                } else {
                    result += ".*";
                }
            } else {
                result += "[^/]*";
            }
        } else if (character == '?') {
            result += "[^/]";
        } else if (character == '[') {
            const size_t closing = pattern.find(']', i + 1);
            if (closing != std::string_view::npos) {
                result.append(pattern.substr(i, closing - i + 1));
                i = closing;
            } else {
                result += "\\[";
            }
        } else {
            if (std::string_view("\\.^$|(){}+").find(character) != std::string_view::npos) {
                result += '\\';
            }
            result += character;
        }
    }
    return result + "$";
}

static bool parse_rule(std::string line, IgnoreRule& rule) {
    line = trim(std::move(line));
    if (line.empty() || line[0] == '#') {
        return false;
    }

    bool escaped_leading_character = line.size() > 1 && line[0] == '\\' && (line[1] == '#' || line[1] == '!');
    if (line[0] == '!' && !escaped_leading_character) {
        rule.negated = true;
        line.erase(0, 1);
    }
    if (escaped_leading_character) {
        line.erase(0, 1);
    }
    if (line.empty()) {
        return false;
    }

    rule.directory_only = line.back() == '/';
    if (rule.directory_only) {
        line.pop_back();
    }
    if (!line.empty() && line.front() == '/') {
        line.erase(0, 1);
    }
    rule.has_slash = line.find('/') != std::string::npos;
    rule.pattern = line;
    rule.expression = std::regex(glob_to_regex(line), std::regex::ECMAScript);
    return true;
}

static void load_ignore_file(const fs::path& file, std::vector<IgnoreRule>& rules) {
    std::ifstream input(file);
    std::string line;
    while (std::getline(input, line)) {
        IgnoreRule rule;
        try {
            if (parse_rule(line, rule)) {
                rules.push_back(std::move(rule));
            }
        } catch (const std::regex_error&) {
            std::cerr << "retree: invalid ignore pattern in " << file << ": " << line << '\n';
        }
    }
}

static bool matches(const IgnoreRule& rule, const std::string& relative_path, const std::string& name, bool is_directory) {
    if (rule.directory_only && !is_directory) {
        return false;
    }
    if (rule.has_slash) {
        return std::regex_match(relative_path, rule.expression);
    }
    return std::regex_match(name, rule.expression);
}

static bool is_ignored(const fs::path& root, const fs::path& entry, const std::vector<IgnoreRule>& rules) {
    const std::string relative_path = entry.lexically_relative(root).generic_string();
    const std::string name = entry.filename().generic_string();
    const bool is_directory = fs::is_directory(entry) && !fs::is_symlink(entry);
    bool ignored = false;
    for (const IgnoreRule& rule : rules) {
        if (matches(rule, relative_path, name, is_directory)) {
            ignored = !rule.negated;
        }
    }
    return ignored;
}

static std::vector<IgnoreRule> default_rules() {
    const std::vector<std::string> patterns = {
        ".git/", ".svn/", ".hg/", ".vscode/", ".idea/",
        "node_modules/", "bower_components/", "jspm_packages/", "vendor/",
        "venv/", ".venv/", "env/", ".env/", "dsp_env/", "virtualenv/",
        "__pycache__/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
        ".tox/", ".nox/", ".ipynb_checkpoints/", "site-packages/", "*.egg-info/",
        "build/", "Build/", "dist/", "Dist/", "out/", "target/", "bin/", "Bin/",
        "obj/", "Obj/", "lib/", "Lib/", "include/", "Include/", "Scripts/",
        "coverage/", ".coverage", ".cache/", "__MACOSX/", ".next/", ".nuxt/",
        ".turbo/", ".parcel-cache/", ".gradle/", ".terraform/", "cmake-build-*/",
        "*.pyc", "*.pyo", "*.pyd", "*.dll", "*.so", "*.a", "*.o", "*.obj",
        "*.whl", "*.log"
    };
    std::vector<IgnoreRule> rules;
    for (const std::string& pattern : patterns) {
        IgnoreRule rule;
        parse_rule(pattern, rule);
        rules.push_back(std::move(rule));
    }
    return rules;
}

static void print_tree(const fs::path& root, const fs::path& directory, const std::string& prefix,
                       const std::vector<IgnoreRule>& rules) {
    std::vector<fs::directory_entry> entries;
    std::error_code error;
    for (fs::directory_iterator iterator(directory, fs::directory_options::skip_permission_denied, error), end;
         iterator != end; iterator.increment(error)) {
        if (!error && !is_ignored(root, iterator->path(), rules)) {
            entries.push_back(*iterator);
        }
        error.clear();
    }
    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
        return left.path().filename().generic_string() < right.path().filename().generic_string();
    });

    for (size_t index = 0; index < entries.size(); ++index) {
        const bool last = index + 1 == entries.size();
        const fs::path path = entries[index].path();
        std::cout << prefix << (last ? "`-- " : "|-- ") << path.filename().generic_string() << '\n';
        if (entries[index].is_directory(error) && !entries[index].is_symlink(error)) {
            print_tree(root, path, prefix + (last ? "    " : "|   "), rules);
        }
        error.clear();
    }
}

static void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [directory] [options]\n"
              << "Options:\n"
              << "  --no-defaults       Do not ignore standard dependency/build directories\n"
              << "  --ignore PATTERN    Add an ignore pattern (repeatable)\n"
              << "  -h, --help          Show this help\n";
}

int main(int argc, char* argv[]) {
    fs::path root = fs::current_path();
    Options options;
    bool directory_seen = false;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--no-defaults") {
            options.use_defaults = false;
        } else if (argument == "--ignore" && index + 1 < argc) {
            options.extra_patterns.push_back(argv[++index]);
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!directory_seen && argument.rfind("-", 0) != 0) {
            root = argument;
            directory_seen = true;
        } else {
            std::cerr << "retree: unknown or incomplete argument: " << argument << '\n';
            print_usage(argv[0]);
            return 2;
        }
    }

    std::error_code error;
    root = fs::weakly_canonical(root, error);
    if (error || !fs::is_directory(root, error)) {
        std::cerr << "retree: not a directory: " << root << '\n';
        return 1;
    }

    std::vector<IgnoreRule> rules = options.use_defaults ? default_rules() : std::vector<IgnoreRule>{};
    load_ignore_file(root / ".ignore", rules);
    for (const std::string& pattern : options.extra_patterns) {
        IgnoreRule rule;
        try {
            if (parse_rule(pattern, rule)) {
                rules.push_back(std::move(rule));
            }
        } catch (const std::regex_error&) {
            std::cerr << "retree: invalid ignore pattern: " << pattern << '\n';
            return 2;
        }
    }

    std::cout << root.filename().generic_string() << '\n';
    print_tree(root, root, "", rules);
    return 0;
}
