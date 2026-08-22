#ifndef LVIS_PROVIDER_HPP
#define LVIS_PROVIDER_HPP

#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace datasets {

class LVISProvider {
public:
    explicit LVISProvider(const std::filesystem::path& root_dir)
        : root_dir_(root_dir) {}

    virtual ~LVISProvider() = default;

    virtual bool initialize() = 0;
    virtual size_t size() const = 0;

protected:
    std::filesystem::path root_dir_;
};

} // namespace datasets

#endif // LVIS_PROVIDER_HPP
