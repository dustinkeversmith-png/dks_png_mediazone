#ifndef SBD_PROVIDER_HPP
#define SBD_PROVIDER_HPP

#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace datasets {

class SBDProvider {
public:
    explicit SBDProvider(const std::filesystem::path& root_dir)
        : root_dir_(root_dir) {}

    virtual ~SBDProvider() = default;

    virtual bool initialize() = 0;
    virtual size_t size() const = 0;

protected:
    std::filesystem::path root_dir_;
};

} // namespace datasets

#endif // SBD_PROVIDER_HPP
