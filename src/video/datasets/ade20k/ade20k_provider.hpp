#ifndef ADE20K_PROVIDER_HPP
#define ADE20K_PROVIDER_HPP

#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace datasets {

class ADE20KProvider {
public:
    explicit ADE20KProvider(const std::filesystem::path& root_dir)
        : root_dir_(root_dir) {}

    virtual ~ADE20KProvider() = default;

    virtual bool initialize() = 0;
    virtual size_t size() const = 0;

protected:
    std::filesystem::path root_dir_;
};

} // namespace datasets

#endif // ADE20K_PROVIDER_HPP
