#ifndef DIS5K_PROVIDER_HPP
#define DIS5K_PROVIDER_HPP

#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace datasets {

class DIS5KProvider {
public:
    explicit DIS5KProvider(const std::filesystem::path& root_dir)
        : root_dir_(root_dir) {}

    virtual ~DIS5KProvider() = default;

    virtual bool initialize() = 0;
    virtual size_t size() const = 0;

protected:
    std::filesystem::path root_dir_;
};

} // namespace datasets

#endif // DIS5K_PROVIDER_HPP
