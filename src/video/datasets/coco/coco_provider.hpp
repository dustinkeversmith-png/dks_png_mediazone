#ifndef COCO_PROVIDER_HPP
#define COCO_PROVIDER_HPP

#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace datasets {

class COCOProvider {
public:
    explicit COCOProvider(const std::filesystem::path& root_dir)
        : root_dir_(root_dir) {}

    virtual ~COCOProvider() = default;

    virtual bool initialize() = 0;
    virtual size_t size() const = 0;

protected:
    std::filesystem::path root_dir_;
};

} // namespace datasets

#endif // COCO_PROVIDER_HPP
