#ifndef BSDS500_PROVIDER_HPP
#define BSDS500_PROVIDER_HPP

#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace datasets {

class BSDS500Provider {
public:
    explicit BSDS500Provider(const std::filesystem::path& root_dir)
        : root_dir_(root_dir) {}

    virtual ~BSDS500Provider() = default;

    virtual bool initialize() = 0;
    virtual size_t size() const = 0;

protected:
    std::filesystem::path root_dir_;
};

} // namespace datasets

#endif // BSDS500_PROVIDER_HPP
