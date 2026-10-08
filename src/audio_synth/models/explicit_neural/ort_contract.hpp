#pragma once

#include <piper_onnx/ort_session_options.hpp>

#include <algorithm>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>

namespace vocal::detail {
inline Ort::SessionOptions explicit_session_options(int threads, const std::string& affinities) {
    if (threads <= 0) throw std::invalid_argument("ONNX thread count must be positive");
    auto options = cpu_session_options(threads);
    if (!affinities.empty()) options.AddConfigEntry("session.intra_op_thread_affinities", affinities.c_str());
    return options;
}

// Also checks static dimensions at each call so fixed-size exports fail clearly.
inline void require_input(const Ort::Session& session, const std::string& name,
                          ONNXTensorElementDataType type, std::span<const std::int64_t> shape) {
    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
        const auto input_name = session.GetInputNameAllocated(i, allocator);
        if (name != input_name.get()) continue;
        const auto info = session.GetInputTypeInfo(i);
        if (info.GetONNXType() != ONNX_TYPE_TENSOR)
            throw std::runtime_error("input " + name + " must be a tensor");
        const auto tensor = info.GetTensorTypeAndShapeInfo();
        const auto actual = tensor.GetShape();
        if (tensor.GetElementType() != type || actual.size() != shape.size())
            throw std::runtime_error("input " + name + " has incompatible type or rank");
        for (std::size_t dim = 0; dim < actual.size(); ++dim)
            if (actual[dim] >= 0 && shape[dim] >= 0 && actual[dim] != shape[dim])
                throw std::runtime_error("input " + name + " has incompatible dimensions");
        return;
    }
    throw std::runtime_error("missing required ONNX input: " + name);
}

inline void require_output(const Ort::Session& session, const std::string& name) {
    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
        const auto output_name = session.GetOutputNameAllocated(i, allocator);
        if (name == output_name.get()) return;
    }
    throw std::runtime_error("missing required ONNX output: " + name);
}

inline std::vector<std::int64_t> float_output_shape(const Ort::Value& value) {
    if (!value.IsTensor()) throw std::runtime_error("ONNX output must be a float32 tensor");
    const auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("ONNX output must be a float32 tensor");
    return info.GetShape();
}
} // namespace vocal::detail
