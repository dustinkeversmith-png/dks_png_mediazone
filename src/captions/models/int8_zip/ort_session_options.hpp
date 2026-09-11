#pragma once
#include <onnxruntime_cxx_api.h>

namespace captions::detail {
inline Ort::SessionOptions cpu_session_options(int threads) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetInterOpNumThreads(1);
    if (threads > 0) options.SetIntraOpNumThreads(threads);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    options.EnableCpuMemArena();
    options.EnableMemPattern();
    return options;
}
} // namespace captions::detail
