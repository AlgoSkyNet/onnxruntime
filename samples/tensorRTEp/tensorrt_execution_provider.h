#pragma once
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include <string>

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

namespace onnxruntime {

struct TensorrtExecutionProvider : public OrtExecutionProvider {
    TensorrtExecutionProvider(const char* ep_type, const ProviderOptions& provider_options);
    void CreateNodeComputeInfoFromPrecompiledEngine(const OrtGraphViewer* graph_body_viewer, const OrtNode* fused_node,
                                                    std::unordered_map<std::string, size_t>& input_map,
                                                    std::unordered_map<std::string, size_t>& output_map,
                                                    OrtNodeComputeInfo** node_compute_funcs);
private:
    bool external_stream_ = false;
};

struct TensorrtExecutionProviderFactory : public OrtExecutionProviderFactory {
    TensorrtExecutionProviderFactory();
};
}

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_API OrtExecutionProviderFactory* RegisterCustomEp();

#ifdef __cplusplus
}
#endif
