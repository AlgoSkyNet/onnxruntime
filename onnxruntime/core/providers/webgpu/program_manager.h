// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <string>
#include <unordered_map>

#include <webgpu/webgpu_cpp.h>

#include "core/common/common.h"

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
class Tensor;

namespace webgpu {

struct ProgramUniformInfo {
  ProgramUniformVariableDataType data_type;
  size_t offset;
  size_t length;
};

class ProgramArtifact {
 public:
  ProgramArtifact(const Program& program, wgpu::ComputePipeline compute_pipeline);

  std::string name;
  wgpu::ComputePipeline compute_pipeline;
  std::vector<ProgramUniformInfo> uniforms;
  size_t uniform_total_size;

  ProgramArtifact(ProgramArtifact&&) = default;
  ProgramArtifact& operator=(ProgramArtifact&&) = default;

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ProgramArtifact);
};

class ProgramManager {
  using DispatchGroupSize = std::tuple<uint32_t, uint32_t, uint32_t>;

 public:
  ProgramManager(const wgpu::Device& device, const wgpu::Limits& limits) : device_(device), limits_(limits) {}

  DispatchGroupSize NormalizeDispatchGroupSize(DispatchGroupSize dispatch) const;

  Status Build(const Program& program, DispatchGroupSize normalized_dispatch, wgpu::ComputePipeline& compute_pipeline) const;
  const ProgramArtifact* Get(const std::string& key) const;
  const ProgramArtifact* Set(const std::string& key, ProgramArtifact&& program);

 private:
  std::unordered_map<std::string, ProgramArtifact> programs_;
  const wgpu::Device& device_;
  const wgpu::Limits& limits_;
};

}  // namespace webgpu
}  // namespace onnxruntime
