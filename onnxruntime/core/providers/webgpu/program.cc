// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

Program::Program(const std::string& name)
    : name_{name},
      workgroup_dispatch_size_x_{0},
      workgroup_dispatch_size_y_{0},
      workgroup_dispatch_size_z_{0} {
}

Program& Program::Inputs(std::initializer_list<ProgramInput> inputs) {
  inputs_.assign(inputs.begin(), inputs.end());
  return *this;
}

Program& Program::Outputs(std::initializer_list<Tensor*> outputs) {
  outputs_.assign(outputs.begin(), outputs.end());
  return *this;
}

Program& Program::WorkgroupDispatchSize(uint32_t x) {
  return WorkgroupDispatchSize(x, 1, 1);
}

Program& Program::WorkgroupDispatchSize(uint32_t x, uint32_t y) {
  return WorkgroupDispatchSize(x, y, 1);
}

Program& Program::WorkgroupDispatchSize(uint32_t x, uint32_t y, uint32_t z) {
  workgroup_dispatch_size_x_ = x;
  workgroup_dispatch_size_y_ = y;
  workgroup_dispatch_size_z_ = z;
  return *this;
}

Program& Program::UniformVariables(std::initializer_list<ProgramUniformVariable> variables) {
  variables_.insert(variables_.end(), variables.begin(), variables.end());
  return *this;
}

}  // namespace webgpu
}  // namespace onnxruntime