// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class BinaryElementwiseProgramInfo final : public Program<BinaryElementwiseProgramInfo> {
 public:
  BinaryElementwiseProgramInfo(const std::string& kernel_name, const std::string& expression, const std::string& additional_impl = "")
      : Program{kernel_name}, expression_{expression}, additional_impl_{additional_impl} {
  }

  BinaryElementwiseProgramInfo& SetVec4Expression(const std::string& expression_vec4) {
    expression_vec4_ = expression_vec4;
    return *this;
  }

  virtual void CustomImplementation(ShaderHelper& /*shader*/) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32});

 private:
  std::string expression_;
  std::string expression_vec4_;
  std::string additional_impl_;
};

}  // namespace webgpu
}  // namespace onnxruntime
