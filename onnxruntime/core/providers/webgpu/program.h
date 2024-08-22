// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/string_join.h"

namespace onnxruntime {
class Tensor;

namespace webgpu {
class ShaderHelper;

enum class ProgramUniformVariableDataType {
  f32,
  f16,
  u32,
  i32,
};

constexpr size_t ProgramUniformVariableDataTypeSize[] = {
    sizeof(float),
    sizeof(uint16_t),
    sizeof(uint32_t),
    sizeof(int32_t),
};

constexpr std::string_view ProgramUniformVariableDataTypeName[] = {
    "f32",
    "f16",
    "u32",
    "i32",
};

struct ProgramUniformVariable {
  ProgramUniformVariable(const std::string& name, ProgramUniformVariableDataType data_type, const void* data_ptr, size_t count = 1)
      : name(name), data_type(data_type), num_elements(count) {
    ORT_ENFORCE(count > 0, "count must be greater than 0");

    size_t element_size = ProgramUniformVariableDataTypeSize[static_cast<int>(data_type)];

    data.resize(count * element_size);
    memcpy(data.data(), data_ptr, count * element_size);
  }
  std::string name;
  ProgramUniformVariableDataType data_type;
  size_t num_elements;
  std::vector<uint8_t> data;
};

enum class ProgramInputTensorDependency : int {
  None = 0,
  Type = 1,
  Rank = 2,
  Shape = 4,
  TypeAndRank = Type | Rank,
  TypeAndShape = Type | Shape,
};

inline ProgramInputTensorDependency operator|(ProgramInputTensorDependency a, ProgramInputTensorDependency b) {
  return (ProgramInputTensorDependency)((int&)a | (int&)b);
}
inline ProgramInputTensorDependency operator&(ProgramInputTensorDependency a, ProgramInputTensorDependency b) {
  return (ProgramInputTensorDependency)((int&)a & (int&)b);
}
inline ProgramInputTensorDependency& operator|=(ProgramInputTensorDependency& a, ProgramInputTensorDependency b) {
  return (ProgramInputTensorDependency&)((int&)a |= (int&)b);
}
inline ProgramInputTensorDependency& operator&=(ProgramInputTensorDependency& a, ProgramInputTensorDependency b) {
  return (ProgramInputTensorDependency&)((int&)a &= (int&)b);
}

struct ProgramInput {
  const Tensor* tensor;
  ProgramInputTensorDependency dependency;
};

class Program {
 public:
  Program(const std::string& name);
  virtual ~Program() = default;

  //
  // chain-style methods for setting properties
  //

  // set the cache hint for the program
  template <typename... CacheHintArgs>
  Program& CacheHint(CacheHintArgs&&... args) {
    cache_hint_ = StringJoin("|", std::forward<CacheHintArgs>(args)...);
  }

  Program& Inputs(std::initializer_list<ProgramInput> inputs);
  Program& Outputs(std::initializer_list<Tensor*> outputs);

  Program& WorkgroupDispatchSize(uint32_t x);
  Program& WorkgroupDispatchSize(uint32_t x, uint32_t y);
  Program& WorkgroupDispatchSize(uint32_t x, uint32_t y, uint32_t z);

  Program& UniformVariables(std::initializer_list<ProgramUniformVariable> variables);

  //
  // shader code generation
  //

  virtual Status GenerateShaderCode(ShaderHelper& sh) const = 0;

  //
  // Properties Getters
  //

  const std::string& Name() const { return name_; }
  const std::string& CacheHint() const { return cache_hint_; }
  const std::vector<ProgramInput>& Inputs() const { return inputs_; }
  const std::vector<Tensor*>& Outputs() const { return outputs_; }
  std::tuple<uint32_t, uint32_t, uint32_t> WorkgroupDispatchSize() const {
    return std::make_tuple(workgroup_dispatch_size_x_, workgroup_dispatch_size_y_, workgroup_dispatch_size_z_);
  }
  const std::vector<ProgramUniformVariable>& UniformVariables() const { return variables_; }

 private:
  std::string name_;
  std::string cache_hint_;
  std::vector<ProgramInput> inputs_;
  std::vector<Tensor*> outputs_;

  uint32_t workgroup_dispatch_size_x_;
  uint32_t workgroup_dispatch_size_y_;
  uint32_t workgroup_dispatch_size_z_;

  std::vector<ProgramUniformVariable> variables_;
};

}  // namespace webgpu
}  // namespace onnxruntime
