// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include "core/common/safeint.h"
#include "core/framework/tensor_shape.h"

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

const SafeInt<uint32_t> WORKGROUP_SIZE = 64;

enum class ShaderVariableScope {
  Input = 0,
  Output = 1,
  Local = 2,
};

enum class ShaderVariableDataType {
  invalid_type = -1,
  f32,
  vec2f32,
  vec4f32,
  f16,
  vec2f16,
  vec4f16,
  i32,
  vec2i32,
  vec4i32,
  u32,
  vec2u32,
  vec4u32,
  int64,
  uint64,
  vec4bool,
};

ShaderVariableDataType ToShaderVariableDataType(int32_t element_type, int component = 1);

class ShaderVariable {
 public:
  ShaderVariable(const std::string& name, ShaderVariableDataType type, int rank);
  ShaderVariable(const std::string& name, ShaderVariableDataType type, const TensorShape& dims);

  ShaderVariable(ShaderVariable&&) = default;
  ShaderVariable& operator=(ShaderVariable&&) = default;

  std::string GetByOffset(const std::string& offset) const;
  std::string SetByOffset(const std::string& offset, const std::string& value) const;

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ShaderVariable);

  std::string_view Name() const { return name_; }
  std::string_view StorageType() const;

  std::string name_;
  ShaderVariableDataType type_;
  int rank_;
  TensorShape dims_;
  bool use_uniform_;

  friend class ShaderHelper;
};

class ShaderHelper final {
 public:
  ShaderHelper(const Program& program, const wgpu::Device& device, const wgpu::Limits& limits, uint32_t dispatch_group_size_x, uint32_t dispatch_group_size_y, uint32_t dispatch_group_size_z);

  const ShaderVariable& AddVariable(ShaderVariableScope scope, const std::string& name, ShaderVariableDataType type, int rank = 1) {
    return AddVariableImpl(scope, name, type, rank);
  }
  const ShaderVariable& AddVariable(ShaderVariableScope scope, const std::string& name, ShaderVariableDataType type, const TensorShape& dims) {
    return AddVariableImpl(scope, name, type, dims);
  }

  template <typename... Strs>
  ShaderHelper& AppendImplementation(const Strs&... impl) {
    implementation_.push_back(MakeStringWithClassicLocale(impl...));
    return *this;
  }

  template <typename... Strs>
  ShaderHelper& MainFunctionBody(const Strs&... body) { return MainFunctionBody({WORKGROUP_SIZE, 1, 1}, body...); }

  template <typename... Strs>
  ShaderHelper& MainFunctionBody(std::tuple<uint32_t, uint32_t, uint32_t> workgroup_size, const Strs&... body) {
    ORT_ENFORCE(body_.empty(), "Main function body has already been set");

    auto [workgroup_size_x, workgroup_size_y, workgroup_size_z] = workgroup_size;

    ORT_ENFORCE(workgroup_size_x > 0 && workgroup_size_y > 0 && workgroup_size_z > 0,
                "Workgroup size must be greater than 0");
    ORT_ENFORCE(workgroup_size_x <= limits_.maxComputeWorkgroupSizeX &&
                    workgroup_size_y <= limits_.maxComputeWorkgroupSizeY &&
                    workgroup_size_z <= limits_.maxComputeWorkgroupSizeZ,
                "Workgroup size exceeds the maximum allowed size [",
                limits_.maxComputeWorkgroupSizeX, ", ",
                limits_.maxComputeWorkgroupSizeY, ", ",
                limits_.maxComputeWorkgroupSizeZ, "]");

    ORT_ENFORCE(workgroup_size_x * workgroup_size_y * workgroup_size_z <= limits_.maxComputeInvocationsPerWorkgroup,
                "Workgroup size exceeds the maximum allowed invocations ", limits_.maxComputeInvocationsPerWorkgroup);

    bool is_1d_dispatch = workgroup_size_y == 1 && workgroup_size_z == 1;

    constants_["workgroup_size_x"] = static_cast<double>(workgroup_size_x);
    constants_["workgroup_size_y"] = static_cast<double>(workgroup_size_y);
    constants_["workgroup_size_z"] = static_cast<double>(workgroup_size_z);

    std::ostringstream ss;
    ss.imbue(std::locale::classic());

    ss << "@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)\n"
          "fn main(@builtin(global_invocation_id) global_id : vec3<u32>,\n"
          "        @builtin(workgroup_id) workgroup_id : vec3<u32>,\n"
          "        @builtin(local_invocation_id) local_id : vec3<u32>";
    if (!is_1d_dispatch) {
      ss << ",\n"
            "        @builtin(local_invocation_index) local_idx : u32,\n"
            "        @builtin(num_workgroups) num_workgroups : vec3<u32>";
    }
    ss << ") {\n";
    if (is_1d_dispatch) {
      ss << "  let global_idx = global_id.x;\n"
            "  let local_idx = local_id.x;\n";
    } else {
      ss << "  let global_idx = (workgroup_id.z * num_workgroups[0] * num_workgroups[1] + workgroup_id.y * num_workgroups[0] + workgroup_id.x)\n"
            "                     * (workgroup_size_x * workgroup_size_y * workgroup_size_z) + local_idx;\n";
    }

    ss << MakeStringWithClassicLocale(body...) << "\n"
                                                  "}\n";

    body_ = ss.str();
    return *this;
  }

  std::string GuardAgainstOutOfBoundsWorkgroupSizes(const std::string& size) const {
    return "  if (global_idx >= " + size + ") { return; }\n";
  }

 private:
  template <typename T>
  const ShaderVariable& AddVariableImpl(ShaderVariableScope scope, const std::string& name, ShaderVariableDataType type, T&& arg) {
    ORT_ENFORCE((scope == ShaderVariableScope::Input || scope == ShaderVariableScope::Output) &&
                    vars_[static_cast<int>(ShaderVariableScope::Input)].size() + vars_[static_cast<int>(ShaderVariableScope::Output)].size() < limits_.maxStorageBuffersPerShaderStage,
                "Too many storage buffers in shader. Max is ", limits_.maxStorageBuffersPerShaderStage);

    if (type == ShaderVariableDataType::f16 || type == ShaderVariableDataType::vec2f16 || type == ShaderVariableDataType::vec4f16) {
      use_f16_ = true;
    }

    return vars_[static_cast<int>(scope)].emplace_back(name, type, std::forward<T>(arg));
  }

  std::string GetFinalSourceCode() const;
  friend class ProgramManager;

  const wgpu::Device& device_;
  const wgpu::Limits& limits_;
  uint32_t dispatch_group_size_x_;
  uint32_t dispatch_group_size_y_;
  uint32_t dispatch_group_size_z_;

  const Program& program_;

  std::array<std::vector<ShaderVariable>, 3> vars_;
  std::vector<std::string> implementation_;
  std::string body_;

  std::unordered_map<std::string, double> constants_;

  bool use_f16_ = false;
};

}  // namespace webgpu
}  // namespace onnxruntime
