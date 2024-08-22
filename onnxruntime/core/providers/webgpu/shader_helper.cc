// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "core/session/onnxruntime_c_api.h"

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

ShaderVariableDataType ToShaderVariableDataType(int32_t element_type, int component /* = 1 */) {
  if (component == 1) {
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ShaderVariableDataType::f32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ShaderVariableDataType::f16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ShaderVariableDataType::i32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ShaderVariableDataType::u32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return ShaderVariableDataType::int64;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return ShaderVariableDataType::uint64;
      default:
        return ShaderVariableDataType::invalid_type;
    }
  } else if (component == 2) {
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ShaderVariableDataType::vec2f32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ShaderVariableDataType::vec2f16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ShaderVariableDataType::vec2i32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ShaderVariableDataType::vec2u32;
      default:
        return ShaderVariableDataType::invalid_type;
    }
  } else if (component == 4) {
    switch (element_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ShaderVariableDataType::vec4f32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return ShaderVariableDataType::vec4f16;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return ShaderVariableDataType::vec4i32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return ShaderVariableDataType::vec4u32;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return ShaderVariableDataType::vec4bool;
      default:
        return ShaderVariableDataType::invalid_type;
    }
  } else {
    return ShaderVariableDataType::invalid_type;
  }
}

ShaderVariable::ShaderVariable(const std::string& name, ShaderVariableDataType type, int rank) : name_(name), type_(type), rank_(rank), use_uniform_(true) {
  ORT_ENFORCE(type != ShaderVariableDataType::invalid_type, "Invalid type for variable ", name_);
}

ShaderVariable::ShaderVariable(const std::string& name, ShaderVariableDataType type, const TensorShape& dims) : name_(name), type_(type), rank_(0), dims_(dims), use_uniform_(false) {
  ORT_ENFORCE(type != ShaderVariableDataType::invalid_type, "Invalid type for variable ", name_);
}

std::string ShaderVariable::GetByOffset(const std::string& offset) const {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  switch (type_) {
    case onnxruntime::webgpu::ShaderVariableDataType::invalid_type:
      ORT_THROW("Invalid type");
      break;
    case onnxruntime::webgpu::ShaderVariableDataType::int64:
      ss << "i32(" << name_ << "[" << offset << "].x)";
      break;
    case onnxruntime::webgpu::ShaderVariableDataType::uint64:
      ss << "u32(" << name_ << "[" << offset << "].x)";
      break;
    case onnxruntime::webgpu::ShaderVariableDataType::vec4bool:
      ss << "vec4<bool>(bool("
         << name_ << "[" << offset << "] & 0xFFu), bool("
         << name_ << "[" << offset << "] & 0xFF00u), bool("
         << name_ << "[" << offset << "] & 0xFF0000u), bool("
         << name_ << "[" << offset << "] & 0xFF000000u))";
      break;
    default:
      ss << name_ << "[" << offset << "]";
  }

  return ss.str();
}

std::string ShaderVariable::SetByOffset(const std::string& offset, const std::string& value) const {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  switch (type_) {
    case onnxruntime::webgpu::ShaderVariableDataType::invalid_type:
      ORT_THROW("Invalid type");
      break;
    case onnxruntime::webgpu::ShaderVariableDataType::int64:
      ss << name_ << "[" << offset << "]=vec2<u32>(u32(" << value << "), select(0u, 0xFFFFFFFFu, " << value << " < 0));";
      break;
    case onnxruntime::webgpu::ShaderVariableDataType::uint64:
      ss << name_ << "[" << offset << "]=vec2<u32>(u32(" << value << "), 0u);";
      break;
    case onnxruntime::webgpu::ShaderVariableDataType::vec4bool:
      ss << name_ << "[" << offset << "]=dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(" << value << "));";
      break;
    default:
      ss << name_ << "[" << offset << "]=" << value << ";";
  }

  return ss.str();
}

std::string_view ShaderVariable::StorageType() const {
  constexpr std::string_view STORAGE_TYPE[] = {
      "f32",        // f32
      "vec2<f32>",  // vec2f32
      "vec4<f32>",  // vec4f32
      "f16",        // f16
      "vec2<f16>",  // vec2f16
      "vec4<f16>",  // vec4f16
      "i32",        // i32
      "vec2<i32>",  // vec2i32
      "vec4<i32>",  // vec4i32
      "u32",        // u32
      "vec2<u32>",  // vec2u32
      "vec4<u32>",  // vec4u32
      "vec2<u32>",  // int64
      "vec2<u32>",  // uint64
      "u32",        // vec4bool
  };

  return STORAGE_TYPE[static_cast<int>(type_)];
}

ShaderHelper::ShaderHelper(const Program& program, const wgpu::Device& device, const wgpu::Limits& limits, uint32_t dispatch_group_size_x, uint32_t dispatch_group_size_y, uint32_t dispatch_group_size_z)
    : device_{device},
      limits_{limits},
      dispatch_group_size_x_{dispatch_group_size_x},
      dispatch_group_size_y_{dispatch_group_size_y},
      dispatch_group_size_z_{dispatch_group_size_z},
      program_{program},
      use_f16_{false} {
  ORT_ENFORCE(dispatch_group_size_x_ > 0 && dispatch_group_size_y_ > 0 && dispatch_group_size_z_ > 0, "Invalid dispatch group size");
}

std::string ShaderHelper::GetFinalSourceCode() const {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  //
  // Section feature enabling
  //
  if (use_f16_) {
    ORT_ENFORCE(device_.HasFeature(wgpu::FeatureName::ShaderF16), "Program ", program_.Name(), " requires f16 but the device does not support it.");
    ss << "enable f16;\n\n";
  }

  //
  // Section constants
  //
  ss << "const WORKGROUP_SIZE: u32 = " << static_cast<uint32_t>(WORKGROUP_SIZE) << ";\n"
     << "override workgroup_size_x: u32 = WORKGROUP_SIZE;\n"
        "override workgroup_size_y: u32 = 1;\n"
        "override workgroup_size_z: u32 = 1;\n\n";

  //
  // Input/output variables
  //
  int variable_count = 0;
  for (const auto& input : vars_[static_cast<int>(ShaderVariableScope::Input)]) {
    ss << "@group(0) @binding(" << variable_count++ << ") var<storage, read> " << input.Name() << ": array<" << input.StorageType() << ">;\n";
  }
  for (const auto& output : vars_[static_cast<int>(ShaderVariableScope::Output)]) {
    ss << "@group(0) @binding(" << variable_count++ << ") var<storage, read_write> " << output.Name() << ": array<" << output.StorageType() << ">;\n";
  }

  //
  // uniform variables
  //
  const auto& uniforms = program_.UniformVariables();
  if (!uniforms.empty()) {
    bool first = true;
    ss << "struct Uniforms {\n";
    for (const auto& uniform : uniforms) {
      const auto& name = uniform.name;
      const auto& data_type = uniform.data_type;
      const auto& type = ProgramUniformVariableDataTypeName[static_cast<int>(data_type)];
      const auto size = uniform.num_elements;

      if (first) {
        first = false;
      } else {
        ss << ",\n";
      }

      auto alignment = (data_type == ProgramUniformVariableDataType::f16 && size > 4) ? "@align(16) " : "";
      ss << "  " << alignment << name << ": ";
      if (size > 4) {
        if (data_type == ProgramUniformVariableDataType::f16) {
          size_t array_size = (size + 7) / 8;
          ss << "array<mat2x4<" << type << ">, " << array_size << ">";
        } else {
          size_t array_size = (size + 3) / 4;
          ss << "array<vec4<" << type << ">, " << array_size << ">";
        }
      } else if (size > 1) {
        ss << "vec" << size << "<" << type << ">";
      } else {
        ss << type;
      }
    }

    ss << "};\n"
          "@group(0) @binding("
       << variable_count << ") var<uniform> uniforms: Uniforms;\n";
  }

  //
  // Indices helper
  //
  ss << "\n"
        "// TODO: add indices helper functions\n"
        "\n";

  //
  // Additional Implementation
  //
  for (const auto& impl : implementation_) {
    ss << impl << "\n";
  }

  //
  // Main Function Body
  //
  ss << body_;

  return ss.str();
}

}  // namespace webgpu
}  // namespace onnxruntime
