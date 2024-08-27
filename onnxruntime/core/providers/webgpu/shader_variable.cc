// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <sstream>

#include "core/providers/webgpu/shader_variable.h"

namespace onnxruntime {
namespace webgpu {

ShaderVariable::ShaderVariable(const std::string& name, ProgramVariableDataType type, int rank)
    : name_(name), type_(type), rank_(rank), usage_(UseUniform) {
  Init();
}

ShaderVariable::ShaderVariable(const std::string& name, ProgramVariableDataType type, const TensorShape& dims)
    : name_(name), type_(type), rank_(static_cast<int>(dims.NumDimensions())), dims_(dims), usage_(None) {
  Init();
}

void ShaderVariable::Init() {
  ORT_ENFORCE(type_ != ProgramVariableDataType::InvalidType, "Invalid type for variable ", name_);
}

std::string ShaderVariable::GetByOffset(const std::string& offset) const {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  switch (type_) {
    case onnxruntime::webgpu::ProgramVariableDataType::InvalidType:
      ORT_THROW("Invalid type");
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Int64:
      ss << "i32(" << name_ << "[" << offset << "].x)";
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Uint64:
      ss << "u32(" << name_ << "[" << offset << "].x)";
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Vec4Bool:
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
    case onnxruntime::webgpu::ProgramVariableDataType::InvalidType:
      ORT_THROW("Invalid type");
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Int64:
      ss << name_ << "[" << offset << "]=vec2<u32>(u32(" << value << "), select(0u, 0xFFFFFFFFu, " << value << " < 0));";
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Uint64:
      ss << name_ << "[" << offset << "]=vec2<u32>(u32(" << value << "), 0u);";
      break;
    case onnxruntime::webgpu::ProgramVariableDataType::Vec4Bool:
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

}  // namespace webgpu
}  // namespace onnxruntime
