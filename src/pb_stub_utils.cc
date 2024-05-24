// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "pb_stub_utils.h"

#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

DLDataType
triton_to_dlpack_type(TRITONSERVER_DataType triton_dtype)
{
  DLDataType dl_dtype;
  DLDataTypeCode dl_code;

  // Number of bits required for the data type.
  size_t dt_size = 0;

  dl_dtype.lanes = 1;
  switch (triton_dtype) {
    case TRITONSERVER_TYPE_BOOL:
      dl_code = DLDataTypeCode::kDLBool;
      dt_size = 8;
      break;
    case TRITONSERVER_TYPE_UINT8:
      dl_code = DLDataTypeCode::kDLUInt;
      dt_size = 8;
      break;
    case TRITONSERVER_TYPE_UINT16:
      dl_code = DLDataTypeCode::kDLUInt;
      dt_size = 16;
      break;
    case TRITONSERVER_TYPE_UINT32:
      dl_code = DLDataTypeCode::kDLUInt;
      dt_size = 32;
      break;
    case TRITONSERVER_TYPE_UINT64:
      dl_code = DLDataTypeCode::kDLUInt;
      dt_size = 64;
      break;
    case TRITONSERVER_TYPE_INT8:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 8;
      break;
    case TRITONSERVER_TYPE_INT16:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 16;
      break;
    case TRITONSERVER_TYPE_INT32:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 32;
      break;
    case TRITONSERVER_TYPE_INT64:
      dl_code = DLDataTypeCode::kDLInt;
      dt_size = 64;
      break;
    case TRITONSERVER_TYPE_FP16:
      dl_code = DLDataTypeCode::kDLFloat;
      dt_size = 16;
      break;
    case TRITONSERVER_TYPE_FP32:
      dl_code = DLDataTypeCode::kDLFloat;
      dt_size = 32;
      break;
    case TRITONSERVER_TYPE_FP64:
      dl_code = DLDataTypeCode::kDLFloat;
      dt_size = 64;
      break;
    case TRITONSERVER_TYPE_BYTES:
      throw PythonBackendException(
          "TYPE_BYTES tensors cannot be converted to DLPack.");

    default:
      throw PythonBackendException(
          std::string("DType code \"") +
          std::to_string(static_cast<int>(triton_dtype)) +
          "\" is not supported.");
  }

  dl_dtype.code = dl_code;
  dl_dtype.bits = dt_size;
  return dl_dtype;
}

TRITONSERVER_DataType
dlpack_to_triton_type(const DLDataType& data_type)
{
  if (data_type.lanes != 1) {
    // lanes != 1 is not supported in Python backend.
    return TRITONSERVER_TYPE_INVALID;
  }

  if (data_type.code == DLDataTypeCode::kDLFloat) {
    if (data_type.bits == 16) {
      return TRITONSERVER_TYPE_FP16;
    } else if (data_type.bits == 32) {
      return TRITONSERVER_TYPE_FP32;
    } else if (data_type.bits == 64) {
      return TRITONSERVER_TYPE_FP64;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLInt) {
    if (data_type.bits == 8) {
      return TRITONSERVER_TYPE_INT8;
    } else if (data_type.bits == 16) {
      return TRITONSERVER_TYPE_INT16;
    } else if (data_type.bits == 32) {
      return TRITONSERVER_TYPE_INT32;
    } else if (data_type.bits == 64) {
      return TRITONSERVER_TYPE_INT64;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLUInt) {
    if (data_type.bits == 8) {
      return TRITONSERVER_TYPE_UINT8;
    } else if (data_type.bits == 16) {
      return TRITONSERVER_TYPE_UINT16;
    } else if (data_type.bits == 32) {
      return TRITONSERVER_TYPE_UINT32;
    } else if (data_type.bits == 64) {
      return TRITONSERVER_TYPE_UINT64;
    }
  }

  if (data_type.code == DLDataTypeCode::kDLBool) {
    if (data_type.bits == 8) {
      return TRITONSERVER_TYPE_BOOL;
    }
  }

  return TRITONSERVER_TYPE_INVALID;
}
}}}  // namespace triton::backend::python
