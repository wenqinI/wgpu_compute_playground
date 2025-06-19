// BSD 3-Clause License
//
// Copyright (c) 2025, Jianhui Dai
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstring>
#include <random>
#include <sstream>

#include "check.h"
#include "decode_runner.h"
#include "matmulnbits_common.h"

DecodeRunner::DecodeRunner(WGPUContext* wgpu_context)
    : wgpu_context_(wgpu_context) {
  CHECK(wgpu_context);
  std::cout << __func__ << std::endl;
}

DecodeRunner::~DecodeRunner() {}

bool DecodeRunner::initialize(uint32_t m, uint32_t k, uint32_t n, bool is_verbose) {
  M_ = m;
  K_ = k;
  N_ = n;

  CHECK(M_ == 1);
  CHECK(K_ % 32 == 0);
  CHECK(N_ % 32 == 0);

  compute_runner_ = std::make_shared<BaseComputeRunner>(wgpu_context_);
  compute_runner_->initialize();

  if (!configure(is_verbose)) {
    return false;
  }

  return true;
}

bool DecodeRunner::configure(bool is_verbose) {
  std::string source = generate_shader();
  if (source.empty()) {
    return false;
  }

  compute_runner_->set_shader(source, "main");
  if(is_verbose) {
    std::cout << "======\n";
    std::cout << source << std::endl;
  }

  // After `generate_shader()` decides `tile_m_` and `tile_n_`.
  create_buffers();

  compute_runner_->set_dispatch(dispatch_size_[0], dispatch_size_[1],
                                dispatch_size_[2]);
  compute_runner_->initialize_pipeline();

  return true;
}

void DecodeRunner::create_buffers() {
  unsigned int kSeed = 0xDEADBEEF;  // Fixed seed for consistent results
  std::mt19937 gen(kSeed);  // Initialize the generator with the fixed seed

  size_t element_size = 0;
  size_t buffer_size = 0;
  wgpu::BufferUsage buffer_usage = wgpu::BufferUsage::None;
  wgpu::BufferBindingType buffer_binding_type =
      wgpu::BufferBindingType::BindingNotUsed;

  // input_a_: f16
  std::uniform_real_distribution<> input_a_distribution(-1.0, 1.0);
  element_size = CEIL_DIVIDE(M_ * K_, 1);
  input_a_data_.resize(element_size);
  for (size_t i = 0; i < element_size; ++i) {
    input_a_data_[i] = fp16_ieee_from_fp32_value(input_a_distribution(gen));
  }

  buffer_size = input_a_data_.size() * sizeof(input_a_data_[0]);
  buffer_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                 wgpu::BufferUsage::CopyDst;
  buffer_binding_type = wgpu::BufferBindingType::ReadOnlyStorage;
  input_a_fp16_buffer_ = compute_runner_->add_buffer(buffer_size, buffer_usage,
                                                     buffer_binding_type);
  compute_runner_->write_buffer(input_a_fp16_buffer_, input_a_data_.data(),
                                buffer_size);

  // input_b_: int4 block32 quantized
  std::uniform_int_distribution<> input_b_distribution(0, 0xF);
  element_size = CEIL_DIVIDE(K_ * N_, 8);
  input_b_data_.resize(element_size);
  for (size_t i = 0; i < element_size; ++i) {
    uint32_t packed = 0u;
  
    packed |= static_cast<uint8_t>(input_b_distribution(gen)) << 0;
    packed |= static_cast<uint8_t>(input_b_distribution(gen)) << 4;
    packed |= static_cast<uint8_t>(input_b_distribution(gen)) << 8;
    packed |= static_cast<uint8_t>(input_b_distribution(gen)) << 12;
    packed |= static_cast<uint8_t>(input_b_distribution(gen)) << 16;
    packed |= static_cast<uint8_t>(input_b_distribution(gen)) << 20;
    packed |= static_cast<uint8_t>(input_b_distribution(gen)) << 24;
    packed |= static_cast<uint8_t>(input_b_distribution(gen)) << 28;

    input_b_data_[i] = packed;
  }

  buffer_size = input_b_data_.size() * sizeof(input_b_data_[0]);
  buffer_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                 wgpu::BufferUsage::CopyDst;
  buffer_binding_type = wgpu::BufferBindingType::ReadOnlyStorage;
  input_b_int4_buffer_ = compute_runner_->add_buffer(buffer_size, buffer_usage,
                                                     buffer_binding_type);
  compute_runner_->write_buffer(input_b_int4_buffer_, input_b_data_.data(),
                                buffer_size);


  // scales_: block32
  std::uniform_real_distribution<> scales_distribution(-0.05, 0.05);
  element_size = CEIL_DIVIDE(N_ * K_, 32);
  scales_data_.resize(element_size);
  for (size_t i = 0; i < element_size; ++i) {
    scales_data_[i] =
        fp16_ieee_from_fp32_value(scales_distribution(gen));
  }

  buffer_size = scales_data_.size() * sizeof(scales_data_[0]);
  buffer_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                 wgpu::BufferUsage::CopyDst;
  buffer_binding_type = wgpu::BufferBindingType::ReadOnlyStorage;
  scales_buffer_ = compute_runner_->add_buffer(
      buffer_size, buffer_usage, buffer_binding_type);
  compute_runner_->write_buffer(scales_buffer_,
                                scales_data_.data(), buffer_size);

  // output_y_
  element_size = M_ * N_;
  output_y_data_.resize(element_size);

  buffer_size = output_y_data_.size() * sizeof(output_y_data_[0]);
  buffer_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                 wgpu::BufferUsage::CopyDst;
  buffer_binding_type = wgpu::BufferBindingType::Storage;
  output_y_buffer_ = compute_runner_->add_buffer(buffer_size, buffer_usage,
                                                 buffer_binding_type);

  // uniform_
  struct Uniforms {
    alignas(16) uint32_t input_a_shape[3];
    alignas(8) uint32_t input_a_stride[2];
    alignas(16) uint32_t input_b_shape[3];
    alignas(8) uint32_t input_b_stride[2];
    alignas(16) uint32_t output_shape[3];
    alignas(8) uint32_t output_stride[2];
    alignas(4) uint32_t block_size;
  };

  Uniforms uniforms_value = {};
  uniforms_value.input_a_shape[0] = 1;
  uniforms_value.input_a_shape[1] = M_;
  uniforms_value.input_a_shape[2] = K_ / 4;

  uniforms_value.input_a_stride[0] = K_ / 4;
  uniforms_value.input_a_stride[1] = K_ / 4;

  uniforms_value.input_b_shape[0] = N_;
  uniforms_value.input_b_shape[1] = K_ / 32;
  uniforms_value.input_b_shape[2] = 1;

  uniforms_value.input_b_stride[0] = K_ / 32;
  uniforms_value.input_b_stride[1] = K_ / 32;

  uniforms_value.output_shape[0] = 1;
  uniforms_value.output_shape[1] = M_;
  uniforms_value.output_shape[2] = N_;

  uniforms_value.output_stride[0] = N_;
  uniforms_value.output_stride[1] = N_;

  uniforms_value.block_size = 32;

  uniform_data_.resize(sizeof(uniforms_value));
  memcpy(uniform_data_.data(), &uniforms_value, sizeof(uniforms_value));

  buffer_size = uniform_data_.size() * sizeof(uniform_data_[0]);
  buffer_usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopySrc |
                 wgpu::BufferUsage::CopyDst;
  buffer_binding_type = wgpu::BufferBindingType::Uniform;
  uniform_buffer_ = compute_runner_->add_buffer(buffer_size, buffer_usage,
                                                buffer_binding_type);
  compute_runner_->write_buffer(uniform_buffer_, uniform_data_.data(),
                                buffer_size);
}

std::string DecodeRunner::generate_shader() {
  tile_ = 8;

  workgroup_size_.resize(3);
  workgroup_size_ = {16, 8, 1};

  std::cout << "======\n";
  std::cout << "Workgroup Size: " << workgroup_size_[0] << "x"
            << workgroup_size_[1] << "x" << workgroup_size_[2] << std::endl;

  dispatch_size_.resize(3);
  dispatch_size_[0] = CEIL_DIVIDE(N_, tile_);
  dispatch_size_[1] = 1;
  dispatch_size_[2] = 1;

  std::cout << "Dispatch Size: " << dispatch_size_[0] << "x"
            << dispatch_size_[1] << "x" << dispatch_size_[2] << std::endl;

  std::stringstream code;

  // Reset the stringstream:
  code.str("");  // Clear the string buffer
  code.clear();  // Clear error flags

  code << R"(
enable f16;
enable subgroups;
const workgroup_size_x: u32 = 16;
const workgroup_size_y: u32 = 8;
const workgroup_size_z: u32 = 1;
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> input_b: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> scales: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;
struct Uniforms {
  input_a_shape: vec3<u32>,
  input_a_stride: vec2<u32>,
  input_b_shape: vec3<u32>,
  input_b_stride: vec2<u32>,
  output_shape: vec3<u32>,
  output_stride: vec2<u32>,
  block_size: u32
};
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

alias input_a_value_t = vec4<f16>;
alias input_a_indices_t = vec3<u32>;
fn i2o_input_a(indices : input_a_indices_t)->u32 {
  return indices[0] * uniforms.input_a_stride[0] + indices[1] * uniforms.input_a_stride[1] + indices[2];
}
fn get_input_a_by_indices(indices: input_a_indices_t)->input_a_value_t {
  return input_a[i2o_input_a(indices)];
}
alias input_b_value_t = vec4<u32>;
alias input_b_indices_t = vec3<u32>;
fn i2o_input_b(indices : input_b_indices_t)->u32 {
  return indices[0] * uniforms.input_b_stride[0] + indices[1] * uniforms.input_b_stride[1] + indices[2];
}
fn get_input_b_by_indices(indices: input_b_indices_t)->input_b_value_t {
  return input_b[i2o_input_b(indices)];
}
alias output_value_t = f16;
alias output_indices_t = vec3<u32>;
alias output_element_t = f16;
fn o2i_output(offset : u32)->output_indices_t {
  var indices: output_indices_t;
  var current = offset;
  indices[0] = current / uniforms.output_stride[0];
  current = current % uniforms.output_stride[0];
  indices[1] = current / uniforms.output_stride[1];
  current = current % uniforms.output_stride[1];
  indices[2] = current;
  return indices;
}
fn i2o_output(indices : output_indices_t)->u32 {
  return indices[0] * uniforms.output_stride[0] + indices[1] * uniforms.output_stride[1] + indices[2];
}
fn set_output_by_indices(indices: output_indices_t, value: output_value_t) {
  output[i2o_output(indices)]=value;
}

fn mm_readA(batch : u32, row : u32, col : u32) -> input_a_value_t {
  if (col < uniforms.input_a_shape[2]) {
    return get_input_a_by_indices(input_a_indices_t(batch, row, col));
  } else {
    return input_a_value_t(0);
  }
}
var<workgroup> sub_a: array<input_a_value_t, 128>;
var<workgroup> inter_results: array<array<output_value_t, 16>, 8>;
@compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_idx : u32,
        @builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_id : u32,
        @builtin(subgroup_size) sg_size : u32) {
  let global_idx = global_id.x;
  let workgroup_idx = workgroup_id.x;
  let output_indices = o2i_output(workgroup_idx * 8);
  let col = output_indices[2]; // workgroup_idx * 8
  let row = output_indices[1]; // always 0
  let batch = output_indices[0]; // always 0
  let n_blocks_per_col = uniforms.input_b_shape[1]; // 96
  let num_tiles =  (n_blocks_per_col - 1) / 16 + 1; // 6
  for (var tile: u32 = 0; tile < num_tiles; tile += 1) {
    let a_col_start = tile * 128;
    // load one tile A data into shared memory.
    for (var a_offset = local_idx; a_offset < 128; a_offset += 128) {
      let a_col = a_col_start + a_offset;
      sub_a[a_offset] = mm_readA(batch, row, a_col);
    }
    workgroupBarrier();
    let b_row = col + local_id.y;
    let block = tile * 16 + local_id.x;
    let zero_point = output_element_t(8.0);
    var scale = output_element_t(0);
    var b_data = input_b_value_t(0);
    if (block < n_blocks_per_col) {
      scale = scales[b_row * n_blocks_per_col + block];
      b_data = get_input_b_by_indices(input_b_indices_t(b_row, block, 0));
    }
    var word_offset = local_id.x * 8;
    for (var i: u32 = 0; i < 4; i++) {
      let b_value = b_data[i];
      let b_value_lower = unpack4xU8(b_value & 0x0F0F0F0Fu);
      let b_value_upper = unpack4xU8((b_value >> 4) & 0x0F0F0F0Fu);
      let b_quantized_values = mat2x4<output_element_t>(output_element_t(b_value_lower[0]), output_element_t(b_value_upper[0]), output_element_t(b_value_lower[1]), output_element_t(b_value_upper[1]), output_element_t(b_value_lower[2]), output_element_t(b_value_upper[2]), output_element_t(b_value_lower[3]), output_element_t(b_value_upper[3]));
      let b_dequantized_values = (b_quantized_values - mat2x4<output_element_t>(zero_point, zero_point, zero_point, zero_point, zero_point, zero_point, zero_point, zero_point)) * scale;
      inter_results[local_id.y][local_id.x] += dot(sub_a[word_offset], b_dequantized_values[0]) + dot(sub_a[word_offset + 1], b_dequantized_values[1]);
      word_offset += 2;
    }
    workgroupBarrier();
  }
  if (local_idx < 8) {
    var output_value = output_value_t(0);
    for (var b = 0u; b < 16; b++) {
      output_value += inter_results[local_idx][b];
    }
    if (col + local_idx < uniforms.output_shape[2]) {
      set_output_by_indices(output_indices_t(batch, row, col + local_idx), output_value);;
    }
  }

}

)";

  return code.str();
}

void DecodeRunner::compute(uint32_t loop) {
  std::cout << "======\n";
  std::cout << "GPU Compute" << std::endl;

  std::vector<double> latency_list;
  std::vector<uint8_t> readback_output;
  for (uint32_t i = 0; i < loop; ++i) {
    std::pair<double, double> latency = compute_runner_->run();
    double gpu_latency = latency.second;
    latency_list.push_back(gpu_latency);

    readback_output = compute_runner_->read_buffer(output_y_buffer_);

    std::cout << "\rProgress: " << std::setw(2) << 100.0f * i / loop << "%"
              << std::flush;
  }
  std::cout << std::endl;

  std::cout << "======\n";
  std::cout << "output:" << std::endl;
  compute_runner_->log_vector<Float16>(readback_output.data(),
                                       readback_output.size(), 128);

  std::cout << "======" << std::endl;
  log_result(latency_list);
}

void DecodeRunner::verify() {}
