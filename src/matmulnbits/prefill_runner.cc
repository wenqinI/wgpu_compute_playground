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
#include "matmulnbits_common.h"
#include "prefill_runner.h"

PrefillRunner::PrefillRunner(WGPUContext* wgpu_context)
    : wgpu_context_(wgpu_context) {
  CHECK(wgpu_context);
  std::cout << __func__ << std::endl;
}

PrefillRunner::~PrefillRunner() {}

bool PrefillRunner::initialize(uint32_t m, uint32_t k, uint32_t n, bool is_verbose) {
  M_ = m;
  K_ = k;
  N_ = n;

  CHECK(M_ > 1);
  CHECK(K_ % 32 == 0);
  CHECK(N_ % 32 == 0);

  compute_runner_ = std::make_shared<BaseComputeRunner>(wgpu_context_);
  compute_runner_->initialize();

  if (!configure(is_verbose)) {
    return false;
  }

  return true;
}

bool PrefillRunner::configure(bool is_verbose) {
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

void PrefillRunner::create_buffers() {
  unsigned int kSeed = 0xDEADBEEF;  // Fixed seed for consistent results
  std::mt19937 gen(kSeed);  // Initialize the generator with the fixed seed

  size_t element_size = 0;
  size_t buffer_size = 0;
  wgpu::BufferUsage buffer_usage = wgpu::BufferUsage::None;
  wgpu::BufferBindingType buffer_binding_type =
      wgpu::BufferBindingType::BindingNotUsed;

  // input_a_: int8 block128 quantized
  std::uniform_real_distribution<> input_a_distribution(0x0, 0xFF);
  element_size = CEIL_DIVIDE(M_ * K_, 4);
  input_a_data_.resize(element_size);
  for (size_t i = 0; i < element_size; ++i) {
    uint32_t packed = 0u;
    packed |= static_cast<uint8_t>(input_a_distribution(gen)) << 0;
    packed |= static_cast<uint8_t>(input_a_distribution(gen)) << 8;
    packed |= static_cast<uint8_t>(input_a_distribution(gen)) << 16;
    packed |= static_cast<uint8_t>(input_a_distribution(gen)) << 24;

    input_a_data_[i] = packed;
  }

  buffer_size = input_a_data_.size() * sizeof(input_a_data_[0]);
  buffer_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                 wgpu::BufferUsage::CopyDst;
  buffer_binding_type = wgpu::BufferBindingType::ReadOnlyStorage;
  input_a_int8_buffer_ = compute_runner_->add_buffer(buffer_size, buffer_usage,
                                                     buffer_binding_type);
  compute_runner_->write_buffer(input_a_int8_buffer_, input_a_data_.data(),
                                buffer_size);

  // input_a_scales_: block128
  std::uniform_real_distribution<> input_a_scales_distribution(-0.05, 0.05);
  element_size = CEIL_DIVIDE(M_ * K_, 128);
  input_a_scales_data_.resize(element_size);
  for (size_t i = 0; i < element_size; ++i) {
    input_a_scales_data_[i] =
        fp16_ieee_from_fp32_value(input_a_scales_distribution(gen));
  }

  buffer_size = input_a_scales_data_.size() * sizeof(input_a_scales_data_[0]);
  buffer_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                 wgpu::BufferUsage::CopyDst;
  buffer_binding_type = wgpu::BufferBindingType::ReadOnlyStorage;
  input_a_scales_buffer_ = compute_runner_->add_buffer(
      buffer_size, buffer_usage, buffer_binding_type);
  compute_runner_->write_buffer(input_a_scales_buffer_,
                                input_a_scales_data_.data(), buffer_size);

  // input_b_: int4 block32 quantized
  std::uniform_real_distribution<> input_b_distribution(0x0, 0xF);
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

  // input_b_scales_: block32
  std::uniform_real_distribution<> input_b_scales_distribution(-0.05, 0.05);
  element_size = CEIL_DIVIDE(K_ * N_, 32);
  input_b_scales_data_.resize(element_size);
  for (size_t i = 0; i < element_size; ++i) {
    input_b_scales_data_[i] =
        fp16_ieee_from_fp32_value(input_b_scales_distribution(gen));
  }

  buffer_size = input_b_scales_data_.size() * sizeof(input_b_scales_data_[0]);
  buffer_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                 wgpu::BufferUsage::CopyDst;
  buffer_binding_type = wgpu::BufferBindingType::ReadOnlyStorage;
  input_b_scales_buffer_ = compute_runner_->add_buffer(
      buffer_size, buffer_usage, buffer_binding_type);
  compute_runner_->write_buffer(input_b_scales_buffer_,
                                input_b_scales_data_.data(), buffer_size);

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
    alignas(4) uint32_t M;
    alignas(4) uint32_t N;
    alignas(4) uint32_t K;
    alignas(4) uint32_t K8;
    alignas(4) uint32_t K16;
    alignas(4) uint32_t num_N_tile;
  };

  Uniforms uniforms_value = {};
  uniforms_value.M = M_;
  uniforms_value.N = N_;
  uniforms_value.K = K_;
  CHECK(K_ % 16 == 0);
  uniforms_value.K8 = K_ / 8;
  uniforms_value.K16 = K_ / 16;
  uniforms_value.num_N_tile = CEIL_DIVIDE(N_, tile_n_);

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

std::string PrefillRunner::generate_shader() {
  tile_m_ = 64;
  tile_n_ = 64;

  workgroup_size_.resize(3);
  workgroup_size_ = {256, 1, 1};

  std::cout << "======\n";
  std::cout << "Workgroup Size: " << workgroup_size_[0] << "x"
            << workgroup_size_[1] << "x" << workgroup_size_[2] << std::endl;

  dispatch_size_.resize(3);
  dispatch_size_[0] = CEIL_DIVIDE(M_, tile_m_) * CEIL_DIVIDE(N_, tile_n_);
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
const workgroup_size_x: u32 = 256;
const workgroup_size_y: u32 = 1;
const workgroup_size_z: u32 = 1;
@group(0) @binding(0) var<storage, read> input_a: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales_a: array<f16>;
@group(0) @binding(2) var<storage, read> input_b: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> scales_b: array<f16>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f16>>;
struct Uniforms {
  M: u32,
  N: u32,
  K: u32,
  K8: u32,
  K16: u32,
  num_N_tile: u32
};
@group(0) @binding(5) var<uniform> uniforms: Uniforms;

alias input_a_value_t = vec4<u32>;
alias input_a_indices_t = vec3<u32>;
alias output_element_t = f16;


        fn DequantizedFrom4BitsTo8Bits(in: vec2<u32>) -> vec4<u32>
        {
            var out = vec4<u32>(0);
            var value_lower = vec4<i32>(unpack4xU8(in[0] & 0x0F0F0F0Fu)) - vec4<i32>(8);
            var value_upper = vec4<i32>(unpack4xU8((in[0] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(8);
            out[0] = pack4xI8(vec4<i32>(value_lower[0], value_upper[0], value_lower[1], value_upper[1]));
            out[1] = pack4xI8(vec4<i32>(value_lower[2], value_upper[2], value_lower[3], value_upper[3]));
            value_lower = vec4<i32>(unpack4xU8(in[1] & 0x0F0F0F0Fu)) - vec4<i32>(8);
            value_upper = vec4<i32>(unpack4xU8((in[1] >> 4) & 0x0F0F0F0Fu)) - vec4<i32>(8);
            out[2] = pack4xI8(vec4<i32>(value_lower[0], value_upper[0], value_lower[1], value_upper[1]));
            out[3] = pack4xI8(vec4<i32>(value_lower[2], value_upper[2], value_lower[3], value_upper[3]));
            return out;
        }

        // Scaled dot product of 8 packed unsigned integers.
        fn SDP8AI(a1:vec4<u32>, b1:vec4<u32>, a2:vec4<u32>, b2:vec4<u32>, scale:output_element_t) -> output_element_t
        {
            var local_sum = dot4I8Packed(a1[0], b1[0]);
            local_sum += dot4I8Packed(a1[1], b1[1]);
            local_sum += dot4I8Packed(a1[2], b1[2]);
            local_sum += dot4I8Packed(a1[3], b1[3]);
            local_sum += dot4I8Packed(a2[0], b2[0]);
            local_sum += dot4I8Packed(a2[1], b2[1]);
            local_sum += dot4I8Packed(a2[2], b2[2]);
            local_sum += dot4I8Packed(a2[3], b2[3]);
            return output_element_t(local_sum) * scale;
        }
    const block_size = 32;
        const tile_size = 64;
        const subtile_size = 16;
        const tile_size_k =  32;
        const vec_factor = 4;
        const u32_factor = 4;
        const tile_size_k_vec = 2;

        // Shared memory
        var<workgroup> tile_A : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;                     // 64 x 32
        var<workgroup> scale_A : array<output_element_t, tile_size>;                                     // 64 x 1
        var<workgroup> tile_B : array<array<vec4<u32>, tile_size>, tile_size_k_vec>;                     // 64 x 32
        var<workgroup> scale_B : array<output_element_t, tile_size>;                                     // 64 x 1

        fn loadSHMA(a_global_base:u32, kidx_v:u32, row: u32, col: u32)
        {
            let a_global = a_global_base + row;
            if (a_global >= uniforms.M)
            {
                return;
            }
            tile_A[col][row] = input_a[a_global*uniforms.K16+kidx_v+col];
            if (col == 0)
            {
                // kidx_v - covers 16 values of k
                scale_A[row] = scales_a[a_global*(uniforms.K/128) + kidx_v/8];
            }
        }
    
        fn loadSHMB(b_global_base:u32, kidx_v:u32, row: u32, col: u32)
        {
            let b_global = b_global_base + row;
            if (b_global >= uniforms.N)
            {
                return;
            }

            let b_value = input_b[b_global*uniforms.K16+kidx_v+col];
            tile_B[col][row] = DequantizedFrom4BitsTo8Bits(b_value);
            if (col == 0)
            {
                // kidx_v - each kidx_v covers 16 values of k
                scale_B[row] = scales_b[b_global*(uniforms.K/block_size) + kidx_v/(block_size/16)];
            }
        }
    @compute @workgroup_size(workgroup_size_x, workgroup_size_y, workgroup_size_z)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>,
        @builtin(workgroup_id) workgroup_id : vec3<u32>,
        @builtin(local_invocation_index) local_idx : u32,
        @builtin(local_invocation_id) local_id : vec3<u32>,
        @builtin(subgroup_invocation_id) sg_id : u32,
        @builtin(subgroup_size) sg_size : u32) {
  let global_idx = global_id.x;
  let workgroup_idx = workgroup_id.x;

        // During the load phase we use all 256 threads to load 64 rows of A/B.
        // For each row we load tile_size_k_vec (2) vectorized elements, which are 32 elements of K.
        let a_global_base = u32(workgroup_idx / uniforms.num_N_tile) * tile_size;
        let b_global_base = (workgroup_idx % uniforms.num_N_tile) * tile_size;
        let load_AorB = u32(local_idx/128);
        let load_row = u32((local_idx%128)/2);
        let load_col = u32(local_idx%2);

        // During the compute phase, we have the 64x64 tile split into
        // subtiles of 16x16. We have a grid of 4x4 subtiles.
        let subtile_id = u32(local_idx / subtile_size);
        let subtile_idx = u32(subtile_id / 4);
        let subtile_idy = u32(subtile_id % 4);
        let base_A = subtile_idx * 16;
        let base_B = subtile_idy * 16;
        // For each subtile we have 16 threads assigned.
        let a_idx = u32(local_idx % subtile_size);

        var lane_output1: vec4<output_element_t>;
        var lane_output2: vec4<output_element_t>;
        var lane_output3: vec4<output_element_t>;
        var lane_output4: vec4<output_element_t>;
        // K's vectrorization is 16 items per index. See input_a/input_b.
        // tile_size_k_vec - is the k tile size in vectorized space (1/16). That is
        // k tile size is 32. In vectorized space that is 32/16 = 2.
        for (var kidx_v:u32 = 0; kidx_v < uniforms.K16; kidx_v+=tile_size_k_vec)
        {
            // Load Phase: Populate shared memory for the workgroup.
            if (load_AorB == 0)
            {
                loadSHMA(a_global_base, kidx_v, load_row, load_col);
            }
            else
            {
                loadSHMB(b_global_base, kidx_v, load_row, load_col);
            }
            workgroupBarrier();

            // Compute phase: Perform matmul for this subtile 16 x 32 x 16.
            // Step 1: Load from shared memory into registers across entire subgroup.
            var own_a0: vec4<u32> = tile_A[0][base_A + a_idx];
            var own_a1: vec4<u32> = tile_A[1][base_A + a_idx];
            var own_scale_a: output_element_t = scale_A[base_A + a_idx];
            if (sg_size == 16)
            {
                var own_b0: vec4<u32> = tile_B[0][base_B + sg_id];
                var own_b1: vec4<u32> = tile_B[1][base_B + sg_id];
                var own_scale_b: output_element_t  = scale_B[base_B + sg_id];
                // Step 2: Access registers across the subgroup using subgroupShuffle and perform the matmul.
                lane_output1[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 0), own_a1, subgroupShuffle(own_b1, 0), subgroupShuffle(own_scale_b, 0) * own_scale_a);
                lane_output1[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 1), own_a1, subgroupShuffle(own_b1, 1), subgroupShuffle(own_scale_b, 1) * own_scale_a);
                lane_output1[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 2), own_a1, subgroupShuffle(own_b1, 2), subgroupShuffle(own_scale_b, 2) * own_scale_a);
                lane_output1[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 3), own_a1, subgroupShuffle(own_b1, 3), subgroupShuffle(own_scale_b, 3) * own_scale_a);

                lane_output2[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 4), own_a1, subgroupShuffle(own_b1, 4), subgroupShuffle(own_scale_b, 4) * own_scale_a);
                lane_output2[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 5), own_a1, subgroupShuffle(own_b1, 5), subgroupShuffle(own_scale_b, 5) * own_scale_a);
                lane_output2[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 6), own_a1, subgroupShuffle(own_b1, 6), subgroupShuffle(own_scale_b, 6) * own_scale_a);
                lane_output2[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 7), own_a1, subgroupShuffle(own_b1, 7), subgroupShuffle(own_scale_b, 7) * own_scale_a);

                lane_output3[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 8), own_a1, subgroupShuffle(own_b1, 8), subgroupShuffle(own_scale_b, 8) * own_scale_a);
                lane_output3[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 9), own_a1, subgroupShuffle(own_b1, 9), subgroupShuffle(own_scale_b, 9) * own_scale_a);
                lane_output3[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 10), own_a1, subgroupShuffle(own_b1, 10), subgroupShuffle(own_scale_b, 10) * own_scale_a);
                lane_output3[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 11), own_a1, subgroupShuffle(own_b1, 11), subgroupShuffle(own_scale_b, 11) * own_scale_a);

                lane_output4[0] += SDP8AI(own_a0, subgroupShuffle(own_b0, 12), own_a1, subgroupShuffle(own_b1, 12), subgroupShuffle(own_scale_b, 12) * own_scale_a);
                lane_output4[1] += SDP8AI(own_a0, subgroupShuffle(own_b0, 13), own_a1, subgroupShuffle(own_b1, 13), subgroupShuffle(own_scale_b, 13) * own_scale_a);
                lane_output4[2] += SDP8AI(own_a0, subgroupShuffle(own_b0, 14), own_a1, subgroupShuffle(own_b1, 14), subgroupShuffle(own_scale_b, 14) * own_scale_a);
                lane_output4[3] += SDP8AI(own_a0, subgroupShuffle(own_b0, 15), own_a1, subgroupShuffle(own_b1, 15), subgroupShuffle(own_scale_b, 15) * own_scale_a);
            }
            else
            {
                // Code for other subgroup sizes, simply doesnt use subgroups at all.
                // Relies on reads from single location tile_B[][base_B + col] by all
                // being optimized by the hardware.
                lane_output1[0] += SDP8AI(own_a0, tile_B[0][base_B + 0], own_a1, tile_B[1][base_B + 0],  own_scale_a * scale_B[base_B + 0]);
                lane_output1[1] += SDP8AI(own_a0, tile_B[0][base_B + 1], own_a1, tile_B[1][base_B + 1],  own_scale_a * scale_B[base_B + 1]);
                lane_output1[2] += SDP8AI(own_a0, tile_B[0][base_B + 2], own_a1, tile_B[1][base_B + 2],  own_scale_a * scale_B[base_B + 2]);
                lane_output1[3] += SDP8AI(own_a0, tile_B[0][base_B + 3], own_a1, tile_B[1][base_B + 3],  own_scale_a * scale_B[base_B + 3]);

                lane_output2[0] += SDP8AI(own_a0, tile_B[0][base_B + 4], own_a1, tile_B[1][base_B + 4],  own_scale_a * scale_B[base_B + 4]);
                lane_output2[1] += SDP8AI(own_a0, tile_B[0][base_B + 5], own_a1, tile_B[1][base_B + 5],  own_scale_a * scale_B[base_B + 5]);
                lane_output2[2] += SDP8AI(own_a0, tile_B[0][base_B + 6], own_a1, tile_B[1][base_B + 6],  own_scale_a * scale_B[base_B + 6]);
                lane_output2[3] += SDP8AI(own_a0, tile_B[0][base_B + 7], own_a1, tile_B[1][base_B + 7],  own_scale_a * scale_B[base_B + 7]);

                lane_output3[0] += SDP8AI(own_a0, tile_B[0][base_B + 8], own_a1, tile_B[1][base_B + 8],  own_scale_a * scale_B[base_B + 8]);
                lane_output3[1] += SDP8AI(own_a0, tile_B[0][base_B + 9], own_a1, tile_B[1][base_B + 9],  own_scale_a * scale_B[base_B + 9]);
                lane_output3[2] += SDP8AI(own_a0, tile_B[0][base_B + 10], own_a1, tile_B[1][base_B + 10],  own_scale_a * scale_B[base_B + 10]);
                lane_output3[3] += SDP8AI(own_a0, tile_B[0][base_B + 11], own_a1, tile_B[1][base_B + 11],  own_scale_a * scale_B[base_B + 11]);

                lane_output4[0] += SDP8AI(own_a0, tile_B[0][base_B + 12], own_a1, tile_B[1][base_B + 12],  own_scale_a * scale_B[base_B + 12]);
                lane_output4[1] += SDP8AI(own_a0, tile_B[0][base_B + 13], own_a1, tile_B[1][base_B + 13],  own_scale_a * scale_B[base_B + 13]);
                lane_output4[2] += SDP8AI(own_a0, tile_B[0][base_B + 14], own_a1, tile_B[1][base_B + 14],  own_scale_a * scale_B[base_B + 14]);
                lane_output4[3] += SDP8AI(own_a0, tile_B[0][base_B + 15], own_a1, tile_B[1][base_B + 15],  own_scale_a * scale_B[base_B + 15]);
            }
            workgroupBarrier();
        }

        let a_global = a_global_base + base_A + a_idx;
        let b_global = b_global_base + base_B;
        let output_idx = ((a_global) * uniforms.N + b_global)/4;
        // This creates a shader requirement that uniforms.N % 16 == 0
        if (a_global < uniforms.M && b_global < uniforms.N)
        {
            output[output_idx] = lane_output1;
            output[output_idx+1] = lane_output2;
            output[output_idx+2] = lane_output3;
            output[output_idx+3] = lane_output4;
        }
    
}

)";

  return code.str();
}

void PrefillRunner::compute(uint32_t loop) {
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

void PrefillRunner::verify() {}
