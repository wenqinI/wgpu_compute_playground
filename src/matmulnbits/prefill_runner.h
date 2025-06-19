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

#ifndef __PREFILL_RUNNER_H__
#define __PREFILL_RUNNER_H__
#include <cstdint>

#include "abstract_matmulnbits_runner.h"
#include "base_compute_runner.h"
#include "wgpu_context.h"

class PrefillRunner : public AbstractMatMulNBitsRunner {
 public:
  PrefillRunner(WGPUContext* wgpu_context);
  ~PrefillRunner() override;

  bool initialize(uint32_t m, uint32_t k, uint32_t n, bool is_verbose) override;

  void compute(uint32_t loop) override;

  void verify() override;

 private:
  bool configure(bool is_verbose);

  void create_buffers();

  std::string generate_shader();

  WGPUContext* wgpu_context_ = nullptr;

  uint32_t M_ = 0;
  uint32_t K_ = 0;
  uint32_t N_ = 0;

  std::shared_ptr<BaseComputeRunner> compute_runner_;

  // input_a_data_: int8 block128 quantized
  std::vector<uint32_t> input_a_data_;
  std::vector<Float16> input_a_scales_data_;
  // input_b_data_: int4 block32 quantized
  std::vector<uint32_t> input_b_data_;
  std::vector<Float16> input_b_scales_data_;
  std::vector<Float16> output_y_data_;
  std::vector<uint8_t> uniform_data_;

  wgpu::Buffer input_a_int8_buffer_;
  wgpu::Buffer input_a_scales_buffer_;
  wgpu::Buffer input_b_int4_buffer_;
  wgpu::Buffer input_b_scales_buffer_;
  wgpu::Buffer output_y_buffer_;
  wgpu::Buffer uniform_buffer_;

  uint32_t tile_m_ = 0;
  uint32_t tile_n_ = 0;

  std::vector<uint32_t> workgroup_size_;
  std::vector<uint32_t> dispatch_size_;
};
#endif  // __PREFILL_RUNNER_H__
