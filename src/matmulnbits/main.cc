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

#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>

#include <iostream>

#include <cxxopts.hpp>

#include "decode_runner.h"
#include "prefill_runner.h"
#include "wgpu_context.h"

struct Config {
  uint32_t M;
  uint32_t K;
  uint32_t N;

  uint32_t loop;

  bool is_verbose;
};

void ParseCommandLine(int argc, char** argv, Config& config) {
  try {
    std::string program_name = argv[0];
    cxxopts::Options options(program_name, "");

    options.add_option("", {"h, help", "Print help"});

    options.add_option("", {"m", "Specify M",
                            cxxopts::value<uint32_t>()->default_value("128")});
    options.add_option("", {"k", "Specify K",
                            cxxopts::value<uint32_t>()->default_value("3072")});
    options.add_option("", {"n", "Specify N",
                            cxxopts::value<uint32_t>()->default_value("8192")});

    options.add_option("", {"l, loop", "Specify loop",
                            cxxopts::value<uint32_t>()->default_value("100")});

    options.add_option("", {"v, verbose", "Verbose log for print shader code",
                            cxxopts::value<bool>()->implicit_value("true")
                            ->default_value("false")});

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    config.M = result["m"].as<uint32_t>();
    config.K = result["k"].as<uint32_t>();
    config.N = result["n"].as<uint32_t>();

    config.loop = result["loop"].as<uint32_t>();

    config.is_verbose = result["verbose"].as<bool>();
  } catch (const cxxopts::exceptions::exception& e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(-1);
  }
}

int main(int argc, char** argv) {
  Config config;
  ParseCommandLine(argc, argv, config);

  std::cout << "======" << std::endl;
  std::cout << "M: " << config.M << std::endl;
  std::cout << "K: " << config.K << std::endl;
  std::cout << "N: " << config.N << std::endl;

  std::cout << "loop: " << config.loop << std::endl;

  // Initialize Dawn's function pointers.
  dawnProcSetProcs(&dawn::native::GetProcs());

  // Create WGPUContext.
  std::unique_ptr<WGPUContext> wgpu_context = std::make_unique<WGPUContext>();
  wgpu_context->initialize();

  // Create Runner.
  std::unique_ptr<AbstractMatMulNBitsRunner> runner;
  if (config.M > 1) {
    runner = std::make_unique<PrefillRunner>(wgpu_context.get());
  } else {
    CHECK(config.M == 1);
    runner = std::make_unique<DecodeRunner>(wgpu_context.get());
  }

  bool initialzed = runner->initialize(config.M, config.K, config.N, config.is_verbose);
  if (!initialzed) {
    return -1;
  }

  // Run the compute shader.
  runner->compute(config.loop);

  return 0;
}
