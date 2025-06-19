# WebGPU MatMulNBits Workload

![WebGPU Supported](https://img.shields.io/badge/WebGPU-Native%20C++-blue)
![C++](https://img.shields.io/badge/C++-blueviolet?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-brightgreen?logo=cmake&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

This implements the **ONNX Runtime `MatMulNBits`** operation using WebGPU. The WebGPU shaders are directly sourced from the ONNX Runtime project (version `v1.22.0`).

## Prerequisites
Before using or building this project, ensure you have the following prerequisites installed on your system:

**Hareware:**
Basic: ADL(12th) CPU device.
Recommend: LNL(15th) CPU device.

**Software:**
CMake 3.16 or higher, Python3.x, Visual Studio.

## Getting Started

1. **Clone the Repository:**

    ```shell
    git clone https://github.com/daijh/wgpu_compute_playground.git
    cd wgpu_compute_playground
    git checkout -b matmulnbits-dev remotes/origin/matmulnbits-dev
    ```

2. **Initialize/Update Submodules:**

    ```shell
    git submodule update --init
    ```

3. **Build the Project:**

    ```shell
    cmake -S . -B build
    cmake --build build -j8
    ```

4. **Run Tests:**

    ```shell
    build\wgpu\Debug\matmulnbits.exe > result.txt
    python3 diff.py result-ref.txt result.txt output Elements
    ```

5. **Run Benchmarking:**

    ```shell
    for %%x in (128 1024 2048 4096) do (
        build\wgpu\Debug\matmulnbits.exe -m %%x
    )

    ```

## Contributing

Contributions are welcome! If you have any ideas for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is licensed under the BSD 3-Clause "New" or "Revised" License. See the `LICENSE` file for more information.
