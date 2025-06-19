# matmulnbits
 
This implements the **ONNX Runtime `MatMulNBits`** operation using WebGPU. The WebGPU shaders are directly sourced from the ONNX Runtime project (version `v1.22.0`).
 
## Usage
 
```shell
matmulnbits.exe -h
 
usage:
  matmulnbits.exe [OPTION...]
 
  -h, --help        Print help
  -m arg            Specify M (default: 128)
  -k arg            Specify K (default: 3072)
  -n arg            Specify N (default: 8192)
  -l, --loop arg    Specify loop (default: 100)
  -v, --verbose   Verbose log for print shader code
```

## Kernels and data type
This benchmark was used for benchmarking the performance for **MatMulNBits** opeator, it do a matrix multiply like:

$$
A_{m \times k} @ B_{k \times n} = C_{m \times n}
$$

There are two kinds of kernel in this benchmark:
1. **Prefill**: when we're running benchmark with `m > 1`, we will run the perfill kernel, its data type for $A$ is `int8`, for $B$ is `int4`, for C is `float16`.
1. **Decode**: when we're running benchmark with `m = 1`, we will run the deocde kernel, its data type for $A$ is `float16`, for $B$ is `int4`, for C is `float16`.




## Links
 
* [**MatMulNBits**](https://github.com/microsoft/onnxruntime/blob/v1.22.0/docs/ContribOperators.md#com.microsoft.MatMulNBits)
* [**ONNX Runtime**](https://github.com/microsoft/onnxruntime/tree/v1.22.0)