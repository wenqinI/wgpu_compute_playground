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
 
## Links
 
* [**MatMulNBits**](https://github.com/microsoft/onnxruntime/blob/v1.22.0/docs/ContribOperators.md#com.microsoft.MatMulNBits)
* [**ONNX Runtime**](https://github.com/microsoft/onnxruntime/tree/v1.22.0)