# 1) `torch.compile()`

## case: CPU

Original model: `0.9344 s`

Optimized model: `2.5768 s`

much slower! The possible reasons:

1) ResNet-18 is dominated by convolution layers, which are already highly optimized in PyTorch eager mode using oneDNN on CPU. TorchInductor primarily benefits models with heavy pointwise operations or fusion opportunities. For convolution-heavy models, the generated code often cannot outperform optimized vendor libraries.

2) My cpu is Intel i7-12700KF, which has multi cpus. Different execution paths may use different threading strategies. TorchInductor and eager mode can interact differently with OpenMP or oneDNN thread pools, sometimes leading to oversubscription or inefficient core utilization, which can degrade performance.

3) torch.compile is jit compilation. This intruduced compilation overhead for each bytecode executaion.

## case: GPU

GPU Original model: `0.0411 s`

GPU Optimized model: `0.3081 s`

even much more slower! The possible reasons:
1) During compiling trail, I got some warning as output, this may be due to some overhead happed when the compile run the model.forward in first time, for example. looking for correct liberary and the low level apis (pynvml). And also during the backend searching or loading, the compile decide to used some bad backends.

2) Maybe there are other threads are using the GPU and the GPU is too busy to run the trail.

## case: after warmup
After the two initila trail, I ran it again in notebook, the result is much better.

CPU Original model: `0.9321 s`

CPU Optimized model: `0.7850 s`

GPU Original model: `0.0214 s`

GPU Optimized model: `0.0213 s`


1) The possible case is that the model is being cache inside the device and the comiled path is also cached, no more jit graph break and recompiling happend in the runtime.

# 2) Fused Kernel

- a: rewrite the time_modle function and get_data function. And usd the same qkv for both cpu and gpu. Load to correct device before get_data call.
- b: The result is as follow:

CPU Original attention: `0.147642 s`

CPU Fused attention: `0.002589 s`

GPU Original attention: `0.000087 s`

GPU Fused attention: `0.000031 s`

The fusion dose increase the speed a lot. And compare GPU to CPU, the speed increases is limited but still double the speed. This may be the memory bendwidth on GPU is much larger than CPU and the speed up infered that the original attention is indeed limited by the memory bottelneck.
# 3) MXINT8 Optimisations
## a) 
MXINT8 is a fixed point format which requires significantly less hardware to run arithmetic on.
With both the weight and activation of the linear layer done in MXINT8, there is no mixed precision operations involved an no floating point hardware needs to be made on the custom hardware. 

## b) Dequantise kernel 
There are a series of steps to convert a quantised MXINT8 to bfloat16.
MXINT8 and bfloat16 have different interpretations of how the mantissa is laid out.

The bfloat16 mantissa section only encodes the value after the decimal point and the $2^0$ position is always treated as 1.

MXINT8 does not and cannot do this as the exponent is shared between multiple numbers and the only way to encode a discrepancy in exponent is have the mantissa denormalised.

### `dont_need_bias` case handling
The dequantisation converts form a number where normalisation isn't enforced to one where it is. 
```c
auto bias = cutlass::bfloat16_t::bitcast(sign | exp | uint16_t(0));
auto frac = static_cast<uint16_t>((mantissa_abs & 0x3F) << 1);
auto out = cutlass::bfloat16_t::bitcast(sign | exp | frac);
```
The `0x3F` mask takes the 6 LSBs of the 7 bit MXINT8 mantissa and creates a bfloat16 through a bitcast. This has implicitly assumed that the $2^0$ position is 1.

```c
auto dont_need_bias = bool(mantissa_abs & 0x40);
tXrY[i] = dont_need_bias ? out : out - bias;
```
Which is why the case is tested for and the bias is applied if the $2^0$ position was 0.

$bias = (-1)^{sign} \times 2^{exponent} \times (1 + mantissa/2^7)$

The bitrange of the bias mantissa is set to 0' but the resulting number represents 1.0 scaled by the exponent.

## c) Dequantise CUDA kernel
The handwritten CUDA kernel to run the dequantise operation splits the workload into blocks and manually manages the Gmem to Smem transfers. Predication is used as a means to keep the kernel the exact same across threads but allow it to work on non standard sized input tensors.

### Threadblock (layout_sX) 
Whilst `layout_sX` isn't directly used to launch the grid of threadblocks, it shares dimensions with `dimBlock` which is used when launching the kernel. Each thread computes one (m, k) coordinate and handles that part of the tile.  
```c
dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>> ...
```
### Predication
DimBlock may not perfectly divide the input task (Dimensions group_size $\times$ num_groups). So the grid is launched such that the number of blocks launched tile to be larger than the input. Predication is applied on the Gmem, Smem transfers using `copy_if(predication_map, src, dst)`.  

### cta_tiler
The cta tiler tells each thread the bounds of each thread block. Each derives a `cta_coord` based on its `blockIdx` coordinates and uses this to copy and calculate on the right information.

 
## d) 
The memory usage reduction isn't as much as calculated because the quantisation optimisation is only applies to `torch.nn.linear` layers. 