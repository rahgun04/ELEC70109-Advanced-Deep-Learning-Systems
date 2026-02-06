# 1) `torch.compile()` performance reduction
In our scenario, the performance did improve after running `torch.compile` on the model for just the cpu.
```
Device: cuda, Original model: 0.0301 s
Device: cuda, Optimised model: 0.3098 s
Device: cpu, Original model: 2.7724 s
Device: cpu, Optimised model: 2.3336 s
```

# 2) SDPA Kernel 
```
Device: cuda, Original model: 0.0003 s
Device: cuda, Optimised model: 0.0001 s
Device: cpu, Original model: 0.2736 s
Device: cpu, Optimised model: 0.0196 s
```

The profiling was run by averaging 1000 runs.
The kernel fusing optimisation has yielded benefits for GPU and CPU. 

# 3) MXINT8 Optimisations
## a) 
MXINT8 is a fixed point format which requires significantly less hardware to run arithmetic on.
With both the weight and activation of the linear layer done in MXINT8, there is no mixed precision operations involved an no floating point hardware needs to be made on the custom hardware. 

## b) Dequantise kernel 
There are a series of steps to convert a quantised MXINT8 to bfloat16.
MXINT8 and bfloat16 have different interpretations of how the mantissa is layed out.

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
The handwritten CUDA kernel to run the dequantise operation splits the workload into blocks and manualy manages the Gmem to Smem transfers. Predication is used as a means to keep the kernel the exact same across threads but allow it to work on non standard sized input tensors.

### Threadblock (layout_sX) 
Whilst `layout_sX` isn't directly used to launch the grid of threadblocks, it shares dimensions with `dimBlock` which is used when launching the kernel.
```c
dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>> ...
```
# Predication
DimBlock may not perfectly divide the input task (Dimensions group_size $\times$ num_groups). So the grid is launched such that the number of blocks launched tile to be larger than the input. Predication is applied on the Gmem, Smem transfers using `copy_if(predication_map, src, dst)`.  

### cta_tiler
The cta tiler tells each thread the bounds of each thread block. Each derives a `cta_coord` based on its `blockIdx` coordinates and uses this to copy and calculate on the right information.






## d) 
The memory usage reduction isn't as much as calculated because the quantisation optimisation is only applies to `torch.nn.linear` layers. 