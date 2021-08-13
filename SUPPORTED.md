# Arm® Ethos™-N Driver Stack Operator Support


## General Tensor Support Information
- 8 bit signed quantized and 8 bit unsigned quantized datatypes are supported
- Tensors with up to 4 dimensions are supported
- A "Batch" dimension > 1 is not supported
- NHWC tensors are supported
- Per-channel quantization is supported only for the weights and bias tensors of a limited set of operations.
- Most operations are not supported when the input or output tensor depths are too large. For more details see IsTensorDepthSupported in SupportQueries.cpp in the Support Library


## Addition
- Element by element addition is supported
- Addition of a variable with a scalar constant is supported in cases where the quantized values in the output are the same as the input, this is replaced with a Reinterpret Quantization operation. **This is available only when using Arm NN**


## Average Pooling
- Pooling size 3x3, with stride 1,1 and Same padding is supported
- A "Mean" Average pooling is supported where the pooling size is 7 or 8 and the input width and height is equal to the pool size


## Concatentation
- Output quantization scale smaller than input quantization scale / 128 is not supported
- If concatentating the 3rd dimension, the input tensor's 3rd dimension must be a multiple of 16


## Constant
- No specific restrictions


## Convolution 2D
- HWIO format weights are supported
- Kernel heights and widths supported (the kernel does not have to be square): { 1, 2, 3, 5, 7, 9 }
- Stride supported (the height and width stride have to match): { 1, 2 }
- For kernels with height or width > 7 only a stride of 1 is supported
- Same and Valid Padding are supported
- I*W/O must be between 2.33e-10 and 1, where I is the input quantization scale, W is the weight quantization scale and O is the output quantization scale


## Depth To Space
- Block size of 2 is supported
- Depth must be a multiple of the square of the block size


## Depthwise Convolution 2D
- HWIM format weights are supported
- Kernel heights and widths supported (the kernel does not have to be square): { 1, 2, 3, 5, 7, 9 }
- Stride supported (the height and width stride have to match): { 1, 2 }
- For kernels with height or width > 7 only a stride of 1 is supported
- Same and Valid Padding are supported
- Channel multiplier of 1 is supported, > 1 is not supported
- I*W/O must be between 2.33e-10 and 1, where I is the input quantization scale, W is the weight quantization scale and O is the output quantization scale


## Fully Connected
- HWIO format weights are supported, H and W must be 1
- I*W/O must be between 2.33e-10 and 1, where I is the input quantization scale, W is the weight quantization scale and O is the output quantization scale


## Leaky Relu
- Alpha must be less than 1 and greater than 0


## Max Pooling
- Supported configurations:
    - 1x1 pooling size, 2,2 stride (equivalent to downsample 2x2)
    - 2x2 pooling size, 2,2 stride, valid padding, input sizes must be even
    - 2x2 pooling size, 2,2 stride, same padding, input sizes must be odd
    - 3x3 pooling size, 2,2 stride, valid padding, input sizes must even, maximum tensor width is 417
    - 3x3 pooling size, 2,2 stride, same padding, input sizes must be odd, maximum tensor width is 417
- Input size must not be smaller than the pooling size


## MeanXy
- Supports mean reduction of HxW dimensions to 1x1.
- Only supports Nx7x7xC or Nx8x8xC input with Nx1x1xC output in both cases.


## Multiplication
- A multiplication with constant tensor with shape 1x1x1xC will be replaced with an equivalent depthwise convolution. **This is available only when using Arm NN**
- Multiplication of a variable with a scalar constant is supported in cases where the quantized values in the output are the same as the input, this is replaced with a Reinterpret Quantization operation. **This is available only when using Arm NN**


## Reinterpret Quantization
- No specific restrictions


## Relu
- Lower bound must be less than the upper bound


## Requantize
- Output scale must be bigger than input scale / 128


## Reshape
- No specific restrictions


## Resize
- The resized height or width must 2n or 2n-1 where n in the original height or width
- The resized height and width must both be odd or both be even.


## Sigmoid
- The output for Sigmoid always has a quantization zero point equal to the minimum value of the quantized data type and a quantization scale of 1 / 256.


## Space To Depth
- Block size of greater than 1 is supported
- Input width and height must be a multiple of the block size


## Split
- If spltting the 3rd dimension, the input tensor's 3rd dimension must be a multiple of 16


## Tanh
- The output for Tanh always has a quantization zero point equal to the middle value of the quantized data type and a quantization scale of 1 / 128.


## Transpose
- Transpose is allowed for height, width and channel dimensions only


## Transpose Convolution 2D
- HWIO format weights are supported
- Kernel heights and widths supported (the kernel does not have to be square): { 1, 2, 3, 5, 7, 9 }
- Stride supported (the kernel does have to be square): { 1, 2 }
- Same and Valid Padding are supported
- I*W/O must be between 2.33e-10 and 1, where I is the input quantization scale, W is the weight quantization scale and O is the output quantization scale


For additional details please see driver/support_library/src/SupportQueries.cpp.

**Please contact your Arm FAE for any further questions**
