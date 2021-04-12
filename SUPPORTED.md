# Arm® Ethos™-N Driver Stack Operator Support

## General Tensor Support
- 8 bit Signed quantized and 8 Bit Unsigned quantized datatypes are supported
- Tensors with up to 4 dimensions are supported
- A "Batch" dimension > 1 is not supported
- For operations which utilize the "Channels" dimension (e.g. Convolution) there is a limit to how large the input channel dimension can be. For more details see IsTensorDepthSupported in SupportQueries.cpp in the Support Library
- NHWC and NCHW tensors are supported
- Per tensor and per channel quantization (for output channels only) is supported


## Convolution 2D
- HWIO format weights are supported
- Kernel heights and widths supported (The kernel does not have to be square): { 1, 2, 3, 5, 7, 9 }
- Stride supported (The height and width stride have to match): { 1, 2 }
- For kernels with height or width > 7 only a stride of 1 is supported
- Same and Valid Padding is supported
- I*W/O must be between 2.33e-10 and 1, where I is the input quantization scale, W is the weight quantization scale and O is the output quantization scale


## Depthwise Convolution 2D
- HWIM format weights are supported
- Kernel heights and widths supported (The kernel does not have to be square): { 1, 2, 3, 5, 7, 9 }
- Stride supported (The height and width stride have to match): { 1, 2 }
- For kernels with height or width > 7 only a stride of 1 is supported
- Same and Valid Padding is supported
- Channel multiplier of 1 is supported, > 1 is not supported
- I*W/O must be between 2.33e-10 and 1, where I is the input quantization scale, W is the weight quantization scale and O is the output quantization scale


## Transpose Convolution 2D
- HWIO format weights are supported
- Kernel heights and widths supported (The kernel does not have to be square): { 1, 2, 3, 5, 7, 9 }
- Stride supported: { 1, 2 }
- Same and Valid Padding is supported
- I*W/O must be between 2.33e-10 and 1, where I is the input quantization scale, W is the weight quantization scale and O is the output quantization scale


## Concatentation
- Output quantization scale smaller than input quantization scale / 128 is not supported
- If concatentating the 3rd dimension, the input tensors 3rd dimension must be a multiple of 16


## Split
- If spltting the 3rd dimension, the input tensors 3rd dimension must be a multiple of 16


## Addition
- Element by element addition is supported
- Bias add with a constant is supported


## Fully Connected
- HWIO format weights are supported, H and W must be 1
- I*W/O must be between 2.33e-10 and 1, where I is the input quantization scale, W is the weight quantization scale and O is the output quantization scale


## Relu
- Lower bound must be less than the upper bound


## Leaky Relu
- Alpha must be less than 1 and greater than 0


## Requantize
- Output scale must be bigger than input scale / 128


## Sigmoid
- No specific restrictions


## Average Pooling
- Pooling size 3x3, with stride 1,1 and same padding is supported, input height * width must be <= 61440
- A "Mean" Average pooling is supported where he pooling size is 7 or 8 and the input width and height is equal to the pool size


## Max Pooling
- Support configuration:
    - 2x2 pooling size, 2,2 stride, valid padding, input sizes must be even
    - 2x2 pooling size, 2,2 stride, same padding, input sizes must be odd
    - 3x3 pooling size, 2,2 stride, valid padding, input sizes must even, maximum tensor width is 417
    - 3x3 pooling size, 2,2 stride, same padding, input sizes must be odd, maximum tensor width is 417
- Input size must not be smaller than the pooling size


## Reshape
- No specific restrictions


## Depth To Space
- Block size of 2 is supported
- Depth must be a multiple of the square of the block size


## Resize
- The resized height or width must 2n or 2n-1 where n in the original height or width
- The resized height and width must both be odd or both be even.

# Additional substitutions in the Arm NN backend
In the Arm NN backend there are substitutions applied to enable certain operators when using the NPU with Arm NN.

## Multiplication
A multiplication with constant tensor with shape 1x1x1xC will be replaced with an equivalent depthwise convolution.


For additional details please see driver/support_library/src/SupportQueries.cpp.

**Please contact your Arm FAE for any further questions**