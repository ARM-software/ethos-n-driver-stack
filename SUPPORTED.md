# Arm® Ethos™-N Driver Stack Operator Support


## General tensor support information
- 8-bit signed quantized and 8-bit unsigned quantized datatypes are supported.
- Tensors with up to 4 dimensions are supported.
- A "Batch" dimension >1 is not supported.
- NHWC tensors are supported.
- Per-channel quantization is supported only for the weights and bias tensors of a limited set of operations.
- Most operations are not supported when the input or output tensor depths are too large. For more information, see `IsTensorDepthSupported` in `SupportQueries.cpp` in the Support Library.


## Addition
- Element by element addition is supported.
- If Arm NN is used, the addition of a variable with a scalar constant is supported when the quantized values in the output are the same as the input. The Arm NN backend replaces this addition with a reinterpret quantization operation.


## Average pooling
- Pooling size 3 x 3, with stride 1, 1 and SAME padding is supported.
- "Mean" average pooling is supported where the pooling size is 7 or 8 and the input width and height is equal to the pool size.


## Concatenation
- Output quantization scale smaller than the input quantization scale divided by 128 is not supported.
- If concatenating along the channel dimension, the channel dimension of every input tensor must be a multiple of 16.


## Constant
- No specific restrictions.


## Convolution 2D
- HWIO format weights are supported.
- The supported kernel heights and widths (the kernel does not have to be square) are: { 1, 2, 3, 5, 7, 9 }.
- The supported strides (the height and width stride have to match) are: { 1, 2 }.
- For kernels with height or width >7, only a stride of 1 is supported.
- SAME and VALID padding are supported.
- I*W/O must be between 0 and 65536, where:
     - I is the input quantization scale.
     - W is the weight quantization scale.
     - O is the output quantization scale.


## Depth to space
- A block size of 2 is supported.
- Depth must be a multiple of the square of the block size.


## Depthwise convolution 2D
- HWIM format weights are supported.
- The supported kernel heights and widths (the kernel does not have to be square) are: { 1, 2, 3, 5, 7, 9 }.
- The supported strides (the height and width stride have to match) are: { 1, 2 }.
- For kernels with height or width >7, only a stride of 1 is supported.
- SAME and VALID padding are supported.
- A channel multiplier of 1 is supported. A channel multiplier >1 is not supported.
- I*W/O must be between 0 and 65536, where:
    - I is the input quantization scale.
    - W is the weight quantization scale.
    - O is the output quantization scale.


## Fully connected
- HWIO format weights are supported, H and W must be 1.
- I*W/O must be between 0 and 65536, where:
    - I is the input quantization scale.
    - W is the weight quantization scale.
    - O is the output quantization scale.


## Leaky ReLU
- Alpha must be less than 1 and greater than 0.


## Max pooling
- Supported configurations:
    - 1 x 1 pooling size, 2, 2 stride (equivalent to downsample 2 x 2).
    - 2 x 2 pooling size, 2, 2 stride, VALID padding, input sizes must be even.
    - 2 x 2 pooling size, 2, 2 stride, SAME padding, input sizes must be odd.
    - 3 x 3 pooling size, 2, 2 stride, VALID padding, input sizes must even, maximum tensor width is 417.
    - 3 x 3 pooling size, 2, 2 stride, SAME padding, input sizes must be odd, maximum tensor width is 417.
- Input size must not be smaller than the pooling size.


## MeanXy
- Supports mean reduction of H x W dimensions to 1 x 1.
- Only supports:
    - N x 7 x 7 x C input with N x 1 x 1 x C output.
    - N x 8 x 8 x C input with N x 1 x 1 x C output.


## Multiplication
- If Arm NN is used, the multiplication of a tensor with a constant tensor is supported when the constant shape is 1 x 1 x 1 x C. The Arm NN backend replaces this multiplication with a depthwise convolution.
- If Arm NN is used, the multiplication of a variable with a scalar constant is supported when the quantized values in the output are the same as the input. The Arm NN backend replaces this multiplication with a reinterpret quantization operation.


## Pad
- Only zero padding in the H and W dimension is supported.
- Padding of up to 7 each side of the tensor in those dimensions is supported.
    - Padding amounts can differ per side, e.g. pad of 1 before the tensor in the H dimension and a pad of 3 after the tensor in the H dimension.
- Quantization for input and output tensors must be identical.


## Reinterpret quantization
- No specific restrictions.


## ReLU
- Lower bound must be less than the upper bound.


## Requantize
- Output quantization scale smaller than the input quantization scale divided by 128 is not supported.
- Requantize with different input/output type is supported.


## Reshape
- No specific restrictions.


## Resize
- The resized height or width must be 2n or 2n-1 where n is the original height or width.
- If resized height and width are not both odd or both even the result might be less accurate.
- Some Resize Billinear configurations (align_corners=True, half_pixel_centres=True when heights and widths are not both even or both odd) produce inaccurate results.


## Sigmoid
- The output for sigmoid always has a quantization zero point equal to the minimum value of the quantized data type and a quantization scale of 1 / 256.


## Split
- If splitting along the channel dimension, the channel dimension of every output tensor must be a multiple of 16.


## Tanh
- The output for tanh always has a quantization zero point equal to the middle value of the quantized data type and a quantization scale of 1 / 128.


## Transpose convolution 2D
- HWIO format weights are supported.
- The supported kernel heights and widths (the kernel does not have to be square) are: { 1, 2, 3, 5, 7, 9 }.
- Only a stride of 2 is supported.
- SAME and VALID padding are supported.
- I*W/O must be between 0 and 65536, where:
     - I is the input quantization scale.
     - W is the weight quantization scale.
     - O is the output quantization scale.


## Temporarily Disabled Operations
The following operations have been temporarily disabled and will be enabled in a future release. These operations are now only supported at the "EstimateOnly" level. They can be used in SPA and will contribute zero performance impact, but cannot be compiled for execution on the actual hardware.
- Transpose
- SpaceToDepth


For more information, see `driver/support_library/src/SupportQueries.cpp`.

**Please contact your Arm FAE for any further questions.**
