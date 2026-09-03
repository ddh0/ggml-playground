
# A Brief Guide to GGML

> DISCLAIMER: I intend for this guide to be helpful and correct, but I make no guarantees. If you find something is inaccurate or missing, please let me know!

## Intro

GGML is a C/C++ tensor library for machine learning. It implements:
- A set of tensor operations
- Automatic differentiation
- Basic optimization algorithms

GGML features include:
- Low-level cross-platform implementation
- Integer quantization support
- Broad hardware support
- Automatic differentiation
- ADAM and L-BFGS optimizers
- Minimal third-party dependencies
- Zero memory allocations during runtime

It is most well-known for being the engine behind [llama.cpp](https://github.com/ggml-org/llama.cpp) and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

The library allows the user to define a function using available tensor operations. This function definition is represented internally via a **computation graph**. Each tensor operation in the function definition corresponds to a node in the graph. Once the graph is defined, the user can choose to compute the function's value (forward pass) and/or its gradient (backward pass) with respect to input variables.

GGML operations are implemented across several compute backends, including CPU, Metal, CUDA, Vulkan, SYCL, WebGPU, and more. Support for individual operations across each backend is detailed at [llama.cpp/docs/ops.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md).

## Computing with GGML

The lifecycle of a GGML computation has three distinct phases:

1. You define the tensors and the operations connecting them (for example, inference graphs per model architecture). No computation is performed yet.
2. You build the graph to determine the execution order and prepare the required nodes (i.e. forward expansion). No computation is performed yet.
3. You trigger the computation of the graph, utilizing one or more of the GGML compute backends (CPU, Metal, CUDA, Vulkan, ...) for hardware accelerated tensor math.

```c

//
// suppose we want to find the result of the function `f`:
//
//     f(x) = a*x^2 + b
//
// ... where `x` = 2, `a` = 3, and `b` = 4
//

// start by choosing params
struct ggml_init_params params = {
    .mem_size   = 16*1024*1024, // arbitrary for example
    .mem_buffer = NULL,         // if NULL, GGML will allocate the buffer itself
};

// create context with params
struct ggml_context * ctx = ggml_init(params);

// define the input to the graph, and mark it as such
struct ggml_tensor * x  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
ggml_set_param(ctx, x);

// define the components of the graph; tensors are nodes, operations are edges
struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);

// build the graph
struct ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, f);

// set variables
ggml_set_f32(x, 2.0f);
ggml_set_f32(a, 3.0f);
ggml_set_f32(b, 4.0f);

// execute computation graph
ggml_graph_compute_with_ctx(ctx, gf, n_threads);

// result is now stored in `ggml_tensor * f`, we convert that 1D tensor to float
printf("f = %f\n", ggml_get_f32_1d(f, 0));
```

Note that the computation graph must be static; in other words, you cannot implement traditional programming loops (like `for token in sequence`) within the graph itself.

## `ggml_context`

GGML avoids the overhead of frequent `malloc` and `free` calls during runtime by using a static, monolithic compute context.

When you call `ggml_init`, you provide a `ggml_init_params` struct. This defines a contiguous memory buffer used for tensor metadata (the structs themselves) and, if the `.no_alloc` flag is set to false, the actual tensor data.

In CPU-only contexts, the `ggml_context` often holds both metadata and data. However, when utilizing one of the compute backends, `.no_alloc` is set to true, and the `ggml_context` is used only for tensor metadata, while the actual tensor data is allocated in a `ggml_backend_buffer`.

You must know or estimate your memory requirements upfront. This is ideal for embedded or resource-constrained environments.

Because allocations are linear within the buffer, fragmentation is not an issue.

Defining a graph once and running it multiple times uses the same pre-allocated buffer, eliminating runtime allocation overhead.

**TIP:** If you are unsure of your exact memory requirements, allocate a large buffer, run your model once, and then call `ggml_used_mem(ctx)` to find the actual high-water mark. This value can then be used to optimize your production `mem_size`.

---

## `ggml_tensor`

Tensors in `GGML` are instances of `struct ggml_tensor`, which is defined near the top of `llama.cpp/ggml/include/ggml.h` as follows:

```c

    // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type type;

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct ggml_tensor * src[GGML_MAX_SRC];

        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[8];
    };

```

A few things to note about `ggml_tensor`s:
- `ggml_tensors` are technically always 4-dimensional (because `GGML_MAX_DIMS` is 4)
- the smallest valid `ggml_tensor * t` shape is `t->ne[i] == 1` for all i = {0,1,2,3} (written as `[1, 1, 1, 1]`)
- GGML tensors are stored such that the first dimension (ne[0]) is contiguous in memory. This is similar to column-major layout in some frameworks, meaning the fastest-changing index is the first one.
- Strides are explicitly defined in bytes. For example, `t->nb[0]` is always the size of the data type in bytes (e.g., 4 for F32). `t->nb[i]` is the byte-distance between the start of one element and the next in that dimension. This contrasts with frameworks like PyTorch, where strides are measured in number of elements. Byte-level strides allow GGML to support quantized integer datatypes and mixed-precision tensors within a single consistent memory model.
- Many operations like `ggml_transpose`, `ggml_reshape`, or `ggml_view` are zero-copy - no data is moved during computation. Such operations merely create a new descriptor with `ne` and `nb` values mutated as appropriate for the expected result.

**TIP:** The GGML operation `ggml_mul_mat(A, B)` computes the equivalent of **$B^T \times A$** in PyTorch / NumPy! Specifically, it treats the second matrix B as if it were already transposed. If you are converting a model from HF Transformers, be very careful with your dimension ordering; you may need to swap your input tensor shapes to match GGML's expectations.

---

## Automatic Differentiation

_Not usually relevant, but nice to know._

GGML's autograd is driven by the computation graph. By marking a tensor as a parameter using `ggml_set_param(ctx, tensor)`, you instruct the library to track its gradient during the backward pass.

`ggml_build_forward_expand` populates the graph with the sequence of operations required to compute the output.

`ggml_build_backward_expand` traverses the graph in reverse order, applying the chain rule to compute the adjoints (gradients) of the input tensors.

This mechanism is highly efficient because the graph topology is static. You can compute the gradient for a complex neural network hundreds of times without ever re-defining the structure of the network.

---

## Quantization

Quantization in GGML is implemented using a block-based approach. Instead of quantizing every single value individually, values are grouped into blocks (defined by `blck_size`), and a scale factor is applied to the block. This preserves significantly more precision and allows for much more efficient SIMD (Single Instruction, Multiple Data) implementations.

When working with quantized tensors, you must respect the block boundaries, as the library's kernels are optimized for these specific memory alignments.

### `ggml_type` cheat sheet

*Only includes the most common/recommended tensor types (no ternary types, no MXFP4, no unsupported types like Q8_1, etc.).*

| Common name | Bits per weight | Quantization type | `ggml_type` name    | `ggml_type` value | Description                                                             | Requires imatrix? |
| :---------- | :-------------- | :---------------- | :------------------ | :---------------- | :---------------------------------------------------------------------- | :---------------- |
| `F32`       | 32.00           | -                 | `GGML_TYPE_F32`     | `0`               | IEEE 754 single-precision float                                         | NO                |
| `F16`       | 16.00           | -                 | `GGML_TYPE_F16`     | `1`               | IEEE 754 half-precision float                                           | NO                |
| `Q4_0`      | 4.50            | static            | `GGML_TYPE_Q4_0`    | `2`               | Block-wise (32 weights). Symmetric, 4-bit weights + 1 `fp16` scale.     | NO                |
| `Q4_1`      | 5.00            | static            | `GGML_TYPE_Q4_1`    | `3`               | Block-wise (32 weights). 4-bit weights + 1 `fp16` scale + 1 `fp16` min. | NO                |
| `Q5_0`      | 5.50            | static            | `GGML_TYPE_Q5_0`    | `6`               | Block-wise (32 weights). Symmetric, 5-bit weights + 1 `fp16` scale.     | NO                |
| `Q5_1`      | 6.00            | static            | `GGML_TYPE_Q5_1`    | `7`               | Block-wise (32 weights). 5-bit weights + 1 `fp16` scale + 1 `fp16` min. | NO                |
| `Q8_0`      | 8.50            | static            | `GGML_TYPE_Q8_0`    | `8`               | Block-wise (32 weights). Symmetric, 8-bit weights + 1 `fp16` scale.     | NO                |
| `Q2_K`      | 2.625           | k-quant           | `GGML_TYPE_Q2_K`    | `10`              | Block-wise (256 weights). 2-bit weights with hierarchical quantization. | NO                |
| `Q3_K`      | 3.4375          | k-quant           | `GGML_TYPE_Q3_K`    | `11`              | Block-wise (256 weights). 3-bit weights with hierarchical quantization. | NO                |
| `Q4_K`      | 4.50            | k-quant           | `GGML_TYPE_Q4_K`    | `12`              | Block-wise (256 weights). 4-bit weights with hierarchical quantization. | NO                |
| `Q5_K`      | 5.50            | k-quant           | `GGML_TYPE_Q5_K`    | `13`              | Block-wise (256 weights). 5-bit weights with hierarchical quantization. | NO                |
| `Q6_K`      | 6.5625          | k-quant           | `GGML_TYPE_Q6_K`    | `14`              | Block-wise (256 weights). 6-bit weights with hierarchical quantization. | NO                |
| `IQ2_XXS`   | 2.0625          | i-quant           | `GGML_TYPE_IQ2_XXS` | `16`              | Block-wise (256 weights). 2-bit codebook-based quantization.            | YES               |
| `IQ2_XS`    | 2.3125          | i-quant           | `GGML_TYPE_IQ2_XS`  | `17`              | Block-wise (256 weights). 2-bit codebook-based with sub-scales.         | YES               |
| `IQ3_XXS`   | 3.0625          | i-quant           | `GGML_TYPE_IQ3_XXS` | `18`              | Block-wise (256 weights). 3-bit codebook-based quantization.            | NO                |
| `IQ1_S`     | 1.5625          | i-quant           | `GGML_TYPE_IQ1_S`   | `19`              | Block-wise (256 weights). 1-bit codebook-based with sub-scales.         | YES               |
| `IQ4_NL`    | 4.50            | i-quant           | `GGML_TYPE_IQ4_NL`  | `20`              | Block-wise (32 weights). 4-bit non-linear map + 1 `fp16` scale.         | NO                |
| `IQ3_S`     | 3.3125          | i-quant           | `GGML_TYPE_IQ3_S`   | `21`              | Block-wise (256 weights). 3-bit codebook-based with sub-scales.         | NO                |
| `IQ2_S`     | 2.5625          | i-quant           | `GGML_TYPE_IQ2_S`   | `22`              | Block-wise (256 weights). 2-bit codebook-based with sub-scales.         | YES               |
| `IQ4_XS`    | 4.25            | i-quant           | `GGML_TYPE_IQ4_XS`  | `23`              | Block-wise (256 weights). 4-bit non-linear map + hierarchical scales.   | NO                |
| `IQ1_M`     | 1.75            | i-quant           | `GGML_TYPE_IQ1_M`   | `29`              | Block-wise (256 weights). 1-bit codebook-based, "medium" variant.       | YES               |
| `BF16`      | 16.00           | -                 | `GGML_TYPE_BF16`    | `30`              | bfloat16 half-precision float                                           | NO                |
| `Q1_0`      | 1.125           | static            | `GGML_TYPE_Q1_0`    | `41`              | Block-wise (128 weights). Symmetric, 1-bit weights + 1 `fp16` scale.    | NO                |

---

## Misc. notes

- llama.cpp is the primary development ground for GGML. The folder at `ggml-org/llama.cpp/ggml` is occasionally synced with the actual `ggml-org/ggml` repo manually.
- For multi-GPU or hybrid CPU/GPU setups, GGML provides `ggml_backend_sched`, which handles the distribution of tensors and operations across multiple backends and buffers.

---

## References

- [ggml-org/ggml](https://github.com/ggml-org/ggml)
  + [`README`](https://github.com/ggml-org/ggml/blob/master/README.md)
  + [`include/ggml.h`](https://github.com/ggml-org/ggml/blob/master/include/ggml.h)
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
  + [`README`](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
  + [`include/llama.h`](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)
- [🤗 HF Blog - Introduction to GGML](https://huggingface.co/blog/introduction-to-ggml)
