# A decode megakernel for Gemma 4 26B A4B in llama.cpp

`megakernel.patch` applies to [llama.cpp](https://github.com/ggml-org/llama.cpp) at **b11541** (`f2918ca`). It speeds
up text generation for Gemma 4 26B A4B on NVIDIA GPUs.

The model generates text one token at a time; each token is one decode step. Stock llama.cpp runs about 1,250 CUDA
kernels per decode step. With `GGML_CUDA_MEGAKERNEL=1`, the whole step runs as **one persistent CUDA kernel**: the
embedding, all 30 layers, the final norm, and the LM head with the logit softcap. Prompt processing is unchanged.
Without the variable, the output is bitwise identical to stock llama.cpp's.

Decode speed of this patch, in tokens per second. llama-bench generated 128 tokens after a prompt of the given length
("depth"). Stock and the megakernel ran alternately, and each number is the median of 3 rounds. On the H100 it is
also the median of three separate containers; the L40S had ECC on. [`RESULTS.md`](RESULTS.md) has the details, the
spread, other hosts, and a comparison with smaller changes to stock.

| GPU | GGUF | Depth 512 (stock → megakernel) | Depth 32k (stock → megakernel) |
| --- | --- | --- | --- |
| H100 80GB | QAT Q4_0 | 230 → **377** (+64%) | 211 → **311** (+47%) |
| H100 80GB | Q4_K_M | 215 → **342** (+59%) | 200 → **287** (+44%) |
| L40S 48GB | QAT Q4_0 | 170 → **214** (+26%) | 141 → **171** (+22%) |
| L40S 48GB | Q4_K_M | 153 → **183** (+20%) | 129 → **150** (+17%) |

Files: Google's QAT Q4_0 (`gemma-4-26B_q4_0-it.gguf` from `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`) and
bartowski's Q4_K_M (`google_gemma-4-26B-A4B-it-Q4_K_M.gguf` from `bartowski/google_gemma-4-26B-A4B-it-GGUF`).

## Build and run

```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout f2918ca            # b11541
git apply ../megakernel.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;90" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-server llama-bench

GGML_CUDA_MEGAKERNEL=1 ./build/bin/llama-bench -m gemma-4-26B_q4_0-it.gguf -ngl 99 -fa 1 -p 0 -n 128
```

| Variable | Effect |
| --- | --- |
| `GGML_CUDA_MEGAKERNEL=1` | Run decode steps through the megakernel. |
| `GGML_CUDA_MEGAKERNEL_SPLIT=<cells>` | Once the full-attention KV cache holds this many cells, run the attention of the 5 full-attention layers in a separate tensor-core kernel. Default: 12288 on Hopper, 24576 on other GPUs. 0 means always. |
| `GGML_CUDA_MEGAKERNEL_ARITH=stock` | For Q4_0 / Q5_0 mat-vecs, use stock's arithmetic instead of the megakernel's more exact integer-sum version. Used to compare with stock op by op. This is close to stock's arithmetic but not bit-exact (see exception 2 in [`RESULTS.md`](RESULTS.md)). |
| `GGML_CUDA_MEGAKERNEL_REPACK=0/1` | Turn the repacked weight copies off or on. Default: on for Hopper. |

## Requirements and limits

- Gemma 4 26B A4B only: its shapes are built into the kernel. Weight types: Q4_0, Q8_0, Q5_0, Q4_K, Q5_K, Q6_K.
- The whole model on one CUDA GPU (`-ngl 99`), flash attention on (`-fa 1`), one sequence, one token per decode
  step. Because the whole model must fit in VRAM, it does not help 16 GB cards.
- Built and tested with CUDA 12.8 on sm_89 (L40S) and sm_90 (H100) only. It has not been built or run on sm_80
  (A100) or older GPUs.
- Extra memory: a repacked copy of the Q6_K LM head (+0.6 GB). On Hopper, also repacked copies of the Q4_0 / Q8_0 /
  Q5_0 weights; for the QAT file, that is most of the model again. This memory is allocated for each llama.cpp
  context and kept until the process exits. Each additional context adds it again, and so does a context recreated
  at a different address. llama.cpp's memory logs do not show it.
- llama.cpp's own CPU work per token (copying the logits and sampling) is unchanged: about 0.2 ms per token.

### Known issues

- **No message when the megakernel is not used.** If a requirement above is not met, decoding silently runs stock
  llama.cpp. Examples: flash attention off; several sequences decoded in one batch, as a multi-slot server does;
  RoPE scaling; an unsupported model layout.
- **Partial offload aborts.** With part of the model on the CPU (`-ngl` below the layer count), the process aborts
  instead of falling back to stock. A model split across several GPUs is not detected, so do not use one.
- **Kernel errors are not reported.** The kernel has a watchdog: it gives up if a grid barrier waits more than about
  10 s. If that ever happens, that step's logits are wrong and no error is raised. It has not happened in any test.

## How it works

- `src/models/gemma4.cpp`: for a single-token decode step, the graph is one `GGML_OP_CUSTOM` node. Its userdata
  describes all the weights and both KV caches. On the ggml side, the patch adds one public header
  (`ggml/include/ggml-megakernel.h`) and changes only the CUDA backend; `ggml.c` and `ggml.h` are unchanged.
- `ggml/src/ggml-cuda/megakernel.cu`: the CUDA backend takes that node and starts the kernel as a cooperative launch.
- `ggml/src/ggml-cuda/megakernel-persistent.cuh`: the kernel. Its shared constants, block layouts and helpers are in
  `megakernel-common.cuh`. The work is cut into fixed units (rows of each matrix), each assigned in advance to one
  warp. Each layer has 5 grid-wide barriers. Partial sums are always added in the same order, so the output is
  deterministic. MoE routing (choosing 8 of 128 experts) happens on the GPU without an extra barrier. The kernel
  reads llama.cpp's KV cache where it is, without copying it.
- At long contexts, the attention of the full-attention layers runs in a separate tensor-core kernel
  (`mma.m16n8k16`). Q and P are each split into two float16 parts (hi + lo), and sums are kept in float32.
- A debug version of the kernel (`step<true>`, not used in normal runs) can record and replace every intermediate
  tensor. This is how it was checked against stock op by op.

## Correctness

Summary (details in [`RESULTS.md`](RESULTS.md)):

- **Op by op (the main evidence).** Each operation in each layer is given stock's inputs, and its output is compared
  with stock's. The limit is a relative error of 1e-4. The megakernel matches stock everywhere except in two cases:
  where it deliberately computes more exactly (integer-sum Q4_0 / Q5_0 mat-vecs, float32 attention), and a few
  operations on the QAT file, where rare rounding differences push the average just over the limit. These happen
  when activations are quantized to 8 bits. In a test, the check caught every planted error at or above its limit,
  at exactly the right operation and layer.
- **Against the exact answer.** Where the megakernel is deliberately different, a float64 recomputation shows it is
  as close to the exact answer as stock, or closer, in every block.
- **End to end (a sanity check).** Both builds are fed the same tokens on chat and Wikipedia prompts, and their full
  outputs are compared. The megakernel stays within the limits set by stock llama.cpp's own variation (fusions on or
  off, flash attention on or off, the other GPU), on all three test files and both GPUs. This check only catches
  large errors: stock's own variation is large enough that errors of 0.1-1% inside one operation go unnoticed. A
  separate set with one long prompt fails its limits by one token out of 97. The cause was not identified; candidates
  include stock's less exact float16 attention and expert-routing differences.
- **Determinism.** Two runs give bitwise identical output.
- **Sanitizers.** compute-sanitizer memcheck, racecheck and synccheck report no errors.

[`correctness/`](correctness/README.md) contains the dumper, the comparison script, the pass limits and the chat
prompts, with commands to rerun the end-to-end and op-by-op checks against your own stock reference dumps. It does
not include the long prompt or the float64 recomputation, so those two results cannot be reproduced from this
repository alone.

## Status

A research prototype, not proposed upstream.

Which version was measured:

- **Speed:** the numbers here and in the main tables of [`RESULTS.md`](RESULTS.md) come from this patch.
- **Correctness:** most checks (op by op, the float64 comparison, the long prompt, the planted errors of 0.1-1%) come
  from an earlier revision. The near-limit error test and the analysis of exception 2 in `RESULTS.md` come from this
  patch.

The published patch differs from the earlier revision in four ways:

- A guard for very long contexts. Beyond 163,840 full-attention cache cells (roughly 160k tokens of context), the
  tensor-core attention path would write past the end of a buffer. The kernel now uses its one-kernel path there.
- Two checks that refuse to run the megakernel on a model missing tensors that Gemma 4 always has.
- The kernel's two headers were renamed and moved to the top level of `ggml/src/ggml-cuda/`, and some comments were
  edited.
- Two small host-side fixes. A decode step whose input is an embedding instead of a token id (possible with
  multimodal input) now runs stock llama.cpp instead of hitting an assertion. And two standard headers are now
  included before the kernel's namespace.

None of these changes what a decode step computes at the context lengths measured. Checks on the published patch,
with CUDA 12.8:

- It produces the same output as the earlier revision: every end-to-end metric is equal on all three test files and
  both GPUs (KL divergence to within float64 rounding).
- The end-to-end checks pass, two runs are bitwise identical, and compute-sanitizer reports no errors.
- A decode step with an embedding as input now runs; on the earlier revision it hits an assertion.

Some of these checks ran on the version just before the two host-side fixes. Its compiled GPU code is identical to
the published patch's, function by function.

## License

MIT (see [`LICENSE`](LICENSE)). The patch changes and adds to llama.cpp, which is also MIT-licensed (Copyright (c)
2023-2026 The ggml authors). That license still applies to the llama.cpp code the patch touches.
