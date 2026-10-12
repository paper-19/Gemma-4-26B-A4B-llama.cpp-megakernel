# Correctness harness

The tools behind the end-to-end and op-by-op checks in [`../RESULTS.md`](../RESULTS.md):

| File | What it is |
| --- | --- |
| `refdump.cpp` | `llama-refdump`: runs a model one token at a time over fixed prompts. It writes the logits of every position, and optionally intermediate tensors and the KV cache. With `GGML_CUDA_MEGAKERNEL=1`, it uses the megakernel's debug hooks to record, replace and perturb the megakernel's intermediate tensors. |
| `refcmp.py` | Compares a candidate dump with a reference dump and applies the pass/fail limits. Needs only numpy. |
| `policy.json` | The limits: twice the largest difference stock llama.cpp showed against itself under a harmless change. |
| `prompts/chat/` | The 9 chat prompts, with Gemma 4's chat template already applied (no BOS: `llama-refdump` adds it). |

Not included:

- the reference dumps (about 1.5 GB of logits per model, more with intermediate tensors);
- the 2,477-token prompt behind the long-prompt result;
- the float64 recomputation scripts;
- the scripts that ran all of this on cloud GPUs.

## Build

From the llama.cpp checkout with `megakernel.patch` applied and built (see the main README):

```
g++ -O2 -std=c++17 ../correctness/refdump.cpp -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base \
    -Wl,-rpath,"$PWD/build/bin" -o build/bin/llama-refdump
```

The same binary dumps stock llama.cpp (without `GGML_CUDA_MEGAKERNEL`) and the megakernel (with it). It puts every
layer on the GPU and turns flash attention on (`--no-fa` turns it off).

Use the default build, as in the main README, with `GGML_BACKEND_DL` off. `llama-refdump` finds the megakernel's
debug hooks with `dlsym`. With `GGML_BACKEND_DL=ON`, llama.cpp loads the CUDA backend privately (`RTLD_LOCAL`), so
`dlsym` cannot see them. Then a megakernel run with `--tensors` (the op-by-op check, and `--inject`) stops with "no
debug hooks in this build". End-to-end runs do not use the hooks and work with either build.

## Inputs

| Name | Model | Prompts | Tokens generated |
| --- | --- | --- | --- |
| `it` | instruction-tuned Q4_K_M (`google_gemma-4-26B-A4B-it-Q4_K_M.gguf`) | `prompts/chat/*.txt` | 128 |
| `qat` | QAT Q4_0 (`gemma-4-26B_q4_0-it.gguf`) | `prompts/chat/*.txt` | 128 |
| `base` | base model, Q4_K_M | 6 Wikipedia slices (below) | 64 |

No base-model GGUF had been published at the time. Ours was converted from `google/gemma-4-26B-A4B` to BF16 with
llama.cpp's `convert_hf_to_gguf.py`, then quantized to Q4_K_M with `llama-quantize`.

The Wikipedia slices come from the wikitext-2 raw test split (`wiki.test.raw`, the file `llama-perplexity` uses;
Wikipedia text, CC BY-SA). There are six slices of 2,400 characters. Each starts at a section heading and is cut to
whole lines:

```python
txt = open("wiki.test.raw").read()
for i in range(6):
    o = txt.index("\n = ", len(txt) * (i + 1) // 7) + 1
    s = txt[o:o + 2400]
    open(f"wiki-{i:02d}.txt", "w").write(s[:s.rfind("\n") + 1])
```

## End-to-end check

1. Reference: stock llama.cpp generates greedily.
2. Candidate: the megakernel is fed exactly the reference's tokens ("teacher forced", `--tokens-from`). So every
   position sees the same input, and one early difference cannot snowball.
3. Compare every position's full next-token distribution.

```
B=build/bin; P=../correctness
$B/llama-refdump -m it.gguf -o ref/it --gen 128 --ctx 4096 --kv $P/prompts/chat/*.txt
GGML_CUDA_MEGAKERNEL=1 $B/llama-refdump -m it.gguf -o cand/it --tokens-from ref/it --ctx 4096 --kv $P/prompts/chat/*.txt
python $P/refcmp.py ref/it cand/it --policy $P/policy.json --policy-set it-chat
```

Use `--policy-set qat-chat` for the QAT file and `base-wiki` for the base model (`--gen 64`, the Wikipedia slices).

The measures, reported over all positions and separately for the generated tokens (the "continuation"):

| Measure | Meaning |
| --- | --- |
| `kld` | KL divergence from the reference's next-token distribution to the candidate's, over the full vocabulary, in float64 |
| `top1` | how often both pick the same most likely token |
| `confident_flip_pct` | how often they pick different tokens although the reference was more than 50% sure of its token |
| `logit_rms` | RMS of the difference of the raw logits |

## Op-by-op check

Each of the megakernel's operations is given stock's input, and its output is compared with stock's. This way, a
small error inside one operation cannot hide among the rounding differences of the whole model.

1. A "watched" reference: stock, fed the reference's tokens, also records its intermediate tensors.
2. The candidate records the same tensors. At every position and layer, it replaces each operation's input with the
   reference's value (`--replay`), so each operation's error is its own.
3. Compare each operation in each layer: the relative error ||candidate - reference|| / ||reference||, with a limit
   of 1e-4.

```
W=l_out,attn_out,ffn_mlp,ffn_moe-,ffn_moe_topk,ffn_moe_logits,Kcur_pos,Vcur_normed,result_norm,Qcur
OPS=Qcur_pos,Kcur_pos,Vcur_normed,attn_out,ffn_moe_logits,ffn_mlp,ffn_moe-,l_out
$B/llama-refdump -m it.gguf -o ref/it-watch --tokens-from ref/it --tensors $W --ctx 4096 --kv $P/prompts/chat/*.txt
GGML_CUDA_MEGAKERNEL=1 GGML_CUDA_MEGAKERNEL_ARITH=stock $B/llama-refdump -m it.gguf -o cand/it-op \
    --tokens-from ref/it-watch --tensors $W --replay ref/it-watch --replay-tensors $OPS --ctx 4096 --kv $P/prompts/chat/*.txt
python $P/refcmp.py ref/it-watch cand/it-op --policy $P/policy.json --policy-set it-chat-opreplay
```

`GGML_CUDA_MEGAKERNEL_ARITH=stock` switches the Q4_0 / Q5_0 mat-vecs to stock's arithmetic, so that the megakernel's
deliberate improvements do not exceed the limit. This is close to stock's arithmetic, not bit for bit. Two
exceptions are expected:

- the attention output of the 5 full-attention layers: the megakernel's float32 attention is deliberately more exact
  than stock's float16;
- on the QAT file, a few operations at 1.0-1.75e-4, from rare rounding differences when activations are quantized
  to 8 bits (exception 2 in `../RESULTS.md`).

`--inject name@pos:rel` plants a fault. For example, `--inject l_out-14@*:0.0002` adds relative noise of 2e-4 to
layer 14's output at every position. The op-by-op check should report it at exactly that operation and layer.

## How the limits were set

Each limit in `policy.json` is twice the largest difference measured between stock llama.cpp and stock llama.cpp with
one harmless change. The limits were set before any megakernel result was judged. Each set's `_floors` lists the runs
that set it, as `<gpu>/<model>/<variant>`:

| Variant | Change |
| --- | --- |
| `nofusion` | llama.cpp's CUDA kernel fusions off (`GGML_CUDA_DISABLE_FUSION=1`) |
| `nofa` | flash attention off |
| `stock` (on the other GPU) | the same stock build run on the other GPU (H100 against L40S) |
| `replay-*`, `opreplay-*`, `watch-*` | the same changes, in the replay and watched modes |

Two minimums apply where twice the floor would leave no room: top-1 agreement may always drop by 1 point, and
confident flips may always reach 0.5%.

The sets:

- names ending in `-watch`, `-replay` and `-opreplay` judge the watched, replayed and op-by-op modes;
- `it-long` is the long-prompt set (its prompt is not included);
- the `base0` sets (for a base Q4_0 file) are included but not used in `../RESULTS.md`.
