# Results

Measured 2026-10-12 on Modal cloud GPUs (NVIDIA H100 80GB HBM3, NVIDIA L40S 48GB), llama.cpp b11541 built for sm_89 and
sm_90 with CUDA 12.8, `-ngl 99 -fa 1`.

## Decode speed

Tokens per second on the published patch. llama-bench generated 128 tokens after a prompt of the given length
("depth"; `-p 0 -n 128 -d <depth> -r 2`). In each container, stock and the megakernel ran alternately in 3 rounds.
Each cell is the median of the rounds, with the lowest and highest in brackets. "No split" turns off the separate
tensor-core attention kernel for long contexts (`GGML_CUDA_MEGAKERNEL_SPLIT=100000000`), to show what it adds.

Speed depends on the host as well as the GPU: the driver, the CPU and the ECC setting all matter. Compare numbers only
within one table. [Other hosts and the earlier revision](#other-hosts-and-the-earlier-revision) shows how much hosts
differ.

### H100 80GB

Three separate containers, all on the same kind of host (driver 580.95.05, 17 CPU threads). Each cell is the median
of the three containers' results, with the lowest and highest container in brackets. "No repack" turns off the
repacked copies of the Q4_0 / Q8_0 / Q5_0 weights (`GGML_CUDA_MEGAKERNEL_REPACK=0`); the LM head stays repacked.

| GGUF | Depth | Stock | Megakernel | Megakernel, no split | Megakernel, no repack | Megakernel vs stock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QAT Q4_0 | 512 | 230.1 (230.0-230.8) | 377.1 (376.4-377.3) | 378.0 (377.1-378.1) | 370.3 (370.3-370.5) | +63.6% (+63.5 to +64.0) |
| QAT Q4_0 | 4096 | 224.9 (224.9-225.5) | 350.6 (350.5-351.2) | 350.6 (350.2-350.9) | 343.0 (342.4-343.3) | +55.9% (+55.7 to +55.9) |
| QAT Q4_0 | 16384 | 218.2 (217.6-218.4) | 330.3 (329.4-330.5) | 308.3 (307.7-308.7) | 324.2 (324.2-324.4) | +51.4% (+50.8 to +51.8) |
| QAT Q4_0 | 32768 | 210.8 (210.8-212.1) | 311.0 (310.4-311.8) | 268.4 (267.7-269.2) | 305.5 (304.1-305.6) | +47.3% (+47.0 to +47.6) |
| Q4_K_M | 512 | 215.3 (215.3-216.4) | 342.4 (341.6-342.6) | 341.8 (341.2-342.4) | 333.3 (333.2-334.1) | +58.6% (+58.3 to +59.0) |
| Q4_K_M | 4096 | 211.2 (210.8-211.5) | 319.5 (319.2-320.3) | 319.8 (319.8-319.9) | 312.1 (311.9-312.3) | +51.5% (+51.1 to +51.6) |
| Q4_K_M | 16384 | 204.8 (204.8-205.5) | 304.5 (303.7-304.5) | 284.1 (283.7-285.0) | 296.6 (296.3-296.9) | +48.3% (+48.2 to +48.7) |
| Q4_K_M | 32768 | 199.7 (199.2-199.9) | 287.4 (286.9-287.8) | 249.7 (247.7-250.0) | 281.4 (280.7-281.4) | +44.0% (+43.8 to +44.1) |

### L40S 48GB

One container, with ECC on (46,068 MiB visible), like most L40S hosts in these runs.

| GGUF | Depth | Stock | Megakernel | Megakernel, no split | Megakernel vs stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| QAT Q4_0 | 512 | 169.5 (169.3-169.5) | 214.4 (214.4-214.5) | 214.4 (214.4-214.5) | +26.5% |
| QAT Q4_0 | 4096 | 160.4 (160.1-160.7) | 202.9 (202.8-202.9) | 203.0 (202.8-203.0) | +26.5% |
| QAT Q4_0 | 16384 | 150.8 (150.7-151.1) | 186.3 (186.2-186.4) | 186.3 (186.0-186.5) | +23.6% |
| QAT Q4_0 | 32768 | 140.5 (140.3-140.5) | 170.9 (170.7-171.1) | 167.1 (166.9-167.5) | +21.7% |
| Q4_K_M | 512 | 152.5 (152.4-152.8) | 183.0 (183.0-183.0) | 182.6 (182.5-183.1) | +20.0% |
| Q4_K_M | 4096 | 145.1 (144.8-145.1) | 174.8 (174.7-174.9) | 174.3 (173.7-174.8) | +20.5% |
| Q4_K_M | 16384 | 137.1 (137.1-137.3) | 162.7 (162.3-162.8) | 162.3 (161.8-162.3) | +18.7% |
| Q4_K_M | 32768 | 128.5 (128.1-128.5) | 149.8 (149.4-149.9) | 148.1 (147.7-148.2) | +16.6% |

### Repacked weights on Hopper

On the H100, the repacked copies of the Q4_0 / Q8_0 / Q5_0 weights make the megakernel 1.8-2.7% faster. Without
them, it is still 61% faster than stock on the QAT file at depth 512 (370.3 against 230.1 tok/s), and 55% faster on
Q4_K_M. With them, the gains are 64% and 59%. So almost all of the gain does not depend on the weight layout.

### Other hosts and the earlier revision

**L40S with ECC off.** One L40S container landed on a host with ECC turned off (49,140 MiB visible). This leaves more
memory bandwidth. Both builds were faster, and the megakernel gained more, because it is closer to the bandwidth
limit:

| GGUF | Depth | Stock | Megakernel | Megakernel vs stock |
| --- | ---: | ---: | ---: | ---: |
| QAT Q4_0 | 512 | 183.9 | 244.8 | +33.1% |
| QAT Q4_0 | 32768 | 153.0 | 195.3 | +27.6% |
| Q4_K_M | 512 | 167.0 | 212.7 | +27.4% |
| Q4_K_M | 32768 | 141.0 | 173.1 | +22.7% |

**H100 on a different host.** One H100 container landed on a host with a newer NVIDIA driver (610.57.04) and 8 CPU
threads instead of 17. It ran the version just before the two small host-side fixes, whose compiled GPU code is
identical to the published patch's, function by function. Compared with the H100 table above, stock and the
megakernel each lost 0.10-0.18 ms per token. A loss of similar size in both points to work on the CPU rather than on
the GPU. A megakernel token takes less time, so the same loss is a larger share of it:

| GGUF | Depth | Stock | Megakernel | Megakernel, no split | Megakernel vs stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| QAT Q4_0 | 512 | 224.0 (223.9-224.4) | 358.2 (357.9-358.4) | 358.0 (357.2-358.5) | +59.9% |
| QAT Q4_0 | 32768 | 206.4 (206.3-206.8) | 294.9 (294.3-295.5) | 256.7 (256.3-256.9) | +42.9% |
| Q4_K_M | 512 | 210.0 (209.8-210.1) | 326.7 (326.5-326.9) | 325.7 (325.6-325.8) | +55.6% |
| Q4_K_M | 32768 | 194.3 (194.2-194.3) | 274.1 (273.8-274.7) | 240.2 (238.9-240.3) | +41.1% |

**The earlier revision.** The first measurements were made on an earlier revision of the patch: before a
long-context guard, two support checks, the header rename and two host-side fixes. They agree with the published
patch within about 1% on both GPUs (the L40S with ECC on):

<details>
<summary>Earlier revision: full tables</summary>

H100 80GB:

| GGUF | Depth | Stock | Megakernel | Megakernel, no split | Megakernel vs stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| QAT Q4_0 | 512 | 229.9 (227.3-230.9) | 376.6 (375.3-377.4) | 377.3 (375.4-377.8) | +63.8% |
| QAT Q4_0 | 4096 | 224.6 (224.5-225.4) | 350.6 (348.8-351.3) | 348.6 (347.9-350.5) | +56.1% |
| QAT Q4_0 | 16384 | 217.3 (217.0-217.5) | 330.1 (329.6-331.1) | 308.4 (308.2-309.5) | +51.9% |
| QAT Q4_0 | 32768 | 211.5 (210.5-211.6) | 308.9 (308.7-311.8) | 268.2 (265.4-268.7) | +46.0% |
| Q4_K_M | 512 | 215.8 (215.8-216.2) | 342.1 (342.1-343.0) | 341.9 (341.1-342.7) | +58.5% |
| Q4_K_M | 4096 | 211.5 (210.8-211.8) | 319.6 (319.4-319.8) | 319.3 (319.2-319.8) | +51.1% |
| Q4_K_M | 16384 | 204.8 (204.7-205.6) | 304.3 (304.1-304.4) | 284.6 (284.5-284.7) | +48.6% |
| Q4_K_M | 32768 | 200.0 (199.9-200.3) | 286.8 (286.6-288.0) | 249.9 (249.5-250.4) | +43.4% |

L40S 48GB:

| GGUF | Depth | Stock | Megakernel | Megakernel, no split | Megakernel vs stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| QAT Q4_0 | 512 | 171.7 (171.5-171.9) | 216.2 (216.2-216.3) | 216.2 (216.0-216.2) | +25.9% |
| QAT Q4_0 | 4096 | 162.5 (162.3-162.7) | 206.2 (206.1-206.3) | 206.1 (206.0-206.3) | +26.9% |
| QAT Q4_0 | 16384 | 152.5 (152.3-152.5) | 188.6 (188.5-188.8) | 188.5 (188.4-188.6) | +23.7% |
| QAT Q4_0 | 32768 | 141.4 (141.2-141.5) | 172.5 (172.5-172.6) | 168.3 (168.2-168.5) | +22.0% |
| Q4_K_M | 512 | 154.1 (154.1-154.6) | 184.0 (184.0-184.1) | 184.1 (184.0-184.1) | +19.4% |
| Q4_K_M | 4096 | 146.5 (146.4-146.8) | 176.6 (176.6-176.7) | 176.6 (176.6-176.7) | +20.6% |
| Q4_K_M | 16384 | 138.1 (138.0-138.4) | 163.7 (163.7-163.8) | 164.0 (163.9-164.0) | +18.5% |
| Q4_K_M | 32768 | 129.1 (128.9-129.5) | 151.2 (151.1-151.7) | 148.6 (148.4-148.7) | +17.1% |

</details>

## Compared with smaller changes to stock

How much of the gain could stock llama.cpp get without a megakernel? Two smaller changes were measured on this
model:

- **Targeted fusions.** Three changes to the CUDA backend's decode path. The output stays bitwise identical. Each
  activation is quantized to q8_1 once and shared by every mat-vec that reads it; the RMS norm writes q8_1 directly;
  and the end of each Gemma 4 layer is fused. Together they cut the kernels per token from about 1,258 to 1,022. They
  were measured in the same containers and rounds as the earlier revision's tables (in
  [Other hosts and the earlier revision](#other-hosts-and-the-earlier-revision)).
- **A faster Hopper mat-vec for Q4_K**, including the expert weights. It took the H100 from 213.7 to 225.0 tok/s
  (1.05x) with the Q4_K_M file at depth 512, in a separate run. It does not apply to the QAT file, which is Q4_0.

Neither patch is part of this repository. Tokens per second, median of the earlier revision's 3 rounds:

| GPU | GGUF | Depth | Stock | Stock + fusions | Megakernel | Megakernel vs stock + fusions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H100 | QAT Q4_0 | 512 | 229.9 | 240.4 | 376.6 | +56.6% |
| H100 | QAT Q4_0 | 32768 | 211.5 | 210.8 | 308.9 | +46.5% |
| H100 | Q4_K_M | 512 | 215.8 | 224.1 | 342.1 | +52.6% |
| H100 | Q4_K_M | 32768 | 200.0 | 198.8 | 286.8 | +44.3% |
| L40S | QAT Q4_0 | 512 | 171.7 | 179.3 | 216.2 | +20.6% |
| L40S | QAT Q4_0 | 32768 | 141.4 | 146.2 | 172.5 | +18.0% |
| L40S | Q4_K_M | 512 | 154.1 | 160.1 | 184.0 | +14.9% |
| L40S | Q4_K_M | 32768 | 129.1 | 133.1 | 151.2 | +13.6% |

The fusions add 3-5%. The exception is the H100 at long contexts: there attention takes most of the time, and they
add nothing. The megakernel stays well ahead of them at every depth. On Hopper, part of the megakernel's gain comes
from repacked copies of the Q4_0 / Q8_0 / Q5_0 weights, worth 1.8-2.7% of its speed
([Repacked weights on Hopper](#repacked-weights-on-hopper)).

## Correctness

The megakernel cannot give bitwise identical output to stock llama.cpp. It adds numbers up in a different order,
and in two places it deliberately uses more precise arithmetic. So "correct" here means three things, each checked
separately:

1. Its output is as close to stock's as stock's own output is to itself under harmless changes.
2. Every step inside the model matches stock's. The exceptions are the places where it is deliberately different,
   and rare rounding differences that are explained below.
3. Where it is deliberately different, it is closer to the mathematically exact answer than stock is.

The op-by-op check ([part 2](#2-op-by-op-every-step-inside-the-model)) is the main evidence. It caught every
planted error at or above its limit, at exactly the right operation and layer. The end-to-end check
([part 1](#1-end-to-end-the-final-output)) is a sanity check for large errors only. Stock's own output varies enough
that errors of 0.1-1% planted inside a single operation all passed it.

### 1. End to end: the final output

**How stock varies on its own.** Stock llama.cpp's output already changes slightly when you change things that
should not matter: turning its kernel fusions off, turning flash attention off, or running on the other GPU. The
largest of those changes is the noise floor. A candidate passes if its difference from stock is at most twice that
floor.

**What was compared.** Fixed prompts were run through both builds: 9 chat prompts for the instruction-tuned files,
and Wikipedia text for the base model. Both builds were fed the same tokens at every step ("teacher forced"), so one
early difference cannot snowball. At every position, the full next-token probabilities were compared on four
measures:

- KL divergence between the two probability distributions;
- how often both pick the same top token;
- "confident flips": how often they pick different tokens although stock was more than 50% sure;
- the RMS difference of the raw logits.

**Result.** PASS on all three test files (IT Q4_K_M, base Q4_K_M, QAT Q4_0) on both GPUs. A separate long-prompt
set FAILS its policy by one token, below.

**FAIL by one token: the long-prompt set** (`it-long`: one 2,477-token prompt, IT file, run on the L40S only).
Averaged over all positions, the megakernel is within stock's noise floor: a mean KL divergence of 0.82, against
stock's own 0.80. But the model generates 97 tokens after the prompt, and 1 of them differs from stock's choice. That
fails two of the set's limits:

- confident flips: 1.03%, against a maximum of 0.5%;
- top-1 agreement: 98.97%, against a minimum of 99.0%.

Every other measure passes. With stock's mat-vec arithmetic, the megakernel differs on 2 tokens.

Why one token is enough to fail: the limits come from two stock variations (fusions off, flash attention off), and
neither changed any of the 97 tokens. The policy has a built-in minimum of 0.5% confident flips, meant to leave room
when the floor is zero. But 0.5% of 97 tokens is less than one token, so a single differing token fails the set. The
limits were fixed before any megakernel result was judged, and they are not changed after the fact, so the verdict
stays FAIL.

The cause of the differing token was not identified. Candidates include stock's float16 attention, which at these
lengths is about 1,000 times further from the exact answer than the megakernel's
([part 3](#3-against-the-exact-answer)), and a difference in expert routing at some layer; neither was isolated.

### 2. Op by op: every step inside the model

End-to-end checks can miss a small bug, so each operation was also checked on its own. Stock's intermediate results
(Q, K, V, attention output, expert routing, FFN and MoE outputs, layer output) were recorded at every position and
layer. Each was fed into the megakernel, and the megakernel's next step was compared with stock's. The limit is a
relative error of 1e-4 (0.01%) per operation and layer. Stock compared with itself under harmless changes differs by
at most 6e-8.

For this check, the megakernel was switched to stock's mat-vec arithmetic (`GGML_CUDA_MEGAKERNEL_ARITH=stock`), so
that its deliberate improvements do not show up as differences. The switch reproduces stock's arithmetic closely, but
not bit for bit: the way it quantizes activations to 8 bits differs from stock's in small details (exception 2).

| Check | Result |
| --- | --- |
| All operations | within 1e-4, with two known exceptions below |
| Exception 1: attention output, the 5 full-attention layers | 4e-4 to 2e-3. Expected: the megakernel computes attention in float32, stock in float16 (see [part 3](#3-against-the-exact-answer)) |
| Exception 2: QAT file only, layer 0's Q / K / V, the dense FFN at layer 24 and the MoE output of 17 layers | 1.0-1.75e-4, just over the limit: rare rounding differences, explained below the table |
| Expert routing | the same 8 experts as stock in every layer |
| Does the check catch bugs? | Errors of 0.1-1% were planted in six different operations. The end-to-end check missed all six; this check caught each one at exactly the right operation and layer |
| Does it catch errors near the limit? | Errors of 0.5, 1, 2 and 3 times the limit were planted in three operations: a layer output, a K projection and a MoE output (IT file, L40S). Every error at or above the limit was caught, at exactly its operation and layer. At half the limit, the layer output and the K projection passed, measured at the planted size (5.0e-5, 5.2e-5): without a fault they match stock to 2e-8 and 2.5e-6. The MoE output was flagged even at half the limit. With stock's arithmetic, it already differs from stock by 7.4e-5 there, and the limit applies to the total difference |

**Exception 2 in detail.** The numbers in the table are averages over positions. At the median position, the
megakernel matches stock to 1-2e-7 in every one of these operations. The average comes from a minority of positions:
for the FFN and MoE, 5% of positions carry 90-98% of it. At those positions, one rounding decision goes the other way
when the activation is quantized to 8 bits before a Q4_0 mat-vec.

The megakernel's quantizer differs from stock's in three details:

- it multiplies by 1/d, where stock divides by d;
- it rounds ties to even, where stock rounds them away from zero;
- it adds each block's 32 values in a different order before rounding that sum to float16.

In a float64 emulation on stock's own inputs, each of these produces differences of the same size (up to 5e-4 at a
position). At layer 0, Q, K and V share one quantized input that depends only on the token, so the same 30% of
positions differ in all three. Operations within the limit show the same pattern, with fewer or smaller jumps.

### 3. Against the exact answer

In two places the megakernel is different from stock on purpose. Its Q4_0 / Q5_0 mat-vecs add integers before
converting to floating point, which loses less precision. And its attention uses float32 where stock uses float16.
To show that these differences are improvements, each affected block was recomputed from the same inputs in float64,
which is effectively exact. The table gives each one's distance from that exact answer (relative error; smaller is
better).

| Block | Megakernel | Stock |
| --- | --- | --- |
| QAT Q4_0, dense FFN (layers 5, 10, 11, 20, 29) | 0.59-0.93% | 0.78-2.8% |
| QAT Q4_0, MoE | 0.74-1.2% | 1.7-4.5% |
| QAT Q4_0, whole attention branch | 0.67-1.1% | 1.4-2.9% |
| Q4_K_M, MoE | 0.71-1.3% | 0.84-3.6% |
| Attention core alone, positions 1,200-2,470 (normal path) | 0.000013-0.000023% | 0.02-0.055% |
| Attention core alone, tensor-core path (used at long contexts) | 0.00008-0.00012% | 0.02-0.055% |

The megakernel is as close or closer in every row. On Q4_K_M, the dense FFN and the attention branch use the same
arithmetic as stock, and they are equally close. The one exception is the dense FFN at layer 10, where the megakernel
is closer (0.84% against 2.3%).

At long contexts, stock's float16 attention is about 1,000 times further from exact than the megakernel's.

### Other checks

- **Repeatable.** Two runs give bitwise identical output on every file and GPU.
- **Memory and synchronisation.** compute-sanitizer memcheck, racecheck and synccheck report no errors or hazards on
  both GPUs.
- **Off by default.** Without `GGML_CUDA_MEGAKERNEL`, the patched build's output is bitwise identical to stock's
  (checked on IT, base and QAT).

[`correctness/`](correctness/README.md) contains the dumper, the comparison script, the pass limits and the chat
prompts, with the commands to rerun the end-to-end and op-by-op checks. The float64 recomputation scripts and the
reference dumps are not included.
