# esp-slm

Ternary small language models on the ESP32-P4: 32 MB PSRAM, an SD card and a 400 MHz RISC-V
core with PIE SIMD. The weights do not fit in RAM, so every pass streams them from the
card. A pass that reads the weights once can verify many tokens: up to
`k* = beta F / B` lanes cost no extra time. The work here is to fill those lanes with
good drafts that need no extra weights, and to keep the recurrent state small and exact
enough to verify them.

The PC is the lab. It runs the same model, checks every shortcut bit for bit and
estimates P4 time with a twin of the board.

## Layout

| Path | Contents |
|---|---|
| `src/q35` | Qwen3.5 hybrid engine (Gated DeltaNet + gated attention) for Ternary Bonsai 2 GGUF (PTQ1_0, 1.75 bpw) |
| `src/cli` | `q35` CLI: run, spec, treetest, cmpstate, qtest, dump |
| `src/tern` | BitNet b1.58 engine (`.tbin`) with an emulated PSRAM/flash/SD store and multi-lane passes |
| `src/common` | GGUF reader |
| `src/bench` | `gemvbench`: decode + MAC vs per-lane LUT ternary GEMV |
| `tools` | P4 twin, store simulator, benches, converters, NumPy references |
| `tests` | synthetic models and checks run by `make test` |
| `firmware/p4` | ESP-IDF day-1 benchmarks for the board |

## Build

```
make            # build/q35, build/tern, build/gemvbench
make test       # needs python3 and numpy
```

`OMP=0` builds without OpenMP. `tools/llama_logits.cpp` (built against llama.cpp)
tokenizes prompts and dumps reference logits.

## What is in the engine

- Chain and tree verification in one pass, exact against token-by-token decoding
  (`--check`), with deferred GDN state commit.
- Weight-free drafts: context n-grams, n-gram pool, Jacobi, adjacency (Token Recycling),
  suffix-array datastore (`--ds`, built with `tools/datastore.py`).
- Position-keyed sampling: the same text for any draft schedule.
- Read-only GDN pass (`--state ro8`): int8 base state plus a rank-R delta, materialized
  every R tokens.
- State quantization: Weyl dither (`--dither`), key-column smoothing (`--smooth`),
  int16 activations on the int8 state (`--x16`), 16-bit heads chosen by decay gain (`--hi`).
- Low-bit KV (`--kv-bits 2|4`) with exact rows for the top-R scores (`--kv-refine`) and a
  recent window (`--kv-recent`).

## Examples

```
./build/q35 info model.gguf
./build/q35 spec model.gguf --ids 1,2,3 --gen 128 --tree 16 --gamma 16 --adj 1 --check
./build/q35 cmpstate model.gguf --cands int8,int8+d+s,ro8+d+s+x16:4:2,f32+k2+r32 --ids ...
./build/q35 qtest -
./build/gemvbench
python3 tools/p4_twin.py lanes --model 4B --ctx 1024 --kv-bits 2 --kv-refine 16
python3 tools/q35_tree_bench.py --model model.gguf --llama ./llama_logits --adj --sampling
```

`cmpstate` candidates are `fmt[+mods][:rank[:chunk]]`, with mods `d` (dither),
`s` (smoothing), `x16`, `hNN` (NN % heads in 16 bits), `kB` (KV bits), `rN` (exact rows),
`wN` (recent window).

## Twin estimates

With tau measured on the PC (tree + adjacency) and assumed P4 rates (marked `[H]` in
`tools/p4_twin.py`), at SD 40 MB/s: 2B about 6 s/token, 4B about 13 s/token
(ctx 512, or ctx 1024 with 2-bit KV). The board numbers come from `firmware/p4`.
