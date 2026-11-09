# esp-slm

Ternary small language models on the ESP32-P4: 32 MB PSRAM, an SD card and a 400 MHz RISC-V
core with PIE SIMD. The weights do not fit in RAM, so every pass streams them from the
card. A pass that reads the weights once can verify many tokens: up to
`k* = beta F / B` lanes cost no extra time. The work here is to fill those lanes with
good drafts that need no extra weights, and to keep the recurrent state small and exact
enough to verify them.

## Build

```
make            # build/q35, build/tern, build/gemvbench
make test       # needs python3 and numpy
```

`OMP=0` builds without OpenMP. `tools/llama_logits.cpp` (built against llama.cpp)
tokenizes prompts and dumps reference logits.
