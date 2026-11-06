# esp-slm

Ternary small language models on the ESP32-P4. The weights do not fit in RAM, so every pass
streams them from flash and the SD card. The PC side emulates the tiers, checks the
engines bit for bit and estimates P4 time with a twin of the board.

| Path | Contents |
|---|---|
| `src/q35` | Qwen3.5 hybrid engine for Ternary Bonsai 2 GGUF (PTQ1_0) |
| `src/cli` | `q35` CLI: run, spec, treetest, cmpstate |
| `src/tern` | BitNet b1.58 engine (`.tbin`), emulated PSRAM/flash/SD store, multi-lane passes |
| `src/common` | GGUF reader |
| `tools` | P4 twin, store simulator, benches, converters, NumPy references |
| `tests` | tiny models and checks run by `make test` |

```
make
make test
```
