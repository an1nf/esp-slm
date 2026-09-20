# esp-slm

Ternary small language models on the ESP32-P4. The weights do not fit in RAM, so every pass
streams them from flash and the SD card. The PC side emulates the tiers and checks the
engine bit for bit against a NumPy reference.

| Path | Contents |
|---|---|
| `src/tern` | BitNet b1.58 engine (`.tbin`), emulated PSRAM/flash/SD store, multi-lane passes |
| `tools` | format writer, HF converter, NumPy reference, store simulator, k* sweep |
| `tests` | tiny models and checks run by `make test` |

```
make
make test
```
