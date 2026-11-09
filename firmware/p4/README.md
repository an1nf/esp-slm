# P4 day-1 benchmarks

ESP-IDF (>= 5.5) project for the Waveshare ESP32-P4-NANO (ESP32-P4NRW32X, rev v3).
It measures what the twin (`tools/p4_twin.py`) assumes:

- PSRAM to SRAM copy rate;
- PTQ1_0 decode rate (C, base-3 LUT);
- int8 GEMV with PIE: N = 1 (XACC dot), N = 16 and 32 (QACC, lane-major activations),
  checked against a C reference; target >= 4 GMAC/s at N = 16;
- LUT (per-lane table, 5 trits per byte) vs MAC, with the alpha + kappa N fit;
- SD raw reads on slot 0 (LDO channel 4) at 20/40 MHz and UHS-I 100/200 MHz,
  fitted as bandwidth + per-read latency;
- lanes per pass N for 2B and 4B from the measured rates.

```
idf.py set-target esp32p4
idf.py build flash monitor
```

Not yet run on hardware. The SD test reads raw sectors from 32 MiB on; it does not write.
