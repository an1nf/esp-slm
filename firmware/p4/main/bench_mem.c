#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "hx.h"

void bench_mem(hx_meas_t *m)
{
    enum { CH = 64 * 1024, TOT = 16 * 1024 * 1024 };
    uint8_t *ps = heap_caps_malloc(TOT, MALLOC_CAP_SPIRAM);
    uint8_t *sr = heap_caps_aligned_alloc(64, CH, MALLOC_CAP_INTERNAL);
    uint8_t *sr2 = heap_caps_aligned_alloc(64, CH, MALLOC_CAP_INTERNAL);
    if (!ps || !sr || !sr2) {
        printf("mem: out of memory\n");
        return;
    }
    for (size_t i = 0; i < TOT; i += 4)
        *(uint32_t *)(ps + i) = (uint32_t)i * 2654435761u;

    double t0 = hx_now_s();
    for (size_t off = 0; off < TOT; off += CH)
        memcpy(sr, ps + off, CH);
    double dt = hx_now_s() - t0;
    m->psram_mbps = TOT / dt / 1e6;
    printf("PSRAM -> SRAM memcpy (64 KiB chunks): %.1f MB/s\n", m->psram_mbps);

    uint32_t acc = 0;
    t0 = hx_now_s();
    for (size_t i = 0; i < TOT; i += 16) {
        const uint32_t *p = (const uint32_t *)(ps + i);
        acc += p[0] ^ p[1] ^ p[2] ^ p[3];
    }
    dt = hx_now_s() - t0;
    printf("PSRAM sequential read (cache):        %.1f MB/s (%08x)\n", TOT / dt / 1e6, (unsigned)acc);

    t0 = hx_now_s();
    for (int r = 0; r < 256; r++)
        memcpy(sr2, sr, CH);
    dt = hx_now_s() - t0;
    printf("SRAM -> SRAM memcpy:                  %.1f MB/s\n", 256.0 * CH / dt / 1e6);
    heap_caps_free(ps);
    heap_caps_free(sr);
    heap_caps_free(sr2);
}
