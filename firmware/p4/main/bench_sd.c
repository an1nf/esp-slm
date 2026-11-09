#include <stdio.h>
#include <string.h>

#include "driver/sdmmc_host.h"
#include "esp_heap_caps.h"
#include "hx.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "sdmmc_cmd.h"

/* Raw sector reads on slot 0 (GPIO39-44), card powered by on-chip LDO channel 4. */
static int sd_run(int khz, int uhs, sd_pwr_ctrl_handle_t pwr, uint8_t *buf, hx_meas_t *m)
{
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.slot = SDMMC_HOST_SLOT_0;
    host.max_freq_khz = khz;
    host.pwr_ctrl_handle = pwr;
    sdmmc_slot_config_t slot = {
        .cd = SDMMC_SLOT_NO_CD,
        .wp = SDMMC_SLOT_NO_WP,
        .width = 4,
        .flags = 0,
    };
#ifdef SDMMC_SLOT_FLAG_UHS1
    if (uhs)
        slot.flags |= SDMMC_SLOT_FLAG_UHS1;
#else
    if (uhs) {
        printf("  %d kHz: UHS-I not supported by this IDF\n", khz);
        return -1;
    }
#endif
    if (sdmmc_host_init() != ESP_OK)
        return -1;
    sdmmc_card_t card;
    int rc = -1;
    if (sdmmc_host_init_slot(host.slot, &slot) == ESP_OK && sdmmc_card_init(&host, &card) == ESP_OK) {
        static const int CH[3] = {16, 64, 256};
        double t[3];
        const size_t start = 65536, total = 16u << 20;
        if ((size_t)card.csd.capacity < start + 2 * total / 512) {
            printf("  card too small\n");
        } else {
            for (int c = 0; c < 3; c++) {
                const size_t sec = (size_t)CH[c] * 1024 / 512, n = total / (CH[c] * 1024u);
                const double t0 = hx_now_s();
                size_t i;
                for (i = 0; i < n; i++)
                    if (sdmmc_read_sectors(&card, buf, start + i * sec, sec) != ESP_OK)
                        break;
                t[c] = (hx_now_s() - t0) / (double)n;
                if (i < n)
                    t[c] = -1;
            }
            if (t[0] > 0 && t[2] > 0) {
                const double bw = (CH[2] - CH[0]) * 1024.0 / (t[2] - t[0]);
                const double lat = t[0] - CH[0] * 1024.0 / bw;
                printf("  %6d kHz%s: card %d kHz; %.1f / %.1f / %.1f MB/s at 16/64/256 KiB; "
                       "fit %.1f MB/s + %.3f ms per read\n", khz, uhs ? " UHS-I" : "",
                       card.max_freq_khz, CH[0] * 1024 / t[0] / 1e6, CH[1] * 1024 / t[1] / 1e6,
                       CH[2] * 1024 / t[2] / 1e6, bw / 1e6, lat * 1e3);
                if (m->sd_n < 4) {
                    m->sd_khz[m->sd_n] = khz;
                    m->sd_mbps[m->sd_n] = bw / 1e6;
                    m->sd_lat_ms[m->sd_n] = lat * 1e3;
                    m->sd_n++;
                }
                rc = 0;
            } else {
                printf("  %6d kHz%s: read error\n", khz, uhs ? " UHS-I" : "");
            }
        }
    } else {
        printf("  %6d kHz%s: card init failed\n", khz, uhs ? " UHS-I" : "");
    }
    sdmmc_host_deinit();
    return rc;
}

void bench_sd(hx_meas_t *m)
{
    sd_pwr_ctrl_ldo_config_t ldo = {.ldo_chan_id = 4};
    sd_pwr_ctrl_handle_t pwr = NULL;
    if (sd_pwr_ctrl_new_on_chip_ldo(&ldo, &pwr) != ESP_OK) {
        printf("sd: LDO init failed\n");
        return;
    }
    uint8_t *buf = heap_caps_aligned_alloc(64, 256 * 1024, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!buf) {
        printf("sd: out of memory\n");
        return;
    }
    printf("SD raw read, slot 0, 4-bit:\n");
    sd_run(SDMMC_FREQ_DEFAULT, 0, pwr, buf, m);
    sd_run(SDMMC_FREQ_HIGHSPEED, 0, pwr, buf, m);
    sd_run(100000, 1, pwr, buf, m);
    sd_run(200000, 1, pwr, buf, m);
    heap_caps_free(buf);
    sd_pwr_ctrl_del_on_chip_ldo(pwr);
}
