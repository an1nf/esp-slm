#include <stdio.h>

#include "esp_chip_info.h"
#include "esp_clk_tree.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hx.h"

void app_main(void)
{
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    uint32_t hz = 0;
    esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU, ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &hz);
    printf("\nesp-slm p4 bench: rev v%d.%d, %d cores, CPU %lu MHz\n", ci.revision / 100,
           ci.revision % 100, ci.cores, (unsigned long)(hz / 1000000));
    hx_meas_t m = {0};
    bench_mem(&m);
    bench_kernels(&m);
    bench_sd(&m);
    printf("done\n");
    while (1)
        vTaskDelay(pdMS_TO_TICKS(1000));
}
