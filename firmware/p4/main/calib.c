#include <math.h>
#include <stdio.h>

#include "hx.h"

/* Lanes per pass: minimize max(SD I/O, compute) / tau(N), with tau measured on the PC (tree + adjacency). */
typedef struct {
    const char *name;
    double sd_mb, gmac_lane, tau16, tau32;
} model_t;

static double tau_at(const model_t *md, int n)
{
    const double l1 = 0.0, l16 = 2.7725887, l32 = 3.4657359, x = log((double)n);
    if (n <= 16)
        return 1.0 + (md->tau16 - 1.0) * (x - l1) / (l16 - l1);
    return md->tau16 + (md->tau32 - md->tau16) * (x - l16) / (l32 - l16);
}

void calibrate(const hx_meas_t *m)
{
    static const model_t MD[2] = {
        {"2B ctx 1024", 398.6, 1.881, 1.98, 2.21},
        {"4B ctx 512", 920.0, 4.205, 1.89, 2.28},
    };
    static const int NS[] = {1, 8, 12, 16, 24, 32};
    const double r = m->gmacs_n16 > 0 ? m->gmacs_n16 : 6.4;
    double sd = 16.0, lat = 0.3;
    for (int i = 0; i < m->sd_n; i++)
        if (m->sd_mbps[i] > sd) {
            sd = m->sd_mbps[i];
            lat = m->sd_lat_ms[i];
        }
    printf("calibration: SD %.1f MB/s + %.3f ms/read, lanes %.2f GMAC/s\n", sd, lat, r);
    for (int k = 0; k < 2; k++) {
        const model_t *md = &MD[k];
        const double io = md->sd_mb / sd + md->sd_mb * 1e6 / (256.0 * 1024) * lat * 1e-3;
        int best = 1;
        double bt = 1e30;
        printf("  %s: I/O %.1f s/pass, k* = %.1f lanes;", md->name, io, io * r / md->gmac_lane);
        for (unsigned i = 0; i < sizeof(NS) / sizeof(NS[0]); i++) {
            const double comp = NS[i] * md->gmac_lane / r;
            const double t = (io > comp ? io : comp) / tau_at(md, NS[i]);
            printf(" N=%d %.1f", NS[i], t);
            if (t < bt) {
                bt = t;
                best = NS[i];
            }
        }
        printf(" s/token -> N = %d (%.1f s/token)\n", best, bt);
    }
}
