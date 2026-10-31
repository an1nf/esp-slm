/* KL of state and KV formats against an f32 reference. */
#include "cli.h"

int cmd_cmpstate(const Q35Model *m, const Opts *o, const char *cands)
{
    int *ids = NULL;
    const int n = parse_ids(o->ids, &ids);
    if (n < 2) {
        fprintf(stderr, "cmpstate: --ids required\n");
        return 2;
    }
    const int V = m->c.vocab;
    Q35State R;
    if (q35_state_init(&R, m, n + 8, 16, 0)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    float *ref = (float *)malloc((size_t)n * V * sizeof(float));
    double *lse = (double *)malloc((size_t)n * sizeof(double));
    if (!ref || !lse)
        return 1;
    for (int done = 0; done < n;) {
        const int T = n - done < 16 ? n - done : 16;
        const float *lg = q35_forward(m, &R, ids + done, T, 1);
        if (!lg)
            return 1;
        memcpy(ref + (size_t)done * V, lg, (size_t)T * V * sizeof(float));
        done += T;
    }
    q35_state_free(&R);
    for (int p = 0; p < n; p++) {
        const float *a = ref + (size_t)p * V;
        float mx = a[0];
        for (int j = 1; j < V; j++)
            if (a[j] > mx)
                mx = a[j];
        double sum = 0.0;
        for (int j = 0; j < V; j++)
            sum += exp((double)a[j] - mx);
        lse[p] = mx + log(sum);
    }
    double *kl = (double *)malloc((size_t)n * sizeof(double));
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", cands);
    printf("%d positions; reference: f32 state, f32 KV, chunks of 16\n", n);
    printf("| candidate | argmax equal | KL mean | KL median | KL max | KL mean 2nd half | "
           "materializations |\n|---|---|---|---|---|---|---|\n");
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        char fmt[16] = "";
        int rank = 16, chunk = 16;
        sscanf(tok, "%15[^:]:%d:%d", fmt, &rank, &chunk);
        int sf = -1;
        if (!strcmp(fmt, "f32") || !strcmp(fmt, "f32n")) sf = Q35_STATE_F32;
        else if (!strcmp(fmt, "f16")) sf = Q35_STATE_F16;
        else if (!strcmp(fmt, "bf16")) sf = Q35_STATE_BF16;
        else if (!strcmp(fmt, "int8")) sf = Q35_STATE_I8;
        else if (!strcmp(fmt, "ro32")) sf = Q35_STATE_RO32;
        else if (!strcmp(fmt, "ro8")) sf = Q35_STATE_RO8;
        else if (!strcmp(fmt, "ro16")) sf = Q35_STATE_RO16;
        if (sf < 0 || chunk < 1 || chunk > 64) {
            fprintf(stderr, "unknown candidate: %s\n", tok);
            return 2;
        }
        Q35State C;
        if (q35_state_init(&C, m, n + 8, chunk, 0))
            return 1;
        C.state_fmt = sf;
        C.ro_rmax = rank;
        if (!strcmp(fmt, "f32n"))
            C.noise_eps = (float)pow(10.0, -rank);
        int agree = 0;
        for (int done = 0; done < n;) {
            const int T = n - done < chunk ? n - done : chunk;
            const float *lg = q35_forward(m, &C, ids + done, T, 1);
            if (!lg)
                return 1;
            for (int t = 0; t < T; t++) {
                const int p = done + t;
                const float *a = ref + (size_t)p * V, *b = lg + (size_t)t * V;
                float mb = b[0];
                for (int j = 1; j < V; j++)
                    if (b[j] > mb)
                        mb = b[j];
                double sb = 0.0;
                for (int j = 0; j < V; j++)
                    sb += exp((double)b[j] - mb);
                const double lseb = mb + log(sb);
                double k = 0.0;
                for (int j = 0; j < V; j++) {
                    const double la = (double)a[j] - lse[p];
                    k += exp(la) * (la - ((double)b[j] - lseb));
                }
                kl[p] = k > 0.0 ? k : 0.0;
                agree += argmax(a, V) == argmax(b, V);
            }
            done += T;
        }
        double mean = 0.0, mean2 = 0.0, mx = 0.0;
        for (int p = 0; p < n; p++) {
            mean += kl[p];
            if (p >= n / 2)
                mean2 += kl[p];
            if (kl[p] > mx)
                mx = kl[p];
        }
        double *srt = (double *)malloc((size_t)n * sizeof(double));
        memcpy(srt, kl, (size_t)n * sizeof(double));
        for (int i = 1; i < n; i++)
            for (int j = i; j > 0 && srt[j] < srt[j - 1]; j--) {
                const double x = srt[j];
                srt[j] = srt[j - 1];
                srt[j - 1] = x;
            }
        char name[64];
        if (sf >= Q35_STATE_RO32)
            snprintf(name, sizeof(name), "%s rank %d, %d per pass", fmt, rank, chunk);
        else if (C.noise_eps > 0.0f)
            snprintf(name, sizeof(name), "f32 + noise 1e-%d, %d per pass", rank, chunk);
        else
            snprintf(name, sizeof(name), "%s, %d per pass", fmt, chunk);
        printf("| %s | %d/%d (%.1f %%) | %.2e | %.2e | %.2e | %.2e | %llu |\n", name, agree, n,
               100.0 * agree / n, mean / n, srt[n / 2], mx, mean2 / (n - n / 2),
               (unsigned long long)C.ro_mat);
        fflush(stdout);
        free(srt);
        q35_state_free(&C);
    }
    free(ref); free(lse); free(kl); free(ids);
    return 0;
}
