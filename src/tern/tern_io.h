/* Prefetch rings: one reader thread per tier. */
#ifndef TERN_IO_H
#define TERN_IO_H

#include "tern.h"

typedef struct TernIO TernIO;

TernIO *tern_io_start(const TernModel *m, TernState *s0, const char *path,
                      const double bw[TERN_NTIERS], const double lat[TERN_NTIERS],
                      const size_t ring_bytes[TERN_NTIERS], char *err, size_t errlen);
void tern_io_stop(TernIO *io);
void tern_io_stats(const TernIO *io, double wait_s[TERN_NTIERS], int *mismatch);
void tern_io_reset_stats(TernIO *io);
int  tern_io_slots(const TernIO *io, int tier);

#endif
