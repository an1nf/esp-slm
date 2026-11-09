// PIE kernels for the ESP32-P4 HP core.

    .text
    .option norvc

// int32_t hx_dot_s8(const int8_t *a, const int8_t *b, int k)
// k multiple of 16; a is read 16 bytes past the end.
    .global hx_dot_s8
    .type   hx_dot_s8, @function
    .balign 4
hx_dot_s8:
    esp.zero.xacc
    srli    a3, a2, 4
    beqz    a3, .Ldot_out
    esp.vld.128.ip  q0, a0, 16
    esp.lp.setup    0, a3, .Ldot_end
    esp.vld.128.ip  q1, a1, 16
.Ldot_end:
    esp.vmulas.s8.xacc.ld.ip    q0, a0, 16, q0, q1
.Ldot_out:
    li      a4, 0
    esp.srs.s.xacc  a0, a4
    ret
    .size   hx_dot_s8, .-hx_dot_s8

// void hx_gemv16_s8(const int8_t *w, const int8_t *xl, int k, int32_t *y16)
// y16[n] = sum_k w[k] * xl[16 k + n]: weights as scalars, 16 lanes per vector.
// k multiple of 16; xl is read 32 bytes past the end; y16 16-byte aligned.
    .global hx_gemv16_s8
    .type   hx_gemv16_s8, @function
    .balign 4
hx_gemv16_s8:
    esp.zero.qacc
    srli    a4, a2, 4
    beqz    a4, .Lg16_out
    esp.vld.128.ip  q4, a1, 16
    esp.vld.128.ip  q5, a1, 16
    esp.lp.setup    0, a4, .Lg16_end
    esp.vld.128.ip  q0, a0, 16
    esp.vsmulas.s8.qacc.ld.incp q4, a1, q4, q0, 0
    esp.vsmulas.s8.qacc.ld.incp q5, a1, q5, q0, 1
    esp.vsmulas.s8.qacc.ld.incp q4, a1, q4, q0, 2
    esp.vsmulas.s8.qacc.ld.incp q5, a1, q5, q0, 3
    esp.vsmulas.s8.qacc.ld.incp q4, a1, q4, q0, 4
    esp.vsmulas.s8.qacc.ld.incp q5, a1, q5, q0, 5
    esp.vsmulas.s8.qacc.ld.incp q4, a1, q4, q0, 6
    esp.vsmulas.s8.qacc.ld.incp q5, a1, q5, q0, 7
    esp.vsmulas.s8.qacc.ld.incp q4, a1, q4, q0, 8
    esp.vsmulas.s8.qacc.ld.incp q5, a1, q5, q0, 9
    esp.vsmulas.s8.qacc.ld.incp q4, a1, q4, q0, 10
    esp.vsmulas.s8.qacc.ld.incp q5, a1, q5, q0, 11
    esp.vsmulas.s8.qacc.ld.incp q4, a1, q4, q0, 12
    esp.vsmulas.s8.qacc.ld.incp q5, a1, q5, q0, 13
    esp.vsmulas.s8.qacc.ld.incp q4, a1, q4, q0, 14
.Lg16_end:
    esp.vsmulas.s8.qacc.ld.incp q5, a1, q5, q0, 15
.Lg16_out:
    esp.st.qacc.l.l.128.ip  a3, 16
    esp.st.qacc.l.h.128.ip  a3, 16
    esp.st.qacc.h.l.128.ip  a3, 16
    esp.st.qacc.h.h.128.ip  a3, 16
    ret
    .size   hx_gemv16_s8, .-hx_gemv16_s8
