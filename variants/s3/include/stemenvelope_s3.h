/*
 * stemenvelope_s3 — the per-stem envelope executor, C99, no allocation.
 *
 * This is the part that runs on an ESP32-S3 (or any Cortex-M): the band-domain
 * gain envelope only.  It is a line-by-line port of
 *   src/stemenvelope/envelope.py (presence + slew + bounds)
 * and is checked against the Python reference by tests/parity.c.
 *
 * What is NOT here, deliberately:
 *   - the separator (does not fit, see variants/s3/README.md)
 *   - the STFT (the firmware already has a filterbank; feed band levels in)
 *   - the decision layer / recipe (cheap, but product-specific; run it where
 *     the ambient evidence already is)
 *
 * Contract: caller owns `gains`, sized n_stems*n_bands floats, and calls
 * se_step() once per control tick with the current per-(stem,band) levels and
 * the current curve.  No malloc, no float64, no libm beyond expf/fabsf.
 */
#ifndef STEMENVELOPE_S3_H
#define STEMENVELOPE_S3_H

#ifdef __cplusplus
extern "C" {
#endif

#define SE_OK 0
#define SE_ERR_ARG -1

typedef struct {
    int n_stems;
    int n_bands;
    float tau_rise_s;        /* toward a larger (less cutting) gain */
    float tau_fall_s;        /* toward a smaller (more cutting) gain */
    float presence_window_db;/* how deep into a stem's own spectrum to reach  */
    float silence_floor_db;  /* stem peak below this == absent               */
    float max_boost_db;      /* >= 0                                         */
    float max_cut_db;        /* <= 0                                         */
    float *gains;            /* caller-owned, [n_stems * n_bands], row-major  */
} se_envelope_t;

/* Fill defaults (0.65/0.85 s, 24 dB window, -80 dB floor, +3 / -12 dB).
 * `gains` must point at n_stems*n_bands floats; the struct does not own it. */
int se_init(se_envelope_t *env, int n_stems, int n_bands, float *gains);

/* Zero the envelope without re-binding memory. */
void se_reset(se_envelope_t *env);

/* One control tick.
 *   stem_band_db : [n_stems * n_bands] current level of each stem per band (dB)
 *   curve_db     : [n_bands]           current shared curve (dB)
 *   dt_s         : tick period in seconds
 *   active       : 0 = no valid evidence, decay to 0 dB (fail-open)
 * Returns SE_OK, or SE_ERR_ARG on a null/invalid argument.
 * After the call, env->gains holds the applied gains in dB. */
int se_step(se_envelope_t *env, const float *stem_band_db, const float *curve_db,
            float dt_s, int active);

/* Non-finite inputs are treated as -240 dB, matching the Python reference. */
static inline int se_isfinite(float v)
{
    return (v == v) && (v > -1e30f) && (v < 1e30f);
}

#ifdef __cplusplus
}
#endif

#endif /* STEMENVELOPE_S3_H */
