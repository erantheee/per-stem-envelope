#include "stemenvelope_s3.h"

#include <math.h>
#include <stddef.h>

static float se_clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int se_init(se_envelope_t *env, int n_stems, int n_bands, float *gains)
{
    if (env == NULL || gains == NULL || n_stems < 1 || n_bands < 1) return SE_ERR_ARG;
    env->n_stems = n_stems;
    env->n_bands = n_bands;
    env->tau_rise_s = 0.65f;
    env->tau_fall_s = 0.85f;
    env->presence_window_db = 24.0f;
    env->silence_floor_db = -80.0f;
    env->max_boost_db = 3.0f;
    env->max_cut_db = -12.0f;
    env->gains = gains;
    se_reset(env);
    return SE_OK;
}

void se_reset(se_envelope_t *env)
{
    int i, count;
    if (env == NULL || env->gains == NULL) return;
    count = env->n_stems * env->n_bands;
    for (i = 0; i < count; ++i) env->gains[i] = 0.0f;
}

int se_step(se_envelope_t *env, const float *stem_band_db, const float *curve_db,
            float dt_s, int active)
{
    int stem, band, idx;
    float dt = dt_s > 0.0f ? dt_s : 0.0f;

    if (env == NULL || env->gains == NULL) return SE_ERR_ARG;
    if (env->n_stems < 1 || env->n_bands < 1) return SE_ERR_ARG;
    if (active && (stem_band_db == NULL || curve_db == NULL)) return SE_ERR_ARG;

    for (stem = 0; stem < env->n_stems; ++stem) {
        const float *row = stem_band_db + (ptrdiff_t)stem * env->n_bands;
        float peak = -240.0f;
        float floor_db;
        int silent;

        /* peak over the stem's own bands; non-finite counts as numerically zero */
        if (active) {
            for (band = 0; band < env->n_bands; ++band) {
                float v = se_isfinite(row[band]) ? row[band] : -240.0f;
                if (v > peak) peak = v;
            }
        }
        silent = peak < env->silence_floor_db;
        floor_db = peak - env->presence_window_db;

        for (band = 0; band < env->n_bands; ++band) {
            float target = 0.0f;
            float cur;
            float tau, alpha;

            if (active) {
                float level = se_isfinite(row[band]) ? row[band] : -240.0f;
                float presence = silent ? 0.0f
                                        : se_clampf((level - floor_db) / env->presence_window_db,
                                                    0.0f, 1.0f);
                float bounded = se_clampf(curve_db[band], env->max_cut_db, env->max_boost_db);
                target = se_clampf(presence * bounded, env->max_cut_db, env->max_boost_db);
            }

            idx = stem * env->n_bands + band;
            cur = env->gains[idx];
            tau = target > cur ? env->tau_rise_s : env->tau_fall_s;
            if (tau < 1e-6f) tau = 1e-6f;
            alpha = 1.0f - expf(-dt / tau);
            env->gains[idx] = cur + (target - cur) * alpha;
        }
    }
    return SE_OK;
}
