// stt/gmm_kernels.cuh
#include "gmm_functions.cuh"

#include <cmath>


    // =============================================================================
    // compute_log_gmm
    //
    // WGSL : @workgroup_size(STATE_MIXTURE_COUNT, STATE_COUNT)
    //        @builtin(local_invocation_id)
    //
    // CUDA : dim3 block(STATE_MIXTURE_COUNT, STATE_COUNT)
    //        dim3 grid(1, 1)               ← un seul workgroup
    // =============================================================================
    extern "C"
    __global__
    void compute_log_gmm(
        const double* __restrict__ gmm_mean,
        const double* __restrict__ gmm_covar,
        const double* __restrict__ gmm_weight,
        const double* __restrict__ observations,
        const unsigned int gaussian_dim,
        const unsigned int state_mixture_count,
        const unsigned int state_count,
        double* tmp_result,
        double*                    result,
        unsigned int                  t_max)        // = n_obs (arrayLength / GAUSSIAN_DIM)
    {

        // --- local_invocation_id → threadIdx ---
        const unsigned int gaussian = threadIdx.x;  // id.x  ∈ [0, STATE_MIXTURE_COUNT)
        const unsigned int state    = threadIdx.y;  // id.y  ∈ [0, STATE_COUNT)
        if (gaussian >= state_mixture_count || state >= state_count) {
            return;
        }

        // ------------------------------------------------------------------
        // Phase 1 : chaque thread calcule log_gaussian pour son (state, gaussian)
        //           sur tous les instants t
        // ------------------------------------------------------------------
        for (unsigned int t = 0u; t < t_max; t++) {
            tmp_result[gaussian * t_max
                    + state * state_mixture_count * t_max
                    + t] =
                Stt::Functions::log_gaussian_punded_distr(gaussian_dim,
                                                        state_mixture_count,
                    state, gaussian, t,
                    gmm_mean, gmm_covar, gmm_weight, observations);
        }

        // storageBarrier() → __syncthreads()
        __syncthreads();

        // ------------------------------------------------------------------
        // Phase 2 : réduction log-sum-exp par état
        //           uniquement le thread gaussian == 0 de chaque ligne (state)
        // ------------------------------------------------------------------
        if (gaussian == 0u) {
            for (unsigned int t = 0u; t < t_max; t++) {

                // --- Trouver le max sur les gaussiennes (stabilité numérique) ---
                double max_t = -INFINITY;
                for (unsigned int g = 0u; g < state_mixture_count; g++) {
                    double val = tmp_result[g * t_max
                                        + state * state_mixture_count * t_max
                                        + t];
                    if (val > max_t) max_t = val;
                }

                // --- Somme des exp décalées ---
                double sum_exp = 0.0;
                for (unsigned int g = 0u; g < state_mixture_count; g++) {
                    sum_exp += exp(tmp_result[g * t_max
                                            + state * state_mixture_count * t_max
                                            + t] - max_t);
                }

                // log-sum-exp
                result[state * t_max + t] = max_t + log(sum_exp);
            }
        }
    }

    // =============================================================================
    // compute_log_emissions
    //
    // WGSL : @workgroup_size(STATE_MIXTURE_COUNT)
    //        @builtin(global_invocation_id)
    //
    // CUDA : dim3 block(STATE_MIXTURE_COUNT)
    //        dim3 grid(1)                  ← un seul workgroup
    //                                         (la réduction suppose 1 bloc)
    // =============================================================================
    extern "C"
    __global__
    void compute_log_emissions(
        const double*    __restrict__ gmm_mean,
        const double*    __restrict__ gmm_covar,
        const double*    __restrict__ gmm_weight,
        const double*    __restrict__ observations,
        const unsigned int gaussian_dim,
        const unsigned int state_mixture_count,
        double* tmp_result,
        double*                       result,
        const unsigned int state_id,
        const unsigned int                     t_max)
    {

        // global_invocation_id.x = blockIdx.x * blockDim.x + threadIdx.x
        const unsigned int gaussian = blockIdx.x * blockDim.x + threadIdx.x;

        const unsigned int state = state_id;
        if (gaussian >= state_mixture_count ) {
            return;
        }
        // ------------------------------------------------------------------
        // Phase 1 : log_gaussian pour l'état courant (state_id[0])
        // ------------------------------------------------------------------
        for (unsigned int t = 0u; t < t_max; t++) {
            tmp_result[gaussian * t_max + t] =
                Stt::Functions::log_gaussian_punded_distr(gaussian_dim,
                                                        state_mixture_count,
                    state, gaussian, t,
                    gmm_mean, gmm_covar, gmm_weight, observations);
        }

        // storageBarrier() → __syncthreads()
        __syncthreads();

        // ------------------------------------------------------------------
        // Phase 2 : log-sum-exp (thread gaussian == 0 uniquement)
        // ------------------------------------------------------------------

        if (gaussian == 0u) {
            for (unsigned int t = 0u; t < t_max; t++) {

                // --- Trouver le max sur les gaussiennes (stabilité numérique) ---
                double max_t = -INFINITY;
                for (unsigned int g = 0u; g < state_mixture_count; g++) {
                    double val = tmp_result[g * t_max + t];
                    if (val > max_t) max_t = val;
                }

                // --- Somme des exp décalées ---
                double sum_exp = 0.0;
                for (unsigned int g = 0u; g < state_mixture_count; g++) {
                    sum_exp += exp(tmp_result[g * t_max + t] - max_t);
                }

                // log-sum-exp
                result[t] = max_t + log(sum_exp);
            }
        }
    }
