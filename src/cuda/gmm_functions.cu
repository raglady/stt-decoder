#include <cmath>

namespace Stt {
    namespace Functions {
        __device__ __forceinline__
        unsigned int compute_index(unsigned int gaussian_dim, unsigned int state_mixture_count, unsigned int state_index, unsigned int gaussian_index, unsigned int index)
        {
            return gaussian_index * gaussian_dim
                 + state_index    * gaussian_dim * state_mixture_count
                 + index;
        }

        __device__ __forceinline__ double compute_gmm_ln_det_covar(unsigned int gaussian_dim, unsigned int state_mixture_count, unsigned int state_index, unsigned int gaussian_index,const double* __restrict__ gmm_covar) {

            double sum = 0.0;

                for (unsigned int i = 0u; i < gaussian_dim; i++) {
                    int index = compute_index(gaussian_dim, state_mixture_count,
                                              state_index, gaussian_index, i);
                    sum += log(gmm_covar[index]);
                }
            return sum;
        }

        __device__ __forceinline__
        double log_gaussian_punded_distr(unsigned int gaussian_dim, unsigned int state_mixture_count,
            unsigned int state_index,
            unsigned int gaussian_index,
            unsigned int t,
            const double* __restrict__ gmm_mean,
            const double* __restrict__ gmm_covar,
            const double* __restrict__ gmm_weight,
            const double* __restrict__ observations)
        {
            double sum = 0.0;

            for (unsigned int i = 0u; i < gaussian_dim; i++) {
                unsigned int idx = compute_index(gaussian_dim, state_mixture_count,
                                   state_index, gaussian_index, i);
                double centered = observations[t * gaussian_dim + i] - gmm_mean[idx];
                sum += (centered * centered) / gmm_covar[idx];
            }

            return log(gmm_weight[state_index * state_mixture_count + gaussian_index])
                 - 0.5 * (double)gaussian_dim * log(2.0 * M_PI)
                 - 0.5 * compute_gmm_ln_det_covar(gaussian_dim, state_mixture_count,
                              state_index, gaussian_index, gmm_covar)
                 - 0.5 * sum;
        }
    }
}
