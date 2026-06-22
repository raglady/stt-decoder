namespace Stt {
    namespace Functions {
        __device__ __forceinline__
        unsigned int compute_index(unsigned int gaussian_dim, unsigned int state_mixture_count, unsigned int state_index, unsigned int gaussian_index, unsigned int index)
        ;

        __device__ __forceinline__ double compute_gmm_ln_det_covar(unsigned int gaussian_dim, unsigned int state_mixture_count, unsigned int state_index, unsigned int gaussian_index,const double* __restrict__ gmm_covar) ;

        __device__ __forceinline__
        double log_gaussian_punded_distr(unsigned int gaussian_dim, unsigned int state_mixture_count,
            unsigned int state_index,
            unsigned int gaussian_index,
            unsigned int t,
            const double* __restrict__ gmm_mean,
            const double* __restrict__ gmm_covar,
            const double* __restrict__ gmm_weight,
            const double* __restrict__ observations)
        ;
    }
}
