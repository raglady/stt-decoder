#define_import_path stt::gmm_functions

#import stt::gmm_bindings as GMMBindings
#import stt::constants as Constants

fn compute_gmm_ln_det_covar(state_index: u32, gaussian_index: u32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < #{GAUSSIAN_DIM}u; i += 1u) {
        let index = compute_index(state_index, gaussian_index, i);
        sum += log(GMMBindings::gmm_covar[index]);
    }
    return sum;
}

fn compute_index(state_index: u32, gaussian_index: u32, index: u32) -> u32 {
    return gaussian_index * #{GAUSSIAN_DIM}u + state_index * #{GAUSSIAN_DIM}u * #{STATE_MIXTURE_COUNT}u + index;
}

fn log_gaussian_punded_distr(state_index: u32, gaussian_index: u32, t: u32) -> f32 {

    var sum = 0.0f;

    for (var i = 0u; i < #{GAUSSIAN_DIM}u; i += 1u) {
        let index = compute_index(state_index, gaussian_index, i);
        let centered = GMMBindings::observations[t * #{GAUSSIAN_DIM}u + i] - GMMBindings::gmm_mean[index];
        sum += (centered * centered) / GMMBindings::gmm_covar[index];
    }

    return log(GMMBindings::gmm_weight[state_index * #{STATE_MIXTURE_COUNT}u + gaussian_index]) - 0.5f * f32(#{GAUSSIAN_DIM}u) * log(2.0f * Constants::PI)
        - 0.5f * compute_gmm_ln_det_covar(state_index, gaussian_index)
    - 0.5f * sum;
}
