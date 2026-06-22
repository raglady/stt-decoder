#import stt::gmm_bindings as GMMBindings
#import stt::gmm_functions as GMMCore
#import stt::constants as Constants

@compute @workgroup_size(#{STATE_MIXTURE_COUNT}u , #{STATE_COUNT}u)
fn compute_log_gmm(@builtin(local_invocation_id) id: vec3u) {
    let gaussian = id.x;
    let state = id.y;


    let t_max = u32(arrayLength(&GMMBindings::observations) / #{GAUSSIAN_DIM}u);

    for (var t = 0u; t < t_max; t += 1u) {
        GMMBindings::tmp_result[gaussian * t_max + state * #{STATE_MIXTURE_COUNT}u * t_max + t] = GMMCore::log_gaussian_punded_distr(state, gaussian, t);
    }
    storageBarrier();

    let tmp_length =  arrayLength(&GMMBindings::tmp_result);
    if (gaussian == 0) {
        for (var t = 0u; t < t_max; t += 1u) {
            var max_t = Constants::F32_MIN;
            var sum_exp = 0.0f;
            for (var gaussian_index = 0u; gaussian_index <  #{STATE_MIXTURE_COUNT}u; gaussian_index += 1u) {
                if( GMMBindings::tmp_result[gaussian_index * t_max + state * #{STATE_MIXTURE_COUNT}u * t_max + t] > max_t ){
                    max_t = GMMBindings::tmp_result[gaussian_index*t_max + state * #{STATE_MIXTURE_COUNT}u * t_max + t];
                }
            }

            for (var gaussian_index = 0u; gaussian_index <  #{STATE_MIXTURE_COUNT}u; gaussian_index += 1) {
                sum_exp += exp(GMMBindings::tmp_result[gaussian_index*t_max + state * #{STATE_MIXTURE_COUNT}u * t_max + t] - max_t);
            }
            GMMBindings::result[state * t_max + t] = max_t + log(sum_exp);
        }
    }
}


@compute @workgroup_size(#{STATE_MIXTURE_COUNT}u)
fn compute_log_emissions(@builtin(global_invocation_id) id: vec3u) {
    let gaussian = id.x;

    let t_max = u32(arrayLength(&GMMBindings::observations) / #{GAUSSIAN_DIM}u);

    for (var t = 0u; t < t_max; t += 1u) {
        GMMBindings::tmp_result[gaussian * t_max + t] = GMMCore::log_gaussian_punded_distr(GMMBindings::state_id[0], gaussian, t);
    }
    storageBarrier();

    let tmp_length =  arrayLength(&GMMBindings::tmp_result);
    if (gaussian == 0) {
        for (var t = 0u; t < t_max; t += 1u) {
            var max_t = Constants::F32_MIN;
            var sum_exp = 0.0f;
            for (var gaussian_index = 0u; gaussian_index <  #{STATE_MIXTURE_COUNT}u; gaussian_index += 1u) {
                if( GMMBindings::tmp_result[gaussian_index * t_max +  t] > max_t ){
                    max_t = GMMBindings::tmp_result[gaussian_index*t_max +  t];
                }
            }

            for (var gaussian_index = 0u; gaussian_index <  #{STATE_MIXTURE_COUNT}u; gaussian_index += 1) {
                sum_exp += exp(GMMBindings::tmp_result[gaussian_index*t_max + t] - max_t);
            }
            GMMBindings::result[t] = max_t + log(sum_exp);
        }
    }
}
