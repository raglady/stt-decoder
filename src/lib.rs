use ndarray::{Array2, s};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use stt_pncc::{PNCCArgs, deltas, pncc};

use crate::types::Float;

pub static COMMON_STATES_COUNTS: usize = 3;
pub static FOANA_STATES_COUNTS: usize = 5;
pub static FN_STATES_COUNTS: usize = 1;

pub mod functions;
pub mod gpu;
pub mod hmm_gmm;
pub mod monophone;
pub mod phone_state_path;
pub mod real_time;
pub mod settings;
pub mod traits;
pub mod types;

pub type Signal = Vec<Float>;

/// Convert stereo to mono by averaging channels
pub fn to_mono(signal: &[Float], channels: u16) -> Vec<Float> {
    if channels == 1 {
        return signal.to_vec();
    }

    let mono_len = signal.len() / channels as usize;
    let mut mono = vec![0.0; mono_len];

    mono.par_iter_mut()
        .with_min_len(rayon::current_num_threads())
        .enumerate()
        .for_each(|(i, item)| {
            let sum = (0usize..(channels as usize))
                .into_par_iter()
                .with_min_len(rayon::current_num_threads())
                .fold(|| 0.0, |sum, ch| sum + signal[i * channels as usize + ch])
                .reduce(|| 0.0, |global_sum, sum| global_sum + sum);

            *item = sum / channels as Float;
        });

    mono
}

pub fn get_pncc_features_with_delta(signal: &[Float]) -> Array2<Float> {
    let pnccs = pncc(&PNCCArgs::new(signal));
    let delta = deltas(&pnccs, 5);

    let delta_delta = deltas(&delta, 5);
    let (n, d) = pnccs.dim();
    let mut out = Array2::zeros((n, 3 * d));

    out.slice_mut(s![.., ..d]).assign(&pnccs);
    out.slice_mut(s![.., d..2 * d]).assign(&delta);
    out.slice_mut(s![.., 2 * d..]).assign(&delta_delta);
    out
}

#[inline]
pub fn log_sum_exp(vals: &[Float]) -> Float {
    if vals.is_empty() {
        return Float::NEG_INFINITY;
    }

    // 1. Trouver le maximum pour la stabilité numérique
    let max_val = vals.iter().cloned().fold(Float::NEG_INFINITY, Float::max);

    // Si le max est -inf, la somme est -inf (évite 0.0 * exp)
    if max_val == Float::NEG_INFINITY {
        return Float::NEG_INFINITY;
    }

    // 2. Appliquer la formule : max + ln(sum(exp(x - max)))
    let sum_exp: Float = vals.iter().map(|&x| (x - max_val).exp()).sum();

    max_val + sum_exp.ln()
}
