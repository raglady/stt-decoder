use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::types::Float;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bigram {
    phonemes: Vec<String>,
    log_prob: Array2<Float>,
}

impl Bigram {
    pub fn new(phonemes: &[String], log_prob: &Array2<Float>) -> Self {
        Self {
            phonemes: phonemes
                .iter()
                .map(|p| {
                    if p == " " {
                        "foana".to_string()
                    } else {
                        p.clone()
                    }
                })
                .collect(),
            log_prob: log_prob.clone(),
        }
    }

    pub fn get_phonemes(&self) -> &[String] {
        &self.phonemes
    }

    pub fn get_log_prob(&self) -> &Array2<Float> {
        &self.log_prob
    }

    pub fn get_log_prob_bigram(&self, phone: &str, next_phone: &str) -> Float {
        let mut log_prob = Float::NEG_INFINITY;
        if let Some(phone_index) = self.phonemes.iter().position(|p| p == phone)
            && let Some(next_phone_index) = self.phonemes.iter().position(|p| p == next_phone)
        {
            log_prob = self.log_prob[[phone_index, next_phone_index]];
        }
        log_prob
    }
}
