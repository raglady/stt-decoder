use serde::{Deserialize, Serialize};

use crate::types::Float;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Settings {
    pub predict: PredictSettings,
    pub storage: StorageSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictSettings {
    pub predict_dir: String,
    pub beam_size: Float,
    pub real_time: bool,
    pub enable: bool,
}

impl Default for PredictSettings {
    fn default() -> Self {
        Self {
            predict_dir: "predict".to_string(),
            beam_size: 20.0,
            real_time: false,
            enable: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSettings {
    pub monophone_modele_file: String,
    pub phonemes_file: String,
    pub log_prob_bigram_file: String,
}

impl Default for StorageSettings {
    fn default() -> Self {
        Self {
            monophone_modele_file: "monophone.json".to_string(),
            phonemes_file: "phonemes.json".to_string(),
            log_prob_bigram_file: "log-prob-bigram.json".to_string(),
        }
    }
}
