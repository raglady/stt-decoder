use std::{pin::Pin, sync::Arc};

use ndarray::ArcArray2;
use tokio::sync::RwLock;

use crate::{phone_state_path::PhoneStatePath, types::Float};

pub trait ViterbiTrait: Send + Sync {
    fn viterbi_beam_search(
        self: Arc<Self>,
        set_path_last_score: Arc<RwLock<Vec<(PhoneStatePath, Float)>>>,
        mfcc: ArcArray2<Float>,
        threshold: Float,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'static>>;
}
