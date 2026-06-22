use crate::hmm_gmm::HMMGMM;

pub type Semone = HMMGMM;

cfg_select! {
    feature = "f64" => {
        pub type Float = f64;
    }
    _ => {
        pub type Float = f32;
    }
}
