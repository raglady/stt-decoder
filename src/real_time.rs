use std::sync::Arc;

use ndarray::{Array2, s};
use stt_pncc::PNCCArgs;
use tokio::{
    sync::{RwLock, broadcast, watch},
    time::{Duration, sleep},
};

use crate::{
    get_pncc_features_with_delta, settings::Settings, traits::viterbi_trait::ViterbiTrait,
    types::Float,
};

pub async fn real_time_decode(
    model: Arc<dyn ViterbiTrait>,
    rx_signal: watch::Receiver<Vec<Float>>,
    tx_output_phones: broadcast::Sender<String>,
    settings: Arc<Settings>,
) {
    let min_ms_overlap = 75;
    let min_samples = 16 * min_ms_overlap; // le minimum de sample est 500ms. On applique 20 frames pour le chevauchement
    let mut signals: Vec<Float> = Vec::new();
    let mut pnccs = Array2::<Float>::zeros([1, 1]);

    cfg_select! {
        feature = "f64" => {
            let pncc_row_to_ignore: isize = ((min_samples as Float
                - (PNCCArgs::get_win_len() * 16_000f64)
                + PNCCArgs::get_win_hop() * 16000_f64)
                / (PNCCArgs::get_win_hop() * 16000f64))
                .ceil() as isize;
        }
        _ => {
            let pncc_row_to_ignore: isize = ((min_samples as Float
                - (PNCCArgs::get_win_len() * 16_000f32)
                + PNCCArgs::get_win_hop() * 16000_f32)
                / (PNCCArgs::get_win_hop() * 16000f32))
                .ceil() as isize;
        }
    }

    let buffer = Arc::new(RwLock::new(Vec::new()));
    loop {
        let mut rx_signal_clone = rx_signal.clone();
        if rx_signal_clone.changed().await.is_ok() {
            let mut buffered_signal = rx_signal_clone.borrow_and_update().clone();

            signals.append(&mut buffered_signal); // on fait append ici parce qu'on veut garder en reserve l'ancienne taille buffered_signal pour eviter la réallocation lors de l'ajout des données.

            // on ne recupère pas les signals si c'est moins de min_samples, sauf en cas de dernier signal
            if signals.len() < 16 * 600 {
                // Comme il y a très peu de données, on attend
                sleep(Duration::from_millis(min_ms_overlap as u64)).await;
                continue;
            }
        }

        // do pncc
        pnccs = if pnccs.ncols() == 1 {
            get_pncc_features_with_delta(&signals)
        } else {
            get_pncc_features_with_delta(&signals)
                .slice(s![pncc_row_to_ignore.., ..])
                .to_owned()
        };
        // on garde le dernier pour le chevauchement
        signals = signals
            .rchunks(min_samples)
            .next()
            .unwrap_or_default()
            .to_vec();
        pnccs = pnccs.slice(s![..-pncc_row_to_ignore, ..]).to_owned();

        let arc_model = model.clone();
        let output = arc_model
            .clone()
            .viterbi_beam_search(
                buffer.clone(),
                pnccs.to_shared(),
                settings.predict.beam_size,
            )
            .await;

        tx_output_phones.send(output).unwrap();
    }
}
