use std::{pin::Pin, sync::Arc};

use ndarray::{ArcArray2, Array1};
use tokio::sync::RwLock;

use crate::{
    COMMON_STATES_COUNTS, FN_STATES_COUNTS, FOANA_STATES_COUNTS, phone_state_path::PhoneStatePath,
    traits::viterbi_trait::ViterbiTrait, types::Float,
};

use super::MonoPhone;

impl ViterbiTrait for MonoPhone {
    fn viterbi_beam_search(
        self: Arc<Self>,
        set_path_last_score: Arc<RwLock<Vec<(PhoneStatePath, Float)>>>,
        mfcc: ArcArray2<Float>,
        threshold: Float,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'static>> {
        // let centered_reduced = (&mfcc - &self.global_mean) / self.global_var.sqrt();
        // let mfcc = centered_reduced.view();
        let mut output = String::new();

        let obs_len = mfcc.nrows();

        let (map_index_phone, vec_state_logprob) =
            self.compute_log_prob_for_state_for_every_phone(mfcc);

        Box::pin(async move {
            for t in 0..obs_len {
                let mut path_score = Vec::new();

                let set_path_last_score_clone = set_path_last_score.clone();
                let mut set_path_last_score_guard = set_path_last_score_clone.write().await;

                // Compute the next node of all path in set_path_last_score with its score
                set_path_last_score_guard.iter().for_each(|(path, score)| {
                    let ((phone, semone_id), state) = path.back().unwrap();
                    // proceed the staying in the state and going to the next if it is not the end state
                    let hmm = self.map_phone_hmm.get(phone).unwrap();
                    let mut log_alphas = Array1::from_elem(hmm.get_n_states(), Float::NEG_INFINITY);
                    log_alphas[[*state]] = *score;
                    let index = *map_index_phone.get(phone).unwrap();
                    let vec_score = hmm
                        .compute_next_score(log_alphas.view(), vec_state_logprob[index].column(t));

                    vec_score
                        .iter()
                        .enumerate()
                        .for_each(|(new_state, &new_score)| {
                            // let mut new_score = *new_score;
                            // penalize le self loop
                            // if path.get_state() == state {
                            //     new_score = new_score - threshold / state_self_loop_penality;
                            // }

                            let mut path = path.clone();
                            path.push_back(((phone.clone(), *semone_id), new_state));
                            path_score.push((path, new_score));
                        });

                    // Faire le passage au phoneme suivant en cas de dernier etat

                    // Si on veut ajouter du trigramme, c'est ici qu'il faut penser à le faire
                    if (phone == "fn" && *state == FN_STATES_COUNTS - 1)
                        || (phone == "foana" && *state == FOANA_STATES_COUNTS - 1)
                        || ((phone != "foana" && phone != "fn")
                            && *state == COMMON_STATES_COUNTS - 1)
                    {
                        // proceed to next phoneme
                        self.map_phone_hmm
                            .iter()
                            .filter(|(next_phone, _)| next_phone.ne(&phone))
                            .map(|(next_phone, next_hmm)| {
                                let index = *map_index_phone.get(next_phone).unwrap();
                                let vec_score = next_hmm.compute_next_score(
                                    next_hmm.get_log_initial(),
                                    vec_state_logprob[index].column(t),
                                );

                                (
                                    path,
                                    next_phone,
                                    score
                                        + (self.map_phone_hmm.get(phone).unwrap().get_log_final()
                                            [*state]
                                            + vec_score[0]
                                            + self.bigram.get_log_prob_bigram(
                                                phone,
                                                if next_phone == "foana" || next_phone == "fn" {
                                                    " "
                                                } else {
                                                    next_phone
                                                },
                                            )),
                                )
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                            .for_each(|(path, next_phone, new_score)| {
                                let mut path: PhoneStatePath = path.clone();
                                path.push_back(((next_phone.to_string(), 0), 0));
                                path_score.push((path, new_score));
                            });
                    }
                });

                if set_path_last_score_guard.is_empty() {
                    map_index_phone.iter().for_each(|(phone, index)| {
                        let vec_score = vec_state_logprob[*index].column(t);
                        let log_alphas = Array1::from_elem(vec_score.len(), Float::NEG_INFINITY);
                        let (state, new_score) = self
                            .map_phone_hmm
                            .get(phone)
                            .unwrap()
                            .compute_next_score(log_alphas.view(), vec_score)
                            .into_iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(&b.1))
                            .unwrap();

                        path_score.push((PhoneStatePath::new(phone, 0, state), new_score));
                    });
                }
                // get the best score
                let max_score = path_score
                    .iter()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap()
                    .1;

                // pruning and normalize
                let mut alived_path = path_score
                    .into_iter()
                    .filter(|(_path, score)| score.gt(&(max_score - threshold)))
                    .map(|(path, score)| (path, score - max_score))
                    .collect::<Vec<_>>();

                // println!("alived path: {:?}", alived_path);
                self.path_pruning(&mut alived_path, &mut output, 100).await;

                // assign to set_path_last_score
                *set_path_last_score_guard = alived_path;
            }
            output
        })
    }
}
