pub mod bigram;
pub mod viterbi_impl;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use ndarray::{ArcArray2, Array1, Array2, ArrayView1, Axis};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelExtend,
    ParallelIterator,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{
    RwLock,
    mpsc::{self},
};

use crate::{
    functions::get_phoneme_from_phone_state_path, hmm_gmm::HMMGMM, monophone::bigram::Bigram,
    phone_state_path::PhoneStatePath, types::Float,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonoPhone {
    map_phone_hmm: BTreeMap<String, HMMGMM>,
    bigram: Bigram,
    //n_iter: usize,  // 10 à 15 pour monophone et 20 à 40 pour triphone
    tolerance: Float, // 1e-4,
    convergence: Float,
    global_mean: Array1<Float>,
    global_var: Array1<Float>,
}

// train convergence = (ll(n) - ll(n-1))/ll(n-1) < tol
// pruning : score alived > score max - threshold

impl MonoPhone {
    pub fn new(
        bigram: &Bigram,
        tolerance: Float,
        convergence: Float,
        global_mean: ArrayView1<Float>,
        global_var: ArrayView1<Float>,
    ) -> Self {
        Self {
            map_phone_hmm: BTreeMap::new(),
            bigram: bigram.clone(),
            tolerance,
            convergence,
            global_mean: global_mean.to_owned(),
            global_var: global_var.to_owned(),
        }
    }

    pub fn get_bigram(&self) -> &Bigram {
        &self.bigram
    }

    pub fn get_convergence(&self) -> Float {
        self.convergence
    }

    pub fn get_global_mean(&self) -> ArrayView1<'_, Float> {
        self.global_mean.view()
    }

    pub fn get_global_var(&self) -> ArrayView1<'_, Float> {
        self.global_var.view()
    }

    pub fn get_tolerance(&self) -> Float {
        self.tolerance
    }

    pub fn get_phone_hmm_gmm(&self) -> &BTreeMap<String, HMMGMM> {
        &self.map_phone_hmm
    }

    pub fn get_phone_hmm_gmm_mut(&mut self) -> &mut BTreeMap<String, HMMGMM> {
        &mut self.map_phone_hmm
    }

    // Compute the log prob of each state of each phone
    /// Return hashmap mapping phone to index inside the vec of array2 log prob, and vec of array2 log prob (n_state, n_obs) of each phone
    fn compute_log_prob_for_state_for_every_phone(
        &self,
        mfcc: ArcArray2<Float>,
    ) -> (HashMap<String, usize>, Vec<Array2<Float>>) {
        // Compute the log prob of each state of each phone
        let mut map_index_phone = HashMap::new();
        let mut vec_state_logprob = Vec::new();
        self.map_phone_hmm.iter().for_each(|(phone, hmm)| {
            map_index_phone.insert(phone.clone(), vec_state_logprob.len());
            vec_state_logprob.push(hmm.get_log_observation_per_state(mfcc.clone()));
        });
        (map_index_phone, vec_state_logprob)
    }

    pub async fn path_pruning(
        &self,
        alived_path: &mut Vec<(PhoneStatePath, Float)>,
        output: &mut String,
        max_path_len: usize,
    ) {
        if alived_path.is_empty() {
            return;
        }

        // Add the path inside matrix
        let alived_len = alived_path.len();
        let path_len = alived_path[0].0.len();
        let mut vec = Vec::with_capacity(alived_len * path_len);
        alived_path
            .iter_mut()
            .for_each(|(phone_state_path, score)| {
                vec.par_extend(
                    phone_state_path
                        .par_iter()
                        .cloned()
                        .map(|(p, st)| (p, st, *score)),
                );
            });

        let mut arr = Array2::from_shape_vec([alived_path.len(), path_len], vec).unwrap();

        // add the remaining path inside hashset
        let index_to_save: Arc<RwLock<HashSet<usize>>> =
            Arc::new(RwLock::new(HashSet::from_iter(0..alived_path.len())));
        let index_to_save_clone = index_to_save.clone();

        let (index_to_save_tx, mut index_to_save_rx) = mpsc::unbounded_channel();

        // we don't use axis(1) to performance concern
        arr.reverse_axes();

        rayon::scope(|s| {
            s.spawn(move |_| {
                arr.axis_iter(Axis(0))
                    .into_par_iter()
                    .with_min_len(rayon::current_num_threads())
                    .for_each(|arr_phone_state_score| {
                        let mut vec_save = vec![true; arr_phone_state_score.len()];
                        let mut hash_phone_state_max_score_to_save: HashMap<
                            (String, usize, usize),
                            (usize, Float),
                        > = HashMap::new();
                        arr_phone_state_score.iter().enumerate().for_each(
                            |(index, phone_state_score)| {
                                if let Some((saved_index, saved_score)) =
                                    hash_phone_state_max_score_to_save.get_mut(&(
                                        phone_state_score.0.0.clone(),
                                        phone_state_score.0.1,
                                        phone_state_score.1,
                                    ))
                                {
                                    let end_state = self
                                        .map_phone_hmm
                                        .get(&phone_state_score.0.0)
                                        .unwrap()
                                        .get_n_states()
                                        - 1;

                                    // remove the path within lesser score
                                    if *saved_score < phone_state_score.2
                                        && phone_state_score.1 == end_state
                                    {
                                        vec_save[*saved_index] = false;
                                        *saved_index = index;
                                        *saved_score = phone_state_score.2;
                                    }
                                } else {
                                    hash_phone_state_max_score_to_save.insert(
                                        (
                                            phone_state_score.0.0.clone(),
                                            phone_state_score.0.1,
                                            phone_state_score.1,
                                        ),
                                        (index, phone_state_score.2),
                                    );
                                }
                            },
                        );
                        index_to_save_tx.send(vec_save).unwrap();
                    });
            });
        });

        tokio::spawn(async move {
            let index_to_save = index_to_save_clone.clone();
            while let Some(vec_save) = index_to_save_rx.recv().await {
                let index_to_save_clone = index_to_save.clone();
                let mut index_to_save_guard = index_to_save_clone.write().await;
                *index_to_save_guard = index_to_save_guard
                    .intersection(
                        &vec_save
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| vec_save[*i])
                            .map(|(i, _)| i)
                            .collect::<HashSet<_>>(),
                    )
                    .cloned()
                    .collect::<HashSet<_>>();
                drop(index_to_save_guard);
            }
        })
        .await
        .unwrap();

        let index_to_save_clone = index_to_save.clone();
        let index_to_save_guard = index_to_save_clone.read().await;

        // prune
        *alived_path = alived_path
            .iter()
            .enumerate()
            .filter(|(index, _)| index_to_save_guard.contains(index))
            .map(|(_, val)| val)
            .cloned()
            .collect();

        drop(index_to_save_guard);

        // Émission des phonèmes confirmés
        if !alived_path.is_empty()
            && (alived_path.len() == 1 || alived_path[0].0.len() > max_path_len)
        {
            alived_path.sort_by(|a, b| a.1.total_cmp(&b.1));
            alived_path.reverse();
            let vec = get_phoneme_from_phone_state_path(&alived_path[0].0);
            if let Some(last) = vec.last()
                && last.2.unwrap_or(0) < alived_path[0].0.len() - 1
            {
                for out in vec.iter() {
                    if out.0.0 == "foana" || out.0.0 == "fn" {
                        output.push(' ');
                    } else {
                        output.push_str(out.0.0.as_str());
                    }
                }
                let cutoff = last.2.unwrap();
                for alived in alived_path.iter_mut() {
                    alived.0.remove_to(cutoff);
                }
            }
        }
    }
}
