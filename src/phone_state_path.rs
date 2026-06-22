use std::{
    collections::VecDeque,
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

type Phone = String;
type State = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhoneStatePath {
    inner: VecDeque<((Phone, usize), State)>,
}

impl Deref for PhoneStatePath {
    type Target = VecDeque<((Phone, usize), State)>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for PhoneStatePath {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl PhoneStatePath {
    pub fn new(phone: &str, semone_id: usize, state: usize) -> Self {
        Self {
            inner: [((phone.to_string(), semone_id), state)].into(),
        }
    }

    pub fn remove_to(&mut self, index: usize) {
        if index == self.len() - 1 {
            eprintln!("I (PhoneStatePath) cannot delete myself");
            return;
        }
        self.inner.drain(..index);
    }

    pub fn all_phone(&self, phone: &str) -> bool {
        self.inner
            .par_iter()
            .with_min_len(rayon::current_num_threads())
            .try_fold(
                || true,
                |ret, item| {
                    if ret && item.0.0.eq(phone) {
                        Some(true)
                    } else {
                        None
                    }
                },
            )
            .try_reduce(|| true, |a, b| Some(a && b))
            .unwrap_or(false)
    }

    pub fn one_phone(&self) -> bool {
        self.all_phone(&self.iter().last().unwrap().0.0)
    }
}
