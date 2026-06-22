use std::collections::HashMap;

use crate::{
    COMMON_STATES_COUNTS, FN_STATES_COUNTS, FOANA_STATES_COUNTS, phone_state_path::PhoneStatePath,
};

pub fn get_phoneme_from_phone_state_path(
    phone_state_path: &PhoneStatePath,
) -> Vec<((String, usize), Option<usize>, Option<usize>)> {
    let mut buffer: HashMap<(String, usize), (Option<usize>, Option<usize>)> = HashMap::new();

    let mut ret = Vec::new();

    phone_state_path
        .iter()
        .enumerate()
        .for_each(|(index, ((phone, id), state))| {
            let (mut start, mut end) = buffer
                .get(&(phone.clone(), *id))
                .cloned()
                .unwrap_or((None, None));
            if *state == 0 {
                start = Some(index);
            } else if (phone == "fn" && *state == FN_STATES_COUNTS - 1)
                || (phone == "foana" && *state == FOANA_STATES_COUNTS - 1)
                || ((phone != "foana" && phone != "fn") && *state == COMMON_STATES_COUNTS - 1)
            {
                end = Some(index);
            }
            if start.is_some() && end.is_some() {
                ret.push(((phone.clone(), *id), start, end));
                buffer.remove(&(phone.clone(), *id));
            } else {
                buffer.insert((phone.clone(), *id), (start, end));
            }
        });

    ret
}
