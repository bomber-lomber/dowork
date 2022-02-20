#![allow(dead_code)]
use rayon::prelude::*;
use std::slice::Chunks;
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum Error {
    #[error("The threshold value should be positive value")]
    WrongThresholdValue,
    #[error("Wrong threshold: {0}")]
    WrongThreshold(String),
}

pub enum Threshold<T: Sync> {
    Value(usize),
    Func(fn(seq: &[T]) -> Result<Chunks<T>, Error>),
}

impl<T: Sync> Threshold<T> {
    fn chunks<'a>(&self, seq: &'a [T]) -> Result<Chunks<'a, T>, Error> {
        match self {
            Threshold::Value(chunk_size) if *chunk_size > 0 => Ok(seq.chunks(*chunk_size)),
            Threshold::Value(_) => Err(Error::WrongThreshold(
                "The threshold value should be positive value".to_owned(),
            )),
            Threshold::Func(_fn) => _fn(seq),
        }
    }
}

/// Implement basic function to split some generic computational work between threads.
/// Split should occur only on some threshold - if computational work (input length) is shorter than this threshold,
/// no splitting should occur and no threads should be created.
/// Threshold can be just constant.
/// You should return:
///    1. Up to you, but probably some Vec of the same length as input(1)
///
/// # Examples
///
/// ```
/// use dowork::Error;
/// let result = exec(&vec![1, 2, 3, 4, 5], Threshold::Value(3), |item| {
///   item * 10i32
/// });
/// let expected: Result<Vec<i32>, Error> = Ok(vec![10, 20, 30, 40, 50]);
/// assert_eq!(result, expected);
/// ```
fn exec<T, R, IT>(seq: &[T], threshold: Threshold<T>, item_executor: IT) -> Result<Vec<R>, Error>
where
    T: Sync,
    R: Send,
    IT: Fn(&T) -> R + Sync + Send,
{
    if seq.is_empty() {
        return Ok(vec![]);
    }
    let chunks: Vec<_> = threshold.chunks(seq)?.collect();
    let res = match chunks.len() {
        0 => vec![],
        1 => compute_work(seq, &item_executor),
        _ => chunks
            .into_par_iter()
            .enumerate()
            .map(|(n, items)| {
                println!("Chunk {n}, worker: {:?}", std::thread::current().id());
                compute_work(items, &item_executor)
            })
            .reduce(
                || Vec::<R>::with_capacity(seq.len()),
                |mut a, b| {
                    a.extend(b);
                    a
                },
            ),
    };
    Ok(res)
}

#[inline]
fn compute_work<T, R, IT>(items: &[T], item_executor: &IT) -> Vec<R>
where
    IT: Fn(&T) -> R,
{
    items.iter().map(item_executor).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::ops::Range;
    use std::time::Duration;

    use rand::Rng;

    use crate::{exec, Error, Threshold};

    #[test]
    fn exec_sample() {
        let result = exec(&vec![1, 2, 3, 4, 5], Threshold::Value(3), |item| {
            item * 10i32
        });
        let expected: Result<Vec<i32>, Error> = Ok(vec![10, 20, 30, 40, 50]);
        assert_eq!(result, expected);
    }

    #[test]
    fn exec_fail_wrong_threshold_value() {
        let result = exec(&vec![1, 2, 3, 4, 5], Threshold::Value(0), |item| {
            item * 10i32
        });
        let expected: Result<Vec<i32>, Error> = Err(Error::WrongThresholdValue);
        assert_eq!(result, expected);
    }

    #[test]
    fn exec_fail_threshold_can_not_specify_chunks() {
        let result = exec(
            &vec![1, 2, 3, 4, 5],
            Threshold::Func(|_data| Err(Error::WrongThreshold("Sequence too big".to_owned()))),
            |_item| (),
        );
        assert_eq!(
            result,
            Err(Error::WrongThreshold("Sequence too big".to_owned()))
        );
    }

    #[test]
    fn exec_for_zero_len_array() {
        let seq = Vec::<i32>::new();
        let result = exec(&seq, Threshold::Value(10), |_| ());
        let expected: Result<Vec<()>, Error> = Ok(vec![]);
        assert_eq!(result, expected);
    }

    #[test]
    fn exec_for_one_chunk_array() {
        let seq = generate_test_data(5);
        let expected: Result<Vec<u64>, Error> = Ok(seq
            .iter()
            .map(|item| {
                let mut hasher = DefaultHasher::new();
                item.hash(&mut hasher);
                hasher.finish()
            })
            .collect());
        let result = exec(&seq, Threshold::Value(10), |item| {
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            hasher.finish()
        });
        assert_eq!(result, expected);
    }

    #[test]
    fn exec_for_ton_of_chunks_array() {
        let seq = generate_test_data(5000);
        let expected: Result<Vec<u64>, Error> = Ok(seq
            .iter()
            .map(|item| {
                let mut hasher = DefaultHasher::new();
                item.hash(&mut hasher);
                hasher.finish()
            })
            .collect());
        let result = exec(&seq, Threshold::Value(100), |item| {
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            hasher.finish()
        });
        assert_eq!(result, expected);
    }

    #[test]
    fn exec_for_ton_of_chunks_and_hard_work_array() {
        let seq = generate_test_data(100);
        let expected: Result<Vec<u64>, Error> = Ok(seq
            .iter()
            .map(|item| {
                let mut hasher = DefaultHasher::new();
                item.hash(&mut hasher);
                hasher.finish()
            })
            .collect());
        let result = exec(&seq, Threshold::Func(|seq| Ok(seq.chunks(1))), |item| {
            std::thread::sleep(Duration::from_secs(2));
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            hasher.finish()
        });
        assert_eq!(result, expected);
    }

    fn generate_test_data(seq_len: usize) -> Vec<String> {
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\
                            0123456789)(*&^%$#@!~";
        const ITEM_RANGE: Range<u32> = 0..30;
        let mut seq: Vec<String> = Vec::with_capacity(seq_len);
        let mut rng = rand::thread_rng();
        for _i in 0..seq_len {
            let item: String = ITEM_RANGE
                .map(|_| {
                    let idx = rng.gen_range(0..CHARSET.len());
                    CHARSET[idx] as char
                })
                .collect();
            seq.push(item)
        }
        seq
    }
}
