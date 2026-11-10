//! Ordered window processing on a reusable, bounded Rayon worker pool.
use crate::Result;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::sync::{Arc, Mutex, OnceLock};

struct CachedPool {
    configured: usize,
    workers: usize,
    pool: Arc<ThreadPool>,
}

#[derive(Default)]
struct PoolCache {
    current: Option<CachedPool>,
}

impl PoolCache {
    fn get(&mut self, configured: usize, workers: usize) -> Result<Arc<ThreadPool>> {
        if let Some(current) = &self.current {
            if current.configured == configured && current.workers >= workers {
                return Ok(Arc::clone(&current.pool));
            }
        }
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("minipoa-align-{i}"))
            .build()
            .map_err(|error| format!("cannot start alignment worker pool: {error}"))?;
        let pool = Arc::new(pool);
        self.current = Some(CachedPool {
            configured,
            workers,
            pool: Arc::clone(&pool),
        });
        Ok(pool)
    }
}

/// Map independent alignment windows while retaining their input order.
///
/// The CLI uses one configured thread count for its entire run. Its pool is
/// reused between reads, growing only when more simultaneous windows become
/// available. Retaining only the most recent configuration avoids an unbounded
/// pool cache for library callers that change the thread count. In-flight calls
/// keep their pool alive through `Arc` when another caller replaces the cache.
pub(crate) fn map_ordered<T, R, F>(items: &[T], threads: usize, process: F) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R> + Sync + Send,
{
    let configured = threads.max(1);
    let workers = configured.min(items.len());
    if workers <= 1 {
        return items.iter().map(process).collect();
    }
    static CACHE: OnceLock<Mutex<PoolCache>> = OnceLock::new();
    let pool = CACHE
        .get_or_init(|| Mutex::new(PoolCache::default()))
        .lock()
        .map_err(|_| "alignment worker pool cache is poisoned".to_string())?
        .get(configured, workers)?;
    // Indexed parallel collection preserves order. Collecting Results after the
    // workers finish also makes the first error follow input order.
    let results: Vec<Result<R>> = pool.install(|| items.par_iter().map(process).collect());
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn preserves_order_and_bounds_worker_count() {
        let input: Vec<_> = (0..128).collect();
        let output = map_ordered(&input, 3, |&value| {
            assert!(rayon::current_thread_index().unwrap() < 3);
            assert_eq!(rayon::current_num_threads(), 3);
            Ok(value * value)
        })
        .unwrap();
        assert_eq!(
            output,
            input.iter().map(|value| value * value).collect::<Vec<_>>()
        );
    }

    #[test]
    fn reuses_pool_and_only_grows_for_same_configuration() {
        let mut cache = PoolCache::default();
        let first = cache.get(4, 2).unwrap();
        let second = cache.get(4, 1).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let grown = cache.get(4, 4).unwrap();
        assert!(!Arc::ptr_eq(&first, &grown));
        assert_eq!(grown.current_num_threads(), 4);
        assert!(Arc::ptr_eq(&grown, &cache.get(4, 2).unwrap()));
        let changed = cache.get(2, 2).unwrap();
        assert_eq!(changed.current_num_threads(), 2);
        assert!(!Arc::ptr_eq(&grown, &changed));
    }

    #[test]
    fn empty_and_single_thread_calls_need_no_pool() {
        let calls = AtomicUsize::new(0);
        let empty: Vec<usize> = map_ordered::<usize, _, _>(&[], 100, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(1)
        })
        .unwrap();
        assert!(empty.is_empty());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(map_ordered(&[2, 3], 1, |x| Ok(x * 2)).unwrap(), [4, 6]);
    }

    #[test]
    fn propagates_first_error_in_input_order() {
        let error = map_ordered(&[0, 1, 2, 3], 4, |&value| {
            if value > 0 {
                Err(format!("failure {value}"))
            } else {
                Ok(value)
            }
        })
        .unwrap_err();
        assert_eq!(error, "failure 1");
    }
}
