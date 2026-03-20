use std::mem;
use std::sync::{Arc, Mutex};

use crate::DType;

/// Max size of all buffers, in bytes.
/// Currently 4GB (1024 MB).
const MAX_BUFFERS_SIZE: usize = 4 * 1024 * 1024 * 1024;
/// When total pooled bytes exceed this, trim largest buffers down to this level.
const TRIM_THRESHOLD: usize = MAX_BUFFERS_SIZE / 2;

/// Tracks pool usage statistics.
#[derive(Debug, Clone)]
pub struct PoolMetrics {
    /// Current total capacity of all pooled buffers, in bytes.
    pub current_size: usize,
    /// Number of times a buffer was reused instead of allocated.
    pub hits: usize,
    /// Number of times a new buffer was allocated.
    pub misses: usize,
    /// Number of times a buffer was dropped due to pool size cap.
    pub drops: usize,
}

#[derive(Debug)]
/// A simple buffer pool to reuse Vec allocations across graph evaluation.
pub struct BufferPool<T> {
    pool: Vec<Vec<T>>,
    /// Usage statistics for this pool.
    pub metrics: PoolMetrics,
}

/// Shared, thread-safe reference to a BufferPool for automatic recycling.
pub type SharedPool<T> = Arc<Mutex<BufferPool<T>>>;

#[derive(Debug)]
/// Wrapper around Vec<T> that returns its buffer to the pool on drop.
pub struct PooledBuffer<T: DType> {
    buf: Vec<T>,
    pool: Option<SharedPool<T>>,
}

impl<T: DType> PooledBuffer<T> {
    /// Wrap an existing Vec and attach it to the pool.
    pub fn new(buf: Vec<T>, pool: SharedPool<T>) -> Self {
        PooledBuffer {
            buf,
            pool: Some(pool),
        }
    }

    /// Consume the wrapper and return the inner Vec without recycling.
    pub fn into_inner(mut self) -> Vec<T> {
        let buf = std::mem::take(&mut self.buf);
        self.pool = None;
        buf
    }
}

impl<T: DType> std::ops::Deref for PooledBuffer<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.buf
    }
}

impl<T: DType> std::ops::DerefMut for PooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.buf
    }
}

impl<T: DType> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let buf = std::mem::take(&mut self.buf);
            // Return the buffer to the pool
            let mut pool_guard = pool.lock().unwrap();
            pool_guard.recycle_buffer(buf);
        }
    }
}

impl<T: DType> BufferPool<T> {
    pub fn new() -> Self {
        BufferPool {
            pool: Vec::new(),
            metrics: PoolMetrics {
                current_size: 0,
                hits: 0,
                misses: 0,
                drops: 0,
            },
        }
    }

    /// Grab a Vec with at least `capacity`. Clears and reuses one from the pool if available.
    ///
    /// Returns an uninitialized vector with capacity and len of `capacity`.
    pub fn get_empty_buffer(&mut self, capacity: usize) -> Vec<T> {
        // Find the smallest buf that can fit this capacity.
        let mut smallest_found_buf = None;
        for i in 0..self.pool.len() {
            let this_buf_len = self.pool[i].len();

            let found_smaller_buf = smallest_found_buf
                .is_some_and(|found: usize| self.pool[found].len() < this_buf_len);
            let found_useable_buf = smallest_found_buf.is_none() && this_buf_len >= capacity;
            if found_smaller_buf || found_useable_buf {
                smallest_found_buf = Some(i);
            }
        }

        if let Some(idx) = smallest_found_buf {
            // record a reuse hit
            self.metrics.hits += 1;
            let mut buf = self.pool.swap_remove(idx);
            let buf_capacity = buf.capacity();
            self.metrics.current_size = self
                .metrics
                .current_size
                .saturating_sub(buf_capacity * mem::size_of::<T>());
            buf.clear();
            buf.reserve(capacity);

            debug_assert_eq!(
                self.metrics.current_size,
                self.pool
                    .iter()
                    .map(|b| b.capacity() * size_of::<T>())
                    .sum::<usize>()
            );

            buf
        } else {
            // record an allocation miss
            self.metrics.misses += 1;
            Vec::with_capacity(capacity)
        }
    }

    /// Grab a Vec with at least `capacity`. Clears and reuses one from the pool if available.
    ///
    /// Returns an uninitialized vector with capacity of `capacity` and len of 0.
    pub fn get_buffer(&mut self, capacity: usize) -> Vec<T> {
        let mut buf = self.get_empty_buffer(capacity);
        unsafe {
            buf.set_len(capacity);
        }
        buf
    }

    /// Return a Vec back into the pool for reuse.
    pub fn recycle_buffer(&mut self, buf: Vec<T>) {
        let buffer_bytes = buf.capacity() * mem::size_of::<T>();
        if self.metrics.current_size + buffer_bytes <= MAX_BUFFERS_SIZE {
            self.metrics.current_size += buffer_bytes;
            self.pool.push(buf);
            debug_assert_eq!(
                self.metrics.current_size,
                self.pool
                    .iter()
                    .map(|b| b.capacity() * size_of::<T>())
                    .sum::<usize>()
            );

            self.trim_excess();
        } else {
            // record a dropped buffer due to cap
            self.metrics.drops += 1;
        }
        // Otherwise drop buf and do not grow the pool further
    }

    /// Remove largest buffers until total pooled bytes ≤ TRIM_THRESHOLD.
    fn trim_excess(&mut self) {
        while self.metrics.current_size > TRIM_THRESHOLD {
            // Find index of largest buffer by byte capacity
            let mut max_idx = 0;
            let mut max_bytes = 0;
            for (i, buf) in self.pool.iter().enumerate() {
                let bbytes = buf.capacity() * mem::size_of::<T>();
                if bbytes > max_bytes {
                    max_bytes = bbytes;
                    max_idx = i;
                }
            }
            // Remove and drop that buffer
            self.pool.swap_remove(max_idx);
            self.metrics.current_size = self.metrics.current_size.saturating_sub(max_bytes);
        }

        debug_assert_eq!(
            self.metrics.current_size,
            self.pool
                .iter()
                .map(|b| b.capacity() * size_of::<T>())
                .sum::<usize>()
        );
    }

    /// Returns a snapshot of current pool metrics.
    #[allow(unused)]
    pub fn metrics(&self) -> PoolMetrics {
        self.metrics.clone()
    }
}
