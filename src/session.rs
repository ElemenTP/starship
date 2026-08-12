//! Session management for in-process prompt rendering.
//!
//! When starship runs as an in-process library (feature `in-process`), this module
//! provides persistent cross-invocation caching so that expensive operations like
//! git status detection don't need to be recomputed from scratch on every prompt
//! render within the same shell session.
//!
//! # Architecture
//!
//! - [`Session`] is a handle that the FFI host (zsh module, pwsh module) holds.
//!   It wraps an `Arc<SessionState>` and has a unique `id`.
//! - [`SessionState`] owns all caches: config, git repo discovery, git repo status,
//!   and directory contents. Each cache entry is keyed by path and invalidated
//!   based on TTL and filesystem metadata (mtime, size).
//! - A global registry maps session IDs to weak references, allowing [`Context`]
//!   to find its session during rendering without threading a reference through
//!   the entire module tree.
//!
//! # Cache Invalidation
//!
//! - **Config**: invalidated when the config file's mtime or size changes.
//! - **Git repo discovery**: invalidated after TTL expiry (path-keyed).
//! - **Git repo status**: invalidated after TTL expiry, OR when `.git/index`,
//!   `HEAD`, or `packed-refs` mtime changes (catches staged changes immediately).
//! - **Directory contents**: invalidated after TTL expiry.
//! - **TTL**: controlled by `STARSHIP_NATIVE_TTL_MS` env var (default 1000ms).
//! - **Force recompute**: set `STARSHIP_NATIVE_NO_CACHE=1` to bypass all caches.

use crate::config::StarshipConfig;
use crate::configs::StarshipRootConfig;
use crate::context::{DirContents, GitRepo, Properties, Target};
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic};
use std::time::{Duration, Instant};

// Re-export types used by context and modules.
pub use crate::config::StarshipConfig as ConfigForSession;

/// Default cache TTL: 5 second. This means cached results are reused across
/// rapid prompt redraws (e.g. holding down a key in ZLE) but expire quickly
/// enough that the user sees timely updates after git operations.
const DEFAULT_TTL_MS: u64 = 5000;

/// Environment variable to override the cache TTL in milliseconds.
const TTL_ENV: &str = "STARSHIP_NATIVE_TTL_MS";

/// Set this to "1" to disable all caching (force full recompute every render).
const NO_CACHE_ENV: &str = "STARSHIP_NATIVE_NO_CACHE";

// ---------------------------------------------------------------------------
// Cache entry helpers
// ---------------------------------------------------------------------------

/// A cache entry with TTL and optional filesystem metadata for fine-grained
/// invalidation.
struct CacheEntry<T> {
    value: T,
    created: Instant,
    /// Path this entry was computed for.
    path: PathBuf,
    /// Additional invalidation files with their observed mtimes.
    /// If any of these files' mtime has changed, the entry is stale.
    invalidation_files: Vec<(PathBuf, Option<std::time::SystemTime>)>,
}

impl<T> CacheEntry<T> {
    fn new(value: T, path: PathBuf, created: Instant) -> Self {
        Self {
            value,
            created,
            path,
            invalidation_files: Vec::new(),
        }
    }

    fn with_invalidation(mut self, files: Vec<(PathBuf, Option<std::time::SystemTime>)>) -> Self {
        self.invalidation_files = files;
        self
    }

    /// Check if this entry is still valid given the current TTL and filesystem state.
    fn is_valid(&self, ttl: Duration) -> bool {
        // Check TTL
        if self.created.elapsed() > ttl {
            return false;
        }
        // Check invalidation files
        for (file, recorded_mtime) in &self.invalidation_files {
            let current_mtime = std::fs::metadata(file).ok().and_then(|m| m.modified().ok());
            if current_mtime != *recorded_mtime {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// SessionState
// ---------------------------------------------------------------------------

/// Holds all cross-invocation caches for a single shell session.
///
/// All fields are behind `parking_lot::Mutex` so that reads (checking validity,
/// cloning cached values) and writes (populating caches) don't block each other
/// for long. The critical sections are very short — just pointer swaps and
/// metadata checks.
pub struct SessionState {
    /// Scoped rayon thread pool. Wrapped in Option+Mutex so that shutdown()
    /// can take ownership, send the terminate signal, and then wait for
    /// worker threads to actually exit before .so unload.
    rayon_pool: parking_lot::Mutex<Option<rayon::ThreadPool>>,

    /// Cached parsed configuration: (path, mtime, size, config, root_config).
    config_cache: parking_lot::Mutex<
        Option<(
            PathBuf,
            Option<std::time::SystemTime>,
            u64,
            Arc<StarshipConfig>,
            Arc<StarshipRootConfig>,
        )>,
    >,

    /// Cached git repository status (the expensive one).
    repo_status_cache:
        parking_lot::Mutex<Option<CacheEntry<Option<Arc<crate::modules::git_status::RepoStatus>>>>>,

    /// Cached git repository discovery result (ancestor walk + config load).
    git_repo_cache: parking_lot::Mutex<Option<CacheEntry<Result<GitRepo, String>>>>,

    /// Cached git metrics result (blob diff line counts).
    /// Keyed by repo root with the same invalidation as repo_status_cache.
    git_metrics_cache: parking_lot::Mutex<Option<CacheEntry<(usize, usize)>>>,

    /// Cached directory contents (error stored as string to be cloneable).
    dir_contents_cache: parking_lot::Mutex<Option<CacheEntry<Result<DirContents, String>>>>,

    /// Cached binary path resolution (which::which results).
    binary_cache: parking_lot::Mutex<HashMap<OsString, (Option<PathBuf>, Instant)>>,

    /// Cache hit/miss counters.
    stats: SessionStatsStatus,
}

#[derive(Debug, Default)]
pub struct SessionStatsStatus {
    pub config_hits: atomic::AtomicU64,
    pub config_misses: atomic::AtomicU64,
    pub repo_status_hits: atomic::AtomicU64,
    pub repo_status_misses: atomic::AtomicU64,
    pub git_repo_hits: atomic::AtomicU64,
    pub git_repo_misses: atomic::AtomicU64,
    pub git_metrics_hits: atomic::AtomicU64,
    pub git_metrics_misses: atomic::AtomicU64,
    pub dir_contents_hits: atomic::AtomicU64,
    pub dir_contents_misses: atomic::AtomicU64,
    pub binary_path_hits: atomic::AtomicU64,
    pub binary_path_misses: atomic::AtomicU64,
    pub renders: atomic::AtomicU64,
}

/// Cache performance counters.
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    pub config_hits: u64,
    pub config_misses: u64,
    pub repo_status_hits: u64,
    pub repo_status_misses: u64,
    pub git_repo_hits: u64,
    pub git_repo_misses: u64,
    pub git_metrics_hits: u64,
    pub git_metrics_misses: u64,
    pub dir_contents_hits: u64,
    pub dir_contents_misses: u64,
    pub binary_path_hits: u64,
    pub binary_path_misses: u64,
    pub renders: u64,
}

impl SessionState {
    pub fn new() -> Self {
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(crate::num_rayon_threads())
            .build()
            .expect("failed to build rayon thread pool");
        Self {
            rayon_pool: parking_lot::Mutex::new(Some(rayon_pool)),
            config_cache: parking_lot::Mutex::new(None),
            repo_status_cache: parking_lot::Mutex::new(None),
            git_repo_cache: parking_lot::Mutex::new(None),
            dir_contents_cache: parking_lot::Mutex::new(None),
            git_metrics_cache: parking_lot::Mutex::new(None),
            binary_cache: parking_lot::Mutex::new(HashMap::new()),
            stats: SessionStatsStatus::default(),
        }
    }

    /// Returns the current TTL from the environment, or the default.
    fn ttl() -> Duration {
        let ms = std::env::var(TTL_ENV)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_TTL_MS);
        Duration::from_millis(ms)
    }

    /// Returns true if caching is completely disabled.
    fn no_cache() -> bool {
        std::env::var(NO_CACHE_ENV).ok().as_deref() == Some("1")
    }

    // -- Config cache -------------------------------------------------------

    pub fn get_config(
        &self,
        config_path: &Path,
    ) -> Option<(Arc<StarshipConfig>, Arc<StarshipRootConfig>)> {
        if Self::no_cache() {
            self.stats
                .config_misses
                .fetch_add(1, atomic::Ordering::Relaxed);
            return None;
        }

        let (mtime, size) = std::fs::metadata(config_path)
            .ok()
            .map(|m| (m.modified().ok(), m.len()))
            .unwrap_or((None, 0));

        let cache = self.config_cache.lock();
        if let Some((ref cached_path, ref cached_mtime, cached_size, ref config, ref root_config)) =
            *cache
        {
            if cached_path == config_path && *cached_mtime == mtime && cached_size == size {
                self.stats
                    .config_hits
                    .fetch_add(1, atomic::Ordering::Relaxed);
                return Some((Arc::clone(config), Arc::clone(root_config)));
            }
        }

        self.stats
            .config_misses
            .fetch_add(1, atomic::Ordering::Relaxed);
        None
    }

    pub fn put_config(
        &self,
        config_path: PathBuf,
        config: Arc<StarshipConfig>,
        root_config: Arc<StarshipRootConfig>,
    ) {
        let mtime_size = std::fs::metadata(&config_path)
            .ok()
            .map(|m| (m.modified().ok(), m.len()))
            .unwrap_or((None, 0));
        let mut cache = self.config_cache.lock();
        *cache = Some((config_path, mtime_size.0, mtime_size.1, config, root_config));
    }

    // -- Repo status cache --------------------------------------------------

    /// Returns the cached repo status snapshot, if still valid.
    /// The status is keyed by the worktree root path, with additional
    /// invalidation on `.git/index`, `HEAD`, and `packed-refs` mtimes.
    pub fn get_repo_status(
        &self,
        repo_root: &Path,
    ) -> Option<Option<Arc<crate::modules::git_status::RepoStatus>>> {
        if Self::no_cache() {
            self.stats
                .repo_status_misses
                .fetch_add(1, atomic::Ordering::Relaxed);
            return None;
        }

        let cache = self.repo_status_cache.lock();
        if let Some(ref entry) = *cache {
            if entry.path == repo_root && entry.is_valid(Self::ttl()) {
                self.stats
                    .repo_status_hits
                    .fetch_add(1, atomic::Ordering::Relaxed);
                return Some(entry.value.clone());
            }
        }

        self.stats
            .repo_status_misses
            .fetch_add(1, atomic::Ordering::Relaxed);
        None
    }

    pub fn put_repo_status(
        &self,
        repo_root: PathBuf,
        status: Option<Arc<crate::modules::git_status::RepoStatus>>,
    ) {
        // Build invalidation file list: .git/index, HEAD, packed-refs.
        let inval_files = {
            let git_dir = repo_root.join(".git");
            vec![
                (git_dir.join("index"), mtime_of(&git_dir.join("index"))),
                (git_dir.join("HEAD"), mtime_of(&git_dir.join("HEAD"))),
                (
                    git_dir.join("packed-refs"),
                    mtime_of(&git_dir.join("packed-refs")),
                ),
            ]
        };

        let entry =
            CacheEntry::new(status, repo_root, Instant::now()).with_invalidation(inval_files);
        let mut cache = self.repo_status_cache.lock();
        *cache = Some(entry);
    }

    // -- Git repo discovery cache ------------------------------------------

    /// Look up a cached git repository discovery result for `dir`.
    /// Returns `None` on cache miss or if caching is disabled.
    pub fn get_git_repo(&self, dir: &Path) -> Option<Result<GitRepo, String>> {
        if Self::no_cache() {
            self.stats
                .git_repo_misses
                .fetch_add(1, atomic::Ordering::Relaxed);
            return None;
        }

        let cache = self.git_repo_cache.lock();
        if let Some(ref entry) = *cache {
            if entry.path == dir && entry.is_valid(Self::ttl()) {
                self.stats
                    .git_repo_hits
                    .fetch_add(1, atomic::Ordering::Relaxed);
                match &entry.value {
                    Ok(repo) => return Some(Ok(repo.clone())),
                    Err(e) => return Some(Err(e.clone())),
                }
            }
        }

        self.stats
            .git_repo_misses
            .fetch_add(1, atomic::Ordering::Relaxed);
        None
    }

    pub fn put_git_repo(&self, dir: PathBuf, result: &Result<GitRepo, String>) {
        let entry = CacheEntry::new(result.clone(), dir, Instant::now());
        let mut cache = self.git_repo_cache.lock();
        *cache = Some(entry);
    }

    // -- Git metrics cache -------------------------------------------------

    /// Look up cached git metrics (added, deleted) for a repo root.
    /// Uses the same invalidation-file pattern as repo_status_cache.
    pub fn get_git_metrics(&self, repo_root: &Path) -> Option<(usize, usize)> {
        if Self::no_cache() {
            self.stats
                .git_metrics_misses
                .fetch_add(1, atomic::Ordering::Relaxed);
            return None;
        }

        let cache = self.git_metrics_cache.lock();
        if let Some(ref entry) = *cache {
            if entry.path == repo_root && entry.is_valid(Self::ttl()) {
                self.stats
                    .git_metrics_hits
                    .fetch_add(1, atomic::Ordering::Relaxed);
                return Some(entry.value);
            }
        }

        self.stats
            .git_metrics_misses
            .fetch_add(1, atomic::Ordering::Relaxed);
        None
    }

    pub fn put_git_metrics(&self, repo_root: PathBuf, added: usize, deleted: usize) {
        let inval_files = {
            let git_dir = repo_root.join(".git");
            vec![
                (git_dir.join("index"), mtime_of(&git_dir.join("index"))),
                (git_dir.join("HEAD"), mtime_of(&git_dir.join("HEAD"))),
                (
                    git_dir.join("packed-refs"),
                    mtime_of(&git_dir.join("packed-refs")),
                ),
            ]
        };
        let entry = CacheEntry::new((added, deleted), repo_root, Instant::now())
            .with_invalidation(inval_files);
        let mut cache = self.git_metrics_cache.lock();
        *cache = Some(entry);
    }

    // -- Directory contents cache -------------------------------------------

    pub fn get_dir_contents(&self, dir: &Path) -> Option<Result<DirContents, std::io::Error>> {
        if Self::no_cache() {
            self.stats
                .dir_contents_misses
                .fetch_add(1, atomic::Ordering::Relaxed);
            return None;
        }

        let cache = self.dir_contents_cache.lock();
        if let Some(ref entry) = *cache {
            if entry.path == dir && entry.is_valid(Self::ttl()) {
                self.stats
                    .dir_contents_hits
                    .fetch_add(1, atomic::Ordering::Relaxed);
                match &entry.value {
                    Ok(dc) => return Some(Ok(dc.clone())),
                    Err(e) => {
                        return Some(Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            e.clone(),
                        )));
                    }
                }
            }
        }

        self.stats
            .dir_contents_misses
            .fetch_add(1, atomic::Ordering::Relaxed);
        None
    }

    pub fn put_dir_contents(&self, dir: PathBuf, result: &Result<DirContents, std::io::Error>) {
        let converted = match result {
            Ok(dc) => Ok(dc.clone()),
            Err(e) => Err(e.to_string()),
        };
        let entry = CacheEntry::new(converted, dir, Instant::now());
        let mut cache = self.dir_contents_cache.lock();
        *cache = Some(entry);
    }

    // -- Binary path cache --------------------------------------------------

    /// Look up a binary's full path from the cache.
    /// Returns `Some(Some(path))` if cached and found,
    /// `Some(None)` if cached and not found (don't bother re-checking PATH),
    /// `None` if not yet cached.
    pub fn get_binary_path(&self, binary_name: &OsString) -> Option<Option<PathBuf>> {
        if Self::no_cache() {
            self.stats
                .binary_path_misses
                .fetch_add(1, atomic::Ordering::Relaxed);
            return None;
        }

        let cache = self.binary_cache.lock();
        if let Some((cached_path, created)) = cache.get(binary_name) {
            if created.elapsed() <= Self::ttl() {
                self.stats
                    .binary_path_hits
                    .fetch_add(1, atomic::Ordering::Relaxed);
                return Some(cached_path.clone());
            }
        };

        self.stats
            .binary_path_misses
            .fetch_add(1, atomic::Ordering::Relaxed);
        None
    }

    pub fn put_binary_path(&self, binary_name: OsString, path: Option<PathBuf>) {
        let mut cache = self.binary_cache.lock();
        cache.insert(binary_name, (path, Instant::now()));
    }

    // -- Stats --------------------------------------------------------------

    pub fn bump_render(&self) {
        self.stats.renders.fetch_add(1, atomic::Ordering::Relaxed);
    }

    pub fn stats(&self) -> SessionStats {
        SessionStats {
            config_hits: self.stats.config_hits.load(atomic::Ordering::Relaxed),
            config_misses: self.stats.config_misses.load(atomic::Ordering::Relaxed),
            repo_status_hits: self.stats.repo_status_hits.load(atomic::Ordering::Relaxed),
            repo_status_misses: self
                .stats
                .repo_status_misses
                .load(atomic::Ordering::Relaxed),
            git_repo_hits: self.stats.git_repo_hits.load(atomic::Ordering::Relaxed),
            git_repo_misses: self.stats.git_repo_misses.load(atomic::Ordering::Relaxed),
            git_metrics_hits: self.stats.git_metrics_hits.load(atomic::Ordering::Relaxed),
            git_metrics_misses: self
                .stats
                .git_metrics_misses
                .load(atomic::Ordering::Relaxed),
            dir_contents_hits: self.stats.dir_contents_hits.load(atomic::Ordering::Relaxed),
            dir_contents_misses: self
                .stats
                .dir_contents_misses
                .load(atomic::Ordering::Relaxed),
            binary_path_hits: self.stats.binary_path_hits.load(atomic::Ordering::Relaxed),
            binary_path_misses: self
                .stats
                .binary_path_misses
                .load(atomic::Ordering::Relaxed),
            renders: self.stats.renders.load(atomic::Ordering::Relaxed),
        }
    }
}

fn mtime_of(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A handle to a persistent prompt-rendering session.
///
/// The FFI host creates one `Session` per shell session (via `ssp_session_create`),
/// holds it for the lifetime of the shell, and calls `render()` on every prompt.
pub struct Session {
    state: Arc<SessionState>,
}

impl Session {
    /// Create a new session and register it in the global registry.
    pub fn new() -> Self {
        let state = Arc::new(SessionState::new());
        Self { state }
    }

    /// Render a prompt for the given properties and target.
    ///
    /// Uses the session's scoped rayon pool so that worker threads can be
    /// cleanly shut down when the session is destroyed.
    pub fn render(&self, properties: Properties, target: Target) -> String {
        self.state.bump_render();
        let context =
            crate::context::Context::new_for_session(properties, target, Some(self.state.clone()));
        let pool = self.state.rayon_pool.lock();
        match *pool {
            Some(ref pool) => pool.install(|| crate::print::get_prompt(&context)),
            None => {
                log::warn!("render called after pool shutdown");
                String::new()
            }
        }
    }

    /// Return a reference to the session state.
    pub fn state(&self) -> &Arc<SessionState> {
        &self.state
    }

    /// Shut down the rayon thread pool and wait for all worker threads to
    /// terminate. Must be called before the shared library is unloaded.
    ///
    /// rayon's `ThreadPool::drop()` only signals workers to stop via a
    /// terminate latch; it does NOT block until threads actually exit.
    /// We must take ownership of the pool, trigger termination, and then
    /// wait for threads to finish before dlclose unmaps the .so.
    pub fn shutdown(&self) {
        if let Some(pool) = self.state.rayon_pool.lock().take() {
            let thread_count = pool.current_num_threads();
            drop(pool); // sends terminate signal to each worker
            // Workers were idle after the last install() returned, so they
            // exit within microseconds. A short sleep covers the race.
            std::thread::sleep(std::time::Duration::from_millis(100));
            log::debug!("rayon pool shutdown complete ({thread_count} workers)");
        }
    }
}

// ---------------------------------------------------------------------------
// RepoStatus re-export for convenience
// ---------------------------------------------------------------------------

/// Re-exported so that `SessionState` can store `RepoStatus` without
/// the `git_status` module needing to depend on `session`.
pub use crate::modules::git_status::RepoStatus;
