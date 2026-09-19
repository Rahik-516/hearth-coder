use std::collections::HashMap;

pub const MAX_ENTRIES: usize = 1024;

/// What a cache can do.
pub trait Store {
    fn get(&self, key: &str) -> Option<String>;
}

/// A bounded in-memory cache.
pub struct Cache {
    entries: HashMap<String, String>,
    capacity: usize,
}

impl Cache {
    pub fn new(capacity: usize) -> Self {
        Cache { entries: HashMap::new(), capacity }
    }

    pub fn insert(&mut self, key: String, value: String) -> bool {
        if self.entries.len() >= self.capacity {
            return false;
        }
        self.entries.insert(key, value);
        true
    }
}

impl Store for Cache {
    fn get(&self, key: &str) -> Option<String> {
        self.entries.get(key).cloned()
    }
}

pub fn build_cache() -> Cache {
    Cache::new(MAX_ENTRIES)
}
