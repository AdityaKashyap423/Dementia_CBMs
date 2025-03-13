import langchain
from langchain.cache import SQLiteCache
from sqlalchemy.orm import Session

class CustomSQLiteCache(SQLiteCache):
    """Custom SQLite cache that fixes a bug in the original SQLite cache."""
    enable = True

    # Fixes: https://github.com/hwchase17/langchain/blob/8dfad874a2eadfa00931bbaef1736210f4cbfe1b/langchain/cache.py#L81
    def update(self, prompt: str, llm_string: str, return_val):
        if not self.enable:
            return
        """Look up based on prompt and llm_string."""
        for i, generation in enumerate(return_val):
            item = self.cache_schema(
                prompt=prompt, llm=llm_string, response=generation.text, idx=i
            )
            with Session(self.engine) as session, session.begin():
                # We'll merge the cache entry instead of adding it to avoid UNIQUE constraint
                # errors when adding a batch of results where two prompts were the exact same.
                session.merge(item)


class CacheStore:
    def __init__(self, database_path="./.langchain",  num_trials=1):
        self.database_path = database_path
        self.num_trials = num_trials
        self._setup_llmchain_caches()

    def _setup_llmchain_caches(self):
        """Setup langchain caches."""
        for i in range(self.num_trials):
            langchain.llm_cache = CustomSQLiteCache(f"{self.database_path}-{i}.db")

    def switch_cache(self, trial_num):
        if trial_num >= self.num_trials:
            raise ValueError(f"Trial number {trial_num} is greater than the number of trials {self.num_trials}")
        langchain.llm_cache = CustomSQLiteCache(f"{self.database_path}-{trial_num}.db")
