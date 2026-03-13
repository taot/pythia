def get_cache_dir(model_name: str, revision: str) -> str:
    return f"./.cache/{model_name}/{revision}"
