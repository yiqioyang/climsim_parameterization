def load_manifest_to_memory(path):
    with open(path, "r", buffering=1024 * 1024) as f:
        return [line.rstrip("\n") for line in f if line.strip()]
