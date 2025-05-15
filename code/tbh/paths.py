from pathlib import Path

def get_git_root() -> Path:
    """Return the root of the git repository."""
    path = Path.cwd().resolve()

    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("No .git folder found in path hierarchy")

REPO_ROOT_PATH = get_git_root()
OUTPUT_PARENT_FOLDER = REPO_ROOT_PATH / "remote_cluster" / "outputs"