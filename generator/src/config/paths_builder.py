from pathlib import Path


class PathsBuilder:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.data_path = Path(self.base_path / "data")
        self.paths = {}

    def build_paths(self):
        self.paths = {}
