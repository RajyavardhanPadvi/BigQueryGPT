# semantic.py
import yaml
from typing import Dict
def load_workspaces(path: str = "workspaces.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("workspaces", {})
