import json
from pathlib import Path
from typing import Any, Dict


def load_json_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    # Accept UTF-8 with/without BOM to avoid Windows editor incompatibilities.
    return json.loads(cfg_path.read_text(encoding="utf-8-sig"))


def save_json(path: str, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
