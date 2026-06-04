import importlib
from pathlib import Path

_models_dir = Path(__file__).parent
for _f in sorted(_models_dir.glob("*.py")):
    _name = _f.stem
    if _name.startswith("_") or _name == "base":
        continue
    importlib.import_module(f".{_name}", __package__)
