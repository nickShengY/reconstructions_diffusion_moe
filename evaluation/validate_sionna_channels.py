from pathlib import Path
import runpy


OLD_SCRIPT = Path("/scratch/nickyun/diffusion-test01/evaluation/validate_sionna_channels.py")


if __name__ == "__main__":
    if not OLD_SCRIPT.exists():
        raise FileNotFoundError(f"Missing Sionna validation script: {OLD_SCRIPT}")
    runpy.run_path(str(OLD_SCRIPT), run_name="__main__")
