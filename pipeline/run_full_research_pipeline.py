from pathlib import Path
import runpy


OLD_SCRIPT = Path("/scratch/nickyun/diffusion-test01/pipeline/run_full_research_pipeline.py")


if __name__ == "__main__":
    if not OLD_SCRIPT.exists():
        raise FileNotFoundError(f"Missing full research pipeline script: {OLD_SCRIPT}")
    runpy.run_path(str(OLD_SCRIPT), run_name="__main__")
