from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel



class Environ(BaseModel):
    """Environment configuration for RL4KMC."""
    PROJECT_ROOT: Path = Path(__file__).parent
    
    # General settings
    debug: bool = False

    # Output settings
    output_dir: Path = PROJECT_ROOT / "output"
    enable_worker_debug_log: bool = False

    # Model settings
    model_dir: Path = PROJECT_ROOT / "models"
    model_path: Path = model_dir / "actor_agent.pt"
    
    
def print_environ() -> None:
    print("Environment Configuration:")
    for field_name, value in ENVIRON.model_dump().items():
        print(f"  {field_name}: {value}")


ENVIRON = Environ()

if __name__ == "__main__":
    print_environ()