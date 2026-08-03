from tbh import runner_tools as rt

import multiprocessing as mp
import sys
from itertools import product
from time import time
import yaml
from tbh.paths import OUTPUT_PARENT_FOLDER

ANALYSIS_NAME = "array_job"

# idata_path = OUTPUT_PARENT_FOLDER / "47337364_full_analysis_1scenario" / "task_1"
idata_path = None

# This script is running an array job.
# Here the term "array_job" refers to the higher-level array job, which is a group of individual "tasks".

REGRESSION_RATE_VALUES = [0.5, 1.0, 2.0, 4.0]
REL_SUS_UNREACHABLE_VALUES = [1.0, 1.5, 2.0, 3.0]


def build_param_grid() -> list[dict]:
    return [
        {
            "clinical_regression_rate": regression_rate,
            "infectiousness_loss_rate": regression_rate,
            "rel_sus_unreachable": rel_sus,
        }
        for regression_rate, rel_sus in product(REGRESSION_RATE_VALUES, REL_SUS_UNREACHABLE_VALUES)
    ]


def get_task_config(task_id: int, task_grid: list[dict]) -> dict:
    if task_id < 1 or task_id > len(task_grid):
        raise ValueError(
            f"Task id {task_id} is out of bounds for grid size {len(task_grid)}. "
            f"Use SLURM array 1-{len(task_grid)}."
        )
    return task_grid[task_id - 1]


def dump_task_map(array_job_id: int, task_grid: list[dict]) -> None:
    task_map = {task_id: cfg for task_id, cfg in enumerate(task_grid, start=1)}
    job_output_dir = OUTPUT_PARENT_FOLDER / f"{array_job_id}_{ANALYSIS_NAME}"
    job_output_dir.mkdir(parents=True, exist_ok=True)
    map_path = job_output_dir / "task_config_map.yaml"

    with open(map_path, "w") as yaml_file:
        yaml.dump(
            {
                "analysis_name": ANALYSIS_NAME,
                "grid_size": len(task_grid),
                "task_to_config": task_map,
            },
            yaml_file,
            sort_keys=False,
        )
    print(f"Saved task map to {map_path}", flush=True)

if __name__ == "__main__":
    start_time = time()

    # Prepare output folder
    array_job_id, task_id = int(sys.argv[1]), int(sys.argv[2])
    task_grid = build_param_grid()
    task_config = get_task_config(task_id, task_grid)
    dump_task_map(array_job_id, task_grid)

    mp.set_start_method("spawn")  # previously "forkserver"
    print(f"Create output directory")
    output_dir = rt.create_output_dir(array_job_id, task_id, ANALYSIS_NAME)

    # Specify and run analysis
    analysis_config = rt.DEFAULT_ANALYSIS_CONFIG
    print(f"Start analysis for array_job {array_job_id}, task {task_id}, {ANALYSIS_NAME}")
    print(f"Task {task_id} parameter overrides: {task_config}", flush=True)
    rt.run_full_analysis(
        analysis_config=analysis_config,
        output_folder=output_dir,
        idata_path=idata_path,
        param_overrides=task_config,
    )
    print(f"Finished in {time() - start_time} seconds", flush=True)
