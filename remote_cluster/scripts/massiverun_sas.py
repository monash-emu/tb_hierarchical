from tbh import runner_tools as rt

import multiprocessing as mp
import sys
from itertools import product
from time import time
import yaml
from tbh.paths import OUTPUT_PARENT_FOLDER

ANALYSIS_NAME = "sas"

# idata_path = OUTPUT_PARENT_FOLDER / "47337364_full_analysis_1scenario" / "task_1"
idata_path = None

# This script is running an array job.
# Here the term "array_job" refers to the higher-level array job, which is a group of individual "tasks".

sa_by_taskid = {
    1: "tpt_60",
    2: "subclinical_50",
    3: "homogeneous_mixing",
}

def dump_sa_map(array_job_id: int, sa_by_taskid: dict) -> None:
    job_output_dir = OUTPUT_PARENT_FOLDER / f"{array_job_id}_{ANALYSIS_NAME}"
    job_output_dir.mkdir(parents=True, exist_ok=True)
    map_path = job_output_dir / "sa_config_map.yaml"

    with open(map_path, "w") as yaml_file:
        yaml.dump(
            {
                "analysis_name": ANALYSIS_NAME,
                "grid_size": len(sa_by_taskid),
                "task_to_sa": sa_by_taskid,
            },
            yaml_file,
            sort_keys=False,
        )
    print(f"Saved sensitivity analysis map to {map_path}", flush=True)


if __name__ == "__main__":
    start_time = time()

    # Prepare output folder
    array_job_id, task_id = int(sys.argv[1]), int(sys.argv[2])

    if task_id == 1:
        dump_sa_map(array_job_id, sa_by_taskid)

    mp.set_start_method("spawn")  # previously "forkserver"
    print(f"Create output directory")
    output_dir = rt.create_output_dir(array_job_id, task_id, ANALYSIS_NAME)

    # Specify and run analysis
    analysis_config = rt.DEFAULT_ANALYSIS_CONFIG
    print(f"Start analysis for array_job {array_job_id}, task {task_id}, {ANALYSIS_NAME}")
    rt.run_full_analysis(
        analysis_config=analysis_config,
        output_folder=output_dir,
        idata_path=idata_path,
        sensitivity_analysis=sa_by_taskid[task_id]
    )
    print(f"Finished in {time() - start_time} seconds", flush=True)
