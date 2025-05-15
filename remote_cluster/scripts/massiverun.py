from tbh.runner_tools import (
    run_full_analysis,
)
from tbh.paths import OUTPUT_PARENT_FOLDER

import multiprocessing as mp
import sys
from pathlib import Path
from time import time, sleep


ANALYSIS_NAME = "test_run"

# This script is running an array job.
# Here the term "array_job" refers to the higher-level array job, which is a group of individual "tasks".

if __name__ == "__main__":
    start_time = time()

    # Retrieve country iso3 to run
    task_id = int(sys.argv[2])  # specific to this particular run
    print(f"Start job #{task_id}", flush=True)

    mp.set_start_method("spawn")  # previously "forkserver"

    # create parent output directory for this array_job
    array_job_id = sys.argv[1]  # common to all the tasks from this array job
    array_job_output_dir = OUTPUT_PARENT_FOLDER / f"{array_job_id}_{ANALYSIS_NAME}"
    array_job_output_dir.mkdir(exist_ok=True)

    # create task-specific output dir
    output_dir = array_job_output_dir / f"task_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_config = {
        # Metropolis config
        'chains': 4,
        'tune': 50,
        'draws': 200,

        # Full runs config
        'burn_in': 100,
        'full_runs_samples': 100
    }
    run_full_analysis(analysis_config=analysis_config, output_folder=output_dir)

    print(f"Finished in {time() - start_time} seconds", flush=True)
