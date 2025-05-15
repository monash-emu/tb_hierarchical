from tbh.runner_tools import (
    create_output_dir,
    run_full_analysis,
)
from tbh.paths import OUTPUT_PARENT_FOLDER

import multiprocessing as mp
import sys
from pathlib import Path
from time import time, sleep


ANALYSIS_NAME = "test_full"

# This script is running an array job.
# Here the term "array_job" refers to the higher-level array job, which is a group of individual "tasks".

if __name__ == "__main__":
    start_time = time()

    # Prepare output folder
    array_job_id, task_id = int(sys.argv[1]), int(sys.argv[2])
    mp.set_start_method("spawn")  # previously "forkserver"
    output_dir = create_output_dir(array_job_id, task_id, ANALYSIS_NAME)

    # Specify and run analysis
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
