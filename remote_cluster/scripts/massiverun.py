from tbh import runner_tools as rt

import multiprocessing as mp
import sys
from pathlib import Path
from time import time, sleep
from tbh.paths import OUTPUT_PARENT_FOLDER

ANALYSIS_NAME = "fixed_matrix"

# idata_path = OUTPUT_PARENT_FOLDER / "47337364_full_analysis_1scenario" / "task_1"
idata_path = None

# This script is running an array job.
# Here the term "array_job" refers to the higher-level array job, which is a group of individual "tasks".

if __name__ == "__main__":
    start_time = time()

    # Prepare output folder
    array_job_id, task_id = int(sys.argv[1]), int(sys.argv[2])
    mp.set_start_method("spawn")  # previously "forkserver"
    print(f"Create output directory")
    output_dir = rt.create_output_dir(array_job_id, task_id, ANALYSIS_NAME)

    # Specify and run analysis
    analysis_config = rt.DEFAULT_ANALYSIS_CONFIG
    print(f"Start analysis for array_job {array_job_id}, task {task_id}, {ANALYSIS_NAME}")
    rt.run_full_analysis(analysis_config=analysis_config, output_folder=output_dir, idata_path=idata_path)
    print(f"Finished in {time() - start_time} seconds", flush=True)
