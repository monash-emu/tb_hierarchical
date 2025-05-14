from tbh.runner_tools import (
    get_bcm_object,
    run_metropolis_calibration,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_PARAMS,
)
from tbh.plotting import plot_traces

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
    output_root_dir = Path.home() / "sh30/users/rragonnet/tb_hierarchical/remote_cluster/outputs"
    array_job_id = sys.argv[1]  # common to all the tasks from this array job
    array_job_output_dir = output_root_dir / f"{array_job_id}_{ANALYSIS_NAME}"
    array_job_output_dir.mkdir(exist_ok=True)

    # create task-specific output dir
    output_dir = array_job_output_dir / f"task_{task_id}"
    output_dir.mkdir(exist_ok=True)

    studies_dict = {
        "majuro": {
            "pop_size": 27797,
        },
        "study_2": {
            "pop_size": 50000,
        },
    }
    bcm = get_bcm_object(DEFAULT_MODEL_CONFIG, studies_dict, DEFAULT_PARAMS)
    idata = run_metropolis_calibration(bcm, 500, 100, 4, 4)

    n_io_retries = 3
    for attempt in range(n_io_retries):
        try:
            idata.to_netcdf(output_dir / "idata.nc")
            plot_traces(idata, 200, output_dir)
            break
        except:
            sleep(1)

    print(f"Finished in {time() - start_time} seconds", flush=True)
