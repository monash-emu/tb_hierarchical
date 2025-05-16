from tbh.runner_tools import (
    create_output_dir,
    run_full_analysis,
    get_full_default_params
)
from tbh.paths import OUTPUT_PARENT_FOLDER

import multiprocessing as mp
import sys
from pathlib import Path
from time import time, sleep

ANALYSIS_NAME = "multi_configs"

# This script is running an array job.
# Here the term "array_job" refers to the higher-level array job, which is a group of individual "tasks".

studies_dict_list = [
    {},  # empty dict as taks IDs start at 1
    {  # task 1
        "majuro": {
            "pop_size": 27797,
        },
        "vietnam": { 
            "pop_size": 100.e6,
        }    
    },
    {  # task 2
        "majuro": {
            "pop_size": 27797,
        },
        "majuro_copy": {
            "pop_size": 27797,
        }    
    },  # task 3
        {
        "majuro": {
            "pop_size": 27797,
        },
        "vietnam_no_target": {
            "pop_size": 100.e6,
        }    
    },  # task 4
        {
        "majuro": {
            "pop_size": 27797,
        },
        "vietnam": { 
            "pop_size": 100.e6,
        },
        "majuro_copy": {
            "pop_size": 27797,
        } 
    },  # task 5
        {
        "majuro": {
            "pop_size": 27797,
        },
        "vietnam": { 
            "pop_size": 100.e6,
        },
        "majuro_copy": {
            "pop_size": 27797,
        },
        "vietnam_no_target": {
            "pop_size": 100.e6,
        } 
    },
]



if __name__ == "__main__":
    start_time = time()

    # Prepare output folder
    array_job_id, task_id = int(sys.argv[1]), int(sys.argv[2])
    mp.set_start_method("spawn")  # previously "forkserver"
    print(f"Create output directory")
    output_dir = create_output_dir(array_job_id, task_id, ANALYSIS_NAME)

    # Specify and run analysis
    analysis_config = {
        # Metropolis config
        'chains': 4,
        'tune': 10000,
        'draws': 40000,

        # Full runs config
        'burn_in': 20000,
        'full_runs_samples': 1000
    }

    studies_dict = studies_dict_list[task_id]
    params = get_full_default_params(studies_dict)

    print(f"Start analysis for array_job {array_job_id}, task {task_id}, {ANALYSIS_NAME}")
    run_full_analysis(studies_dict=studies_dict, params=params, analysis_config=analysis_config, output_folder=output_dir)
    print(f"Finished in {time() - start_time} seconds", flush=True)
