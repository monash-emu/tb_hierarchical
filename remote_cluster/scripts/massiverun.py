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

# analysis_details = [
#     {},  # empty dict as task IDs start at 1
#     # TASK 1 
#     {
#         "description": "[Ca Mau]\n - TB prevalence",
#         "studies_dict": {
#             "camau": {
#                 "pop_size": 1194300,
#                 "included_targets": ["tb_prevalence_per100k", "raw_notifications"]
#             },
#         },
#     }, 
#     # TASK 2
#     {
#         "description": "[Ca Mau]\n - LTBI prevalence",
#         "studies_dict": {
#             "camau": {
#                 "pop_size": 1194300,
#                 "included_targets": ["ltbi_prop", "raw_notifications"]
#             },
#         },
#     }, 
#     # TASK 3
#     {
#         "description": "[Ca Mau]\n - TB prevalence \n - LTBI prevalence",
#         "studies_dict": {
#             "camau": {
#                 "pop_size": 1194300,
#                 "included_targets": ["tb_prevalence_per100k", "ltbi_prop", "raw_notifications"]
#             },
#         },
#     },     
#     # TASK 4
#     {
#         "description": "[Ca Mau, Vietnam]",
#         "studies_dict": {
#             "camau": {
#                 "pop_size": 1194300,
#                 "included_targets": ["tb_prevalence_per100k", "ltbi_prop", "raw_notifications"]
#             },
#             "vietnam": { 
#                 "pop_size": 100.e6,
#                 "included_targets": ["ltbi_prop", "tb_prevalence_per100k", "raw_notifications"]
#             }  
#         },
#     },
#     # TASK 5
#     {
#         "description": "[Ca Mau, Majuro]",
#         "studies_dict": {
#             "camau": {
#                 "pop_size": 1194300,
#                 "included_targets": ["tb_prevalence_per100k", "ltbi_prop", "raw_notifications"]
#             },
#             "majuro": { 
#                 "pop_size": 27797,
#                 "included_targets": ["ltbi_prop", "tb_prevalence_per100k", "raw_notifications"]
#             }  
#         },
#     } 
# ]



analysis_details = [
    {},  # empty dict as task IDs start at 1
    # TASK 1 
    {
        "description": "[Vietnam]",
        "studies_dict": {
            "vietnam": { 
                "pop_size": 100.e6,
                "included_targets": ["ltbi_prop", "tb_prevalence_per100k", "raw_notifications"]
            }  
        },
    },
    # TASK 2
    {
        "description": "[Majuro]",
        "studies_dict": {
            "majuro": { 
                "pop_size": 27797,
                "included_targets": ["ltbi_prop", "tb_prevalence_per100k", "raw_notifications"]
            }  
        },
    } 
]


if __name__ == "__main__":
    start_time = time()

    # Prepare output folder
    array_job_id, task_id = int(sys.argv[1]), int(sys.argv[2])
    mp.set_start_method("spawn")  # previously "forkserver"
    print(f"Create output directory")
    output_dir = create_output_dir(array_job_id, task_id, ANALYSIS_NAME)
    # Write analysis description in output folder
    with open(output_dir / "description.txt", "w", encoding="utf-8") as f:
        f.write(analysis_details[task_id]["description"])

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

    studies_dict = analysis_details[task_id]["studies_dict"]
    params = get_full_default_params(studies_dict)

    print(f"Start analysis for array_job {array_job_id}, task {task_id}, {ANALYSIS_NAME}")
    run_full_analysis(studies_dict=studies_dict, params=params, analysis_config=analysis_config, output_folder=output_dir)
    print(f"Finished in {time() - start_time} seconds", flush=True)
