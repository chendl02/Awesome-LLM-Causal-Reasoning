To replicate our results, first navigate to the `src` directory, then run the `eval_all.py` script, which will generate the model results. Alternatively, browse the `llm_result` folder to review the raw data directly. 

Each file in `llm_result` follows the naming convention:  
`{Model_name}_{seed}_{sample_num}_{few_shot}_{direct_io}.json`  
For example: `claude-3-5-sonnet-20240620_seed_42_sample_num_100_few_shot_False_direct_io_True.json`.

To explore the dataset, navigate to the `dataset/{dataset_name}` folder, and for the corresponding prompt, check the `prompt/{dataset_name}` folder. The merged results can be found in the `result` folder.

To acclearate the process, run the bash script `run_all.sh` to generate the results.
