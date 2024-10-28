Here’s a polished version of your README section:

---

Our results are fully reproducible. To replicate them, you can run the `eval_all.py` script, which will generate the model results. Alternatively, if you prefer to review the raw data directly, you can browse the `llm_result` folder. 

Each file in `llm_result` follows the naming convention:  
`{Model_name}_{seed}_{sample_num}_{few_shot}_{direct_io}.json`  
For example: `claude-3-5-sonnet-20240620_seed_42_sample_num_100_few_shot_False_direct_io_True.json`.

To explore the dataset, navigate to the `dataset/{dataset_name}` folder, and for the corresponding prompt, check the `prompt/{dataset_name}` folder. The merged results can be found in the `result` folder.

---

If you want to acclearate the process, you can run the bash script `run_all.sh` to generate the results.