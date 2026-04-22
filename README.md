# KNN

## How to run the learning process on *Metacentrum*

*Note: you can access the command line through the [OnDemand Dashboard](#ondemand) > Clusters > Perian Shell Access (you can use any shell access, not just Perian).*

1. Clone repository
1. Store the data in a directory within the cloned repository (for example `KNN/data`, where `KNN` is the name of the repository)
1. In `learnconfig.json`:
    1. set the paths to the `lmdb` db
    1. set the `output_model_dir` to persistent storage, not `scratch` (e.g., `/storage/brno2/home/<user_name>/output`)
    1. configure the desired parameters
1. In `train_job.sh` configure the PBS job requirements
    1. `-N` is the name of the job
    1. `-l` is list of requirements
    1. `-q` is the name of the queue; use `gpu` for jobs that have walltime set in range `0 - 48:00:00` and `gpu_long` for jobs with `24:00:01 - 336:00:00`
    1. change the `DATADIR` variable to match the cloned repository path
1. Once you have everything configured, you can submit the batch job with `qsub train_job.sh`
1. To monitor the job, you can check the status from the terminal with `qstat <id_ob_job>`, for more information you can use the `-f` option. Or you can use the [metacentrum website](#metacentrum).

### Useful links

[<span id="ondemand">OnDemand</span> (single access point for HPC resources)](https://ondemand.metacentrum.cz/) <br>
[<span id="metacentrum">Web overview / job monitoring</span>](https://metavo.metacentrum.cz/cs/state/) <br>
[Metacentrum documentation](https://docs.metacentrum.cz/en/docs/welcome)