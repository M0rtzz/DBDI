### Deployment Tutorial (English Version)



This tutorial will guide you through the deployment process.

------



#### Hardware Requirements 🖥️



- **Minimum Requirement**: One GPUs with at least 24GB VRAM each (e.g., RTX 3090 or RTX 4090).

------



#### Software Setup ⚙️



The code runs with Python 3 and requires PyTorch. We recommend using Anaconda or miniconda for environment management. Our code has been tested with `python=3.12` and `torch=2.7.0` on Linux.

**Create and activate the conda environment**:

Bash

```
conda env create -f environment.yml
```



------



#### Download Models 🤖



Our dataset is located in the `./data` directory. You will need `git-lfs` to download the models.

1. **Install git-lfs** (Example for Ubuntu):

   Bash

   ```
   # For Ubuntu
   sudo apt install git-lfs
   ```

2. **Run the download script**:

   Bash

   ```
   cd models
   chmod +x ./download_models.sh
   ./download_models.sh
   ```

3. **Verify directory structure**: Place the models in the `./models` directory, structured as follows:

   ```
   ./models
   ├── Llama-2-7b-chat-hf
   ├── Meta-Llama-3-8B-Instruct
   ├── Mistral-7B-Instruct-v0.2
   └── vicuna-7b-v1.5
   ...
   ```

------



#### Run the Application 



1. **Start the Web UI**:

   Bash

   ```
   python ui.py
   ```

2. **Access the Application**: After the script runs successfully, an anonymous URL will be generated in your terminal. Open your browser and visit this URL to access the application.



#### Run the Experiment 

Extract vectors for a single model and a single dataset

```
python extract_harm_vector.py --model llama-2 --datasets advbench --max_samples 100 --neuron_ratio 0.25
```

When running experiments on a single model, a parameter configuration file must be provided

```
python run_attack_experiment.py \
    --models llama-2 \
    --toxic_datasets harmbench \
    --test_datasets advbench \
    --intervention_layer 16 \
    --param_config_file ./parameter/model_name.json
```

