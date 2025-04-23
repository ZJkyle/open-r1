# Open R1

*A fully open reproduction of DeepSeek-R1. This repo is a work in progress, let's build it together!*

**Table of Contents**  
1. [Overview](#overview)  
2. [Plan of attack](#plan-of-attack)  
3. [Installation](#installation)  
4. [Training models](#training-models)  
   - [SFT](#sft)  
   - [GRPO](#grpo)  
5. [Evaluating models](#evaluating-models)  
6. [Reproducing Deepseek's evaluation results](#reproducing-deepseeks-evaluation-results)  
7. [Data generation](#data-generation)  
   - [Generate data from a smol distilled R1 model](#generate-data-from-a-smol-distilled-r1-model)  
   - [Generate data from DeepSeek-R1](#generate-data-from-deepseek-r1)  
8. [Contributing](#contributing)

## Overview

## Installation
```shell 
docker-compose build
```
```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
```

```shell
uv pip install vllm==0.7.2
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
```

This will also install PyTorch `v2.5.1` and it is **very important** to use this version since the vLLM binaries are compiled for it. You can then install the remaining dependencies for your specific use case via `pip install -e .[LIST OF MODES]`. For most contributors, we recommend:

```shell
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, check whether your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

```shell
git-lfs --version
```

If it isn't installed, run:

```shell
sudo apt-get install git-lfs
```

### SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

#### Data decontamination

Following [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) the data can be decontaminated using the script at: [scripts/decontaminate.py](./scripts/decontaminate.py), which decontaminates a dataset using 8-grams and deduplicate the data. Sample run:

```shell
python scripts/decontaminate.py \
    --dataset "open-r1/verifiable-coding-problems-python" \
    --problem_column problem \
    --cleanup
```

It will decontaminate against the benchmark datasets, and remove the contaminated samples afterwards. If no argument `--new_dataset_name` is provided, the same dataset will be reused, adding a `_decontaminated`. It runs against the prompt, which for this dataset is the column `problem`, but a different one can be provided.

Arguments for the script:

```shell
usage: decontaminate.py [-h] --dataset DATASET [--split SPLIT] [--ngram_size NGRAM_SIZE] [--problem_column PROBLEM_COLUMN] [--cleanup] [--new_dataset_name NEW_DATASET_NAME]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Name of the dataset to check for contamination.
  --split SPLIT         Split to check for contamination, defaults to `train`.
  --ngram_size NGRAM_SIZE
                        Size of n-grams to build, defaults to 8.
  --problem_column PROBLEM_COLUMN
                        Name of the column containing the problem (prompt).
  --cleanup           Whether to remove the contaminated rows before pushing the dataset.
  --new_dataset_name NEW_DATASET_NAME
                        New name for the dataset. If not provided, will reuse the name and add a `_decontaminated` to the name.
```

### Launching jobs on a Slurm cluster

If you have access to a Slurm cluster, we provide a `slurm/train.slurm` script that will automatically queue training jobs for you. Here's how you can use it:

```shell
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm {model_name} {task} {config_suffix} {accelerator}
```

Here `{model_name}` and `{task}` are defined as above, while `{config_suffix}` refers to the specific config and `{accelerator}` refers to the choice of 🤗 Accelerate config in `recipes/accelerate_configs`. If you wish to override the default config parameters, you can provide them by appending a space-separated string like `'--arg1=value1 --arg2=value2'`. Here's a concrete example to run SFT on 1 node of 8 GPUs:

```shell
# Launch on Slurm and override default hyperparameters
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm Qwen2.5-1.5B-Instruct sft demo zero3 '--per_device_train_batch_size=1 --num_train_epochs=5'
```

You can scale the number of nodes by increasing the `--nodes` flag.

> [!NOTE]
> The configuration in `slurm/train.slurm` is optimised for the Hugging Face Compute Cluster and may require tweaking to be adapted to your own compute nodes.

## Evaluating models

We use `lighteval` to evaluate models, with custom tasks defined in `src/open_r1/evaluate.py`. For models which fit on a single GPU, run:

```shell
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

> [!IMPORTANT]
> You must set `max_model_length=32768` in the `vllm` command to align with the `max_new_tokens` we define per eval. Without this, `lighteval` will throw an error.

To increase throughput across multiple GPUs, use _data parallel_ as follows:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

For large models which require sharding across GPUs, use _tensor parallel_ and run:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

You can also launch an evaluation with `make evaluate`, specifying the model, task, and optionally the parallelism technique and number of GPUs.

To evaluate on a single GPU:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

To use Data Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

To use Tensor Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## Reproducing Deepseek's evaluation results

> [!NOTE]
> The DeepSeek-R1 paper uses sampling with 64 responses per query to estimate `pass@1`. Below, we report the results from sampling 1 response per query, which likely explains the small 1-3σ discrepancies between our results and theirs.

### AIME 2024

We are able to reproduce Deepseek's reported results on the AIME 2024 benchmark within ~1-3 standard deviations:

| Model                         | AIME 2024 (🤗 LightEval) | AIME 2024 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          26.7           |             28.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          56.6           |             55.5             |
| DeepSeek-R1-Distill-Qwen-14B  |          60.0           |             69.7             |
| DeepSeek-R1-Distill-Qwen-32B  |          73.2           |             72.6             |
| DeepSeek-R1-Distill-Llama-8B  |          43.3           |             50.4             |
| DeepSeek-R1-Distill-Llama-70B |          73.3           |             70.0             |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|aime24|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks aime24
```

### MATH-500

We are able to reproduce Deepseek's reported results on the MATH-500 benchmark within ~1-3 standard deviations:

| Model                         | MATH-500 (🤗 LightEval) | MATH-500 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          84.6           |             83.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          93.0           |             92.8             |
| DeepSeek-R1-Distill-Qwen-14B  |          95.0           |             93.9             |
| DeepSeek-R1-Distill-Qwen-32B  |          96.6           |             94.3             |
| DeepSeek-R1-Distill-Llama-8B  |          88.6           |             89.1             |
| DeepSeek-R1-Distill-Llama-70B |          96.4           |             94.5             |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|math_500|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks math_500
```

### GPQA Diamond

We are able to reproduce Deepseek's reported results on the GPQA Diamond benchmark within ~1-3 standard deviations:

| Model                         | GPQA Diamond (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:---------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |            34.3             |               33.8               |
| DeepSeek-R1-Distill-Qwen-7B   |            50.5             |               49.1               |
| DeepSeek-R1-Distill-Qwen-14B  |            59.6             |               59.1               |
| DeepSeek-R1-Distill-Qwen-32B  |            63.6             |               62.1               |
| DeepSeek-R1-Distill-Llama-8B  |            52.0             |               49.0               |
| DeepSeek-R1-Distill-Llama-70B |            67.2             |               65.2               |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|gpqa:diamond|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks gpqa
```

### LiveCodeBench

We are able to reproduce Deepseek's reported results on the LiveCodeBench code generation benchmark within ~1-3 standard deviations:

| Model                         | LiveCodeBench (🤗 LightEval) | LiveCodeBench (DeepSeek Reported) |
|:------------------------------|:----------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |             16.3             |               16.9               |
| DeepSeek-R1-Distill-Qwen-7B   |             36.6             |               37.6               |
| DeepSeek-R1-Distill-Qwen-14B  |             51.5             |               53.1               |
| DeepSeek-R1-Distill-Qwen-32B  |             56.6             |               57.2               |
| DeepSeek-R1-Distill-Llama-8B  |             37.0             |               39.6               |
| DeepSeek-R1-Distill-Llama-70B |             54.5             |               57.5               |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models, or data_parallel_size=8 with the smaller models for speed
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks lcb
```

## Data generation

### Generate data from a smol distilled R1 model

The following example can be run in 1xH100. 
First install the following dependencies:

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

Now save the following snippet into a file named `pipeline.py` and run it with `python pipeline.py`. It will generate 4 outputs for each of the 10 examples (change the username for the repository to your org/user name):

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

Take a look at the sample dataset at [HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b).


### Generate data from DeepSeek-R1

To run the bigger DeepSeek-R1, we used 2 nodes, each with 8×H100 GPUs using the slurm file present in this repo at `slurm/generate.slurm`. First, install the dependencies:

(for now we need to install the vllm dev wheel that [fixes the R1 cuda graph capture](https://github.com/vllm-project/vllm/commits/221d388cc5a836fa189305785ed7e887cea8b510/csrc/moe/moe_align_sum_kernels.cu))
```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

And then run the following command:

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]  
> While the job is running, you can setup an SSH tunnel through the cluster login node to access the Ray dashboard from your computer running `ssh -L 8265:ray_ip_head_node:8265 <login_node>`, then browsing `http://localhost:8265`

## Contributing

Contributions are welcome. Please refer to https://github.com/huggingface/open-r1/issues/23.

## Acknowledgements

This project is built with the collective efforts of many groups and individuals in the open AI community. We are especially grateful to the vLLM and SGLang teams for creating high-performance tooling to scale the rollouts of GRPO. We also thank the teams at [OpenThoughts](https://www.open-thoughts.ai), [Prime Intellect](https://www.primeintellect.ai), and [General Reasoning](https://gr.inc) for creating and sharing high-quality datasets for reasoning.

## Citation

If you find this project is useful in your own work, please consider citing as follows:

```
@misc{openr1,
    title = {Open R1: A fully open reproduction of DeepSeek-R1},
    url = {https://github.com/huggingface/open-r1},
    author = {Hugging Face},
    month = {January},
    year = {2025}
}
```
