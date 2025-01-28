# Weak-to-Strong Generalization with Failure Trajectories:  A Tree-based Approach to Elicit Optimal Policy in Strong Models
## Installation

Get start with the environment setup:

```
conda env create -f tra_environment.yaml

```

## Quick Start
Use the folowing command to sft model

```
./lora.sh

```

Use the folowing command to sft model

```
./lora.sh

```

W2SG DPO:
```
./dpo.sh

```

Tree construction and Optimization Trajectory:
```
python mcts.py

```

## Evaluation
First download files from [Google file](https://example.com). Unzip the file, save data under the eval_agent and save data.zip and indexes.zip underthe envs/webshop.(The google drive link will be added later)

Then, launch the controller of FastChat
```
python -m fastchat.serve.controller

```

Use the following commands to conduct the evaluation, take WebShop Task as an example:
```
python -m fastchat.serve.model_worker --model-path <your model path> --port 21021 --worker-address http://localhost:21021 --gpus 0,1,2,3 --num-gpus 4 --max-gpu-memory 40GB --controller-address http://localhost:21020

python -m eval_agent.main --agent_config fastchat --model_name <your model name> --exp_config webshop --split test --verbose

```


