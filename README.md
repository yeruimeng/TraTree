<h1 align="center">
  Weak-to-Strong Generalization with Failure Trajectories: A Tree-based Approach to Elicit Optimal Policy in Strong Models
</h1>

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv-2507.18858-b31b1b.svg)](https://arxiv.org/abs/2507.18858)

</div>

<p align="center">
    <a href="https://www.linkedin.com/in/ruimeng-ye-14416b262/">Ruimeng&nbsp;Ye<sup>1</sup></a>, 
    <a href="https://zihanwang314.github.io/">Zihan&nbsp;Wang<sup>2</sup></a>, 
    <a href="https://scholar.google.com/citations?user=FvnT29sAAAAJ&hl=zh-TW">Yang&nbsp;Xiao<sup>1</sup></a>, 
    <a href="https://openreview.net/profile?id=~Zinan_Ling1">Zinan&nbsp;Ling<sup>1</sup></a>, 
    <a href="https://limanling.github.io/">Manling&nbsp;Li<sup>2</sup></a>, 
    <a href="https://scholar.google.com/citations?user=cdwA-5IAAAAJ&hl=en">Bo&nbsp;Hui<sup>1</sup></a>
</p>

<p align="center"><sup>1</sup>University&nbsp;of&nbsp;Tulsa,&nbsp; <sup>2</sup>Northwestern&nbsp;University</p>

## 📢 Updates

- **[2025-07-28]** Our paper is available on arXiv, check it out [here](https://www.arxiv.org/abs/2507.18858).

## 🌟 Overview

TraTree is a modular toolkit that upgrades a strong model using both successes and failures from a weaker agent. It builds trajectory trees, prunes them with MCTS, converts branches into preference pairs, and fine-tunes the strong model with SFT and DPO. Below are ready-to-run scripts let you test gains on WebShop and other benchmarks quickly.

## ⚙️ Installation

Get start with the environment setup:

```
conda env create -f tra_environment.yaml

```

## 🚀 Quick Start
Use the folowing command to sft model

```
./lora.sh

```

Then, construct preference pairs：
```
python construct_preference.py --model <your model name> --task <task name> --golden_traj_path <your data path> --output_path <output path>

```

W2SG DPO:
```
./dpo.sh

```

Tree construction and Optimization Trajectory:
```
python mcts.py

```

## 📊 Evaluation
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


