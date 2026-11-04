# Displacement Externalities in Endogenous Position Games for Multi-Agent LLM Communication

Multi-agent LLM systems use shared transcripts to coordinate reasoning, yet
message lengths determine both communication cost and the context available to
later agents. Early writers can therefore benefit from additional text while
reducing the value of downstream communication. To study this interaction, we
formulate an endogenous position game with position-dependent attention and
token costs. We derive its unique subgame-perfect equilibrium in closed form and
show that message lengths decay geometrically across speaking positions. The
resulting displacement externality reduces team welfare, with inefficiency
growing as the team expands. The analysis also predicts a calibration reversal
in which flatter attention improves coverage of a given transcript but reduces
equilibrium efficiency as writers adapt. To address these incentives, we study
drawn-order rotation and token pricing, and derive a continuation-value toll
that implements the planner allocation. Under linear attention, this toll
reduces to a slot-dependent price. A policy trained with this price reaches
96.2% of planner welfare. Across three LLM families and four reasoning
benchmarks, slot pricing improves average accuracy by 14.8 points over
free-order communication while reducing generated tokens by 46.4%. It also
exceeds the strongest baseline within 0.4k tokens of its budget by 7.1 points.

![Endogenous position formation and incentive design](assets/figure1.png)

## Installation

Use Python 3.10 or later and install PyTorch for your system.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local,data]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

## Usage

Train the length policies:

```bash
python -m deepg train --schedule unpriced uniform slot \
  --output checkpoints
```

Training settings are in [configs/paper.yaml](configs/paper.yaml).

## Evaluation

Download the datasets from [HotpotQA](https://hotpotqa.github.io/),
[MuSiQue](https://github.com/StonyBrookNLP/musique),
[2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop), or
[FEVEROUS](https://fever.ai/dataset/feverous.html). Follow the
[data guide](docs/data.md) to prepare the required files.

For example, prepare HotpotQA and evaluate the trained slot-pricing policy:

```bash
python -m deepg prepare --dataset hotpotqa \
  --input data/hotpotqa/hotpot_dev_distractor_v1.json \
  --output data/hotpotqa/prepared.jsonl

python -m deepg evaluate --model Qwen3-8B --dataset hotpotqa \
  --method slot_trained --data data/hotpotqa/prepared.jsonl \
  --policy checkpoints --output results/qwen_hotpot_slot.jsonl

python -m deepg summarize results/qwen_hotpot_slot.jsonl \
  --output results/summary.csv
```

Other communication methods and baseline integrations are described in the
[evaluation guide](docs/protocols.md).

## License

The code uses the [MIT License](LICENSE). The paper image retains its published license.
