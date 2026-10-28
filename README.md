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
Code and configuration are available at
<https://anonymous.4open.science/r/DEEPG-94B7>.

![Endogenous position formation and incentive design](assets/figure1.png)

## Installation

Use Python 3.10 or later. Install a PyTorch build appropriate for the execution
hardware before loading a language model.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local,data,test]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

## Analytical model and length-policy training

Compute the equilibrium, planner allocation, and slot prices for the three-agent
game.

```bash
python -m deepg theory --alpha 0.08 --agents 3 --rounds 1 \
  --output results/theory.json
```

Train the tabular length policies under the three price conditions. The default
configuration uses a 201-point capacity/action grid, 4,000 updates, groups of 32
turns, and seeds 5–9. The resulting policies assign message-length caps during
language-model evaluation.

```bash
python -m deepg train --schedule unpriced uniform slot \
  --output checkpoints
```

The training output contains one `policy.npz` and learning record per condition
and seed. Analytical verification entry points are in [scripts/](scripts).

## Benchmark preparation

Obtain the official benchmark releases and retain their native development
splits.

| Benchmark | Official resource | Input |
| --- | --- | --- |
| HotpotQA | [Project](https://hotpotqa.github.io/) | Distractor development questions and contexts |
| MuSiQue-Ans | [Release](https://github.com/StonyBrookNLP/musique) | Answerable development questions and paragraphs |
| 2WikiMultihopQA | [Release](https://github.com/Alab-NII/2wikimultihop) | Development questions, contexts, and answer aliases |
| FEVEROUS | [Project](https://fever.ai/dataset/feverous.html) | Development claims and retrieved evidence |

The [input reference](docs/data.md) specifies native fields, answer
normalization, and FEVEROUS evidence records. Prepare a shared-format file
before evaluation.

```bash
python -m deepg prepare --dataset hotpotqa \
  --input data/hotpotqa/hotpot_dev_distractor_v1.json \
  --output data/hotpotqa/prepared.jsonl
```

Agent inputs contain the question and evidence passages. Reference answers
supply independent exact-match scoring after generation.

## Evaluation

Evaluate a trained slot-pricing policy with Qwen3-8B.

```bash
python -m deepg evaluate --model Qwen3-8B --dataset hotpotqa \
  --method slot_trained --data data/hotpotqa/prepared.jsonl \
  --policy checkpoints --output results/qwen_hotpot_slot.jsonl

python -m deepg summarize results/qwen_hotpot_slot.jsonl \
  --output results/summary.csv
```

Evaluation resumes matching completed question–seed records. Run metadata binds
inputs, decoding settings, and trained policies. Additional communication
procedures and upstream baseline interfaces are described in [the protocol
reference](docs/protocols.md).

## Configuration

[configs/paper.yaml](configs/paper.yaml) specifies the shared settings.

| Setting | Value |
| --- | --- |
| Backbones | Qwen3-8B, Gemma-3-27B, Mistral-Small-3.1-24B |
| Communication | 3 agents, 3 rounds |
| Token cost | 0.08 |
| Decoding | Temperature 0.7, top-p 0.95 |
| Context and communication capacity | 32,768 tokens |
| Maximum message length | 8,192 tokens |
| Evaluation seeds | 5, 6, 7, 8, 9 |

Pass an alternative configuration with `python -m deepg --config PATH ...`. The
command help lists procedure, attention-profile, and budget options.

## Software checks

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

The CPU suite checks analytical solutions, policy updates, communication
scheduling, benchmark inputs, and metric accounting.

## License

The code uses the [MIT License](LICENSE). The paper image retains its published
license.
