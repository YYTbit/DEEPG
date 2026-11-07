# Displacement Externalities in Endogenous Position Games for Multi-Agent LLM Communication

Multi-agent LLM systems use shared transcripts to coordinate reasoning, yet message lengths determine both communication cost and the context available to later agents. Early writers can therefore benefit from additional text while reducing the value of downstream communication. To study this interaction, we formulate an endogenous position game with position-dependent attention and token costs. We derive its unique subgame-perfect equilibrium in closed form and show that message lengths decay geometrically across speaking positions. The resulting displacement externality reduces team welfare, with inefficiency growing as the team expands. The analysis also predicts a calibration reversal in which flatter attention improves coverage of a given transcript but reduces equilibrium efficiency as writers adapt. To address these incentives, we study drawn-order rotation and token pricing, and derive a continuation-value toll that implements the planner allocation. Under linear attention, this toll reduces to a slot-dependent price. A policy trained with this price reaches 96.2% of planner welfare. Across three LLM families and four reasoning benchmarks, slot pricing improves average accuracy by 14.8 points over free-order communication while reducing generated tokens by 46.4%. It also exceeds the strongest baseline within 0.4k tokens of its budget by 7.1 points.

## Algorithm Flow

The method overview is as follows:

![Endogenous position formation and incentive design](assets/figure1.png)

**Figure 1.** Endogenous position formation, calibration reversal, and incentive design. Longer early messages shift downstream communication toward weaker positions. Drawn-order rotation and length pricing act on the corresponding positional and marginal incentives.

Each token written by an early agent consumes shared capacity and shifts subsequent messages to later positions. The writer receives the private benefit of additional text, while downstream agents bear part of its cost.

Drawn-order rotation assigns and announces a random speaking order, after which writers commit their lengths in parallel. Per-token pricing charges for additional context occupation.

The algorithm flow includes:

1. **Slot prices.** The exact instrument is a toll derived from downstream continuation values, which implements the planner allocation for continuous attention profiles. Under linear attention, it reduces to a slot-dependent price. Earlier slots pay more as their tokens displace more downstream communication, while the final slot carries a zero price.

2. **Training.** To learn the price response, a tabular policy maps remaining capacity and slot index to a distribution over feasible lengths on a 201-point grid. We train it for 4000 updates using groups of 32 trajectories and policy gradients with each agent’s group-mean reward as a baseline. The reward is its payoff under the slot price, while reported welfare uses the true token cost.

3. **Drawn-order rotation and deployment.** The lengths are then committed in parallel, with every agent writing without reading the round it is part of, and the committed messages are placed in the drawn order. At deployment, policy outputs set per-slot token caps before parallel drafting. The prompted variant communicates the same price through a sentence in the prompt. Both variants use the released language-model weights.

## Results

### Main Results

**Table 1.** Main results on four benchmarks, grouped by backbone. Values are five-seed means with standard deviations. Middle is the mean attention share on the middle third of the transcript. Tokens reports thousands of generated tokens per team, or per agent for single-agent inference. The two proposed variants use the same slot-price schedule through a trained policy or a prompted price signal.

![Main results from Table 1](assets/main_results.png)

**Slot pricing improves accuracy while reducing communication cost.** Table 1 compares methods on shared questions and checkpoints. The trained policy reaches 62.8 average accuracy with 3.7k generated tokens, compared with 48.0 and 6.9k under free order. This is a 14.8-point gain with a paired-bootstrap 95% CI of [12.5, 17.1], resampling questions and seeds, and a 46.4% reduction in generated tokens. The prompted price reaches 62.2 accuracy.

**Pricing improves allocation beyond protocol-level controls.** Predetermined order reaches 55.3 accuracy and round-robin 55.7; the four released designs reach 53.2–55.4 with 3.9–4.3k tokens. Slot pricing exceeds the strongest baseline within 0.4k tokens of its budget by 7.1 points, with a 95% CI of [5.6, 8.6]. These controls regulate speaking order, interaction structure, or participation. Slot pricing additionally charges early messages for the downstream communication they displace.

## Getting Started

### Environment Setup

```bash
conda create -n deepg python=3.10 -y
conda activate deepg
pip install -e '.[local,data]'
```

### Data Preparation

Download [HotpotQA](https://hotpotqa.github.io/), [MuSiQue](https://github.com/StonyBrookNLP/musique), [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop), and [FEVEROUS](https://fever.ai/dataset/feverous.html). Set the dataset paths in `configs/paper.yaml` and save retrieved FEVEROUS evidence to `data/feverous/evidence.json`. The loaders accept the benchmark formats described in [data inputs](docs/data.md).

The model profiles in `configs/paper.yaml` use [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Gemma-3-27B](https://huggingface.co/google/gemma-3-27b-it), and [Mistral-Small-3.1-24B](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503). Obtain access to each checkpoint under its model license before evaluation.

## Training

Train the length policies:

```bash
python -m deepg train --schedule unpriced uniform slot \
  --output checkpoints
```

Add `--resume` to continue saved training with the same configuration.

## Evaluation

Evaluate the trained and prompted slot-price variants across all three backbones and four benchmarks, then aggregate the outputs:

```bash
python scripts/run_matrix.py --config configs/paper.yaml --policy checkpoints \
  --evidence data/feverous/evidence.json --execute --output results/main

python -m deepg summarize results/main/*/*/*.jsonl \
  --output results/summary.csv
```

Use `--models` and `--datasets` to select entries from `configs/paper.yaml`. Evaluation resumes completed questions from the output records. Additional communication protocols and upstream baselines are described in [execution protocols](docs/protocols.md).

## License

The code uses the [MIT License](LICENSE).
