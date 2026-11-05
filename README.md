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

## Algorithm Flow

To connect positional attention with communication incentives, we first derive the best response and equilibrium allocation. We then quantify the displacement externality and welfare loss. Finally, we derive rotation and pricing mechanisms that change positional and marginal incentives.

1. To learn the price response, a tabular policy maps remaining capacity and slot index to a distribution over feasible lengths on a 201-point grid.
2. We train it for 4000 updates using groups of 32 trajectories and policy gradients with each agent’s group-mean reward as a baseline. The reward is its payoff under the slot price, while reported welfare uses the true token cost.
3. The designer draws a permutation uniformly from all possible speaking orders and announces it before the round, and each agent knows the slot it will occupy and the price that slot carries.
4. At deployment, policy outputs set per-slot token caps before parallel drafting. The lengths are then committed in parallel, with every agent writing without reading the round it is part of, and the committed messages are placed in the drawn order.

The prompted variant communicates the same price through a sentence in the prompt. Both variants use the released language-model weights.

![Endogenous position formation and incentive design](assets/figure1.png)

**Figure 1.** Endogenous position formation, calibration reversal, and incentive design. Longer early messages shift downstream communication toward weaker positions. Drawn-order rotation and length pricing act on the corresponding positional and marginal incentives.

## Results

### Main Results

**Table 1.** Main results on four benchmarks, grouped by backbone. Values are five-seed means with standard deviations. Middle is the mean attention share on the middle third of the transcript. Tokens reports thousands of generated tokens per team, or per agent for single-agent inference. The two proposed variants use the same slot-price schedule through a trained policy or a prompted price signal.

![Main results from Table 1](assets/main_results.png)

### Ablation Study

**Table 2.** Coarse and fine ablations with three agents, averaged over three backbones and five seeds. Deviations are standard deviations across seeds. Matched variants use the 6.9k-token free-order budget; price multipliers scale the theory-guided uniform price. Δ is the accuracy gain over free order. Middle and Tokens follow Table 1.

![Component ablations from Table 2](assets/ablations.png)

### Analysis

![Analysis from Figure 2](assets/analysis.png)

**Figure 2.** Mechanism controls. (a) Accuracy under prompt controls and pricing on a predetermined order. (b) Predicted and measured capacity shares. (c) First-agent share as the stated price increases. (d) Message-length distributions. Error bars in (a)–(c) show five-seed standard deviations.

## Getting Started

### Environment Setup

Use Python 3.10 or later and install PyTorch for your system.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local,data]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

### Data Preparation

Download [HotpotQA](https://hotpotqa.github.io/),
[MuSiQue](https://github.com/StonyBrookNLP/musique),
[2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop), and
[FEVEROUS](https://fever.ai/dataset/feverous.html).
Set the dataset paths in `configs/paper.yaml` and save retrieved FEVEROUS evidence
to `data/feverous/evidence.json`. The loaders accept the benchmark formats described
in [data inputs](docs/data.md). Prepare the upstream baselines as described in
[baseline execution](docs/protocols.md).

### Training and Evaluation

Train the length policies, then evaluate the configured models, datasets, and
communication methods:

```bash
python -m deepg train --schedule unpriced uniform slot \
  --output checkpoints

python scripts/run_matrix.py --config configs/paper.yaml --policy checkpoints \
  --evidence data/feverous/evidence.json --execute --output results/main

python -m deepg summarize results/main/*/*/*.jsonl \
  --output results/summary.csv
```

## License

The code uses the [MIT License](LICENSE).
