# Execution protocols

The length-policy update follows Equation (50) of the supplementary material, with one baseline per slot and the stated group-sum step size. Training writes portable NumPy checkpoints and the observed learning trajectory.

Free order asks each agent for a nonnegative speaking delay before drafting the round. Agents then append in their chosen order and read the transcript available at their turn. Model-selected delays implement the agent's timing choice; the numeric delay contract is an execution convention. `free_order_delays` can instead supply an observed arrival trace. Scheduling calls and their generated tokens are recorded separately from message communication.

The communication limit is explicit in `generation.communication_tokens` and can be changed with `--communication-tokens`. The default matches the 32,768-token context ceiling. Each generation fits its output cap to the complete serialized input; evidence and transcript text are retained. Parallel messages commit their fitted caps before drafting. Outputs record the requested and effective caps.

The supplementary verification entrypoints write analytical or numerical JSON outputs from the implemented objectives.

```bash
python scripts/grid_verify.py --output results/grid_verify.json
python scripts/theory_closed.py --output results/theory_closed.json
python scripts/paper_numbers.py --output results/paper_numbers.json
python scripts/ablation_true.py --training checkpoints/slot/5/training.json \
  --output results/ablation_true.json
```

Attention-class ablations solve the sequential game and planner separately for each curve. Learning curves are read from training records produced by `python -m deepg train`.
