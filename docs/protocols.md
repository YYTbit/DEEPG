# Execution protocols

The length-policy update follows Equation (50) of the supplementary material, with one baseline per slot and the stated group-sum step size. Training writes portable NumPy checkpoints and the observed learning trajectory.

The training entry updates the tabular length policy. The three language-model backbones retain their released weights during communication. `--resume` restores the completed update, state visits, learning history and NumPy generator state. It checks the cost, price schedule, grid, seed and training settings before continuing.

The local backend uses Transformers 4.57.3 with each backbone's native model class, tokenizer, chat template and generation configuration. The shared temperature, top-p and precision come from `configs/paper.yaml`. A resolved repository revision is shared by configuration, tokenizer and weights and recorded with evaluation outputs; `deepg evaluate --revision` can select a particular revision. The matrix runner resolves each backbone once for the selected matrix. Interrupted evaluations resume complete question records with the same inputs and model identity.

Free order asks each agent for a nonnegative speaking delay before drafting the round. Agents then append in their chosen order and read the transcript available at their turn. Model-selected delays implement the agent's timing choice; the numeric delay contract is an execution convention. `free_order_delays` can instead supply an observed arrival trace. Scheduling calls and their generated tokens are recorded separately from message communication.

The communication limit is explicit in `generation.communication_tokens` and can be changed with `--communication-tokens`. The default matches the 32,768-token context ceiling. Each generation fits its output cap to the complete serialized input; evidence and transcript text are retained. Parallel messages commit their fitted caps before drafting. Outputs record the requested and effective caps.

## Upstream communication designs

The matrix accepts `--methods single_agent free_order predetermined_order round_robin debate` for the protocol controls. The four topology designs use their official implementations at the revisions in `configs/baselines.json`. Fetch them with

```bash
python -m deepg baseline-fetch --output external
```

Install each upstream project's dependencies and train its topology using that project's training entry. The settings in `configs/*_graph.json` select the graph construction at the pinned revision. The bridge reads a PyTorch dictionary whose keys name the corresponding graph attributes and whose values are their tensors or module state dictionaries. For GPTSwarm, the keys name attributes of `graph.connection_dist`. The loader checks parameter names and shapes and loads module states strictly.

Set the trained topology paths in `configs/baseline_commands.json`. The matrix then accepts `--methods AgentPrune AgentDropout GDesigner GPTSwarm`. Each upstream process receives the question and passages, while the evaluator retains reference answers. The bridge records generated-token counts and obtains attention measurements from the selected backbone.

## Attention calibration

The local `calibration` control accepts `--attention-profile` with a positive one-dimensional NumPy or JSON profile measured on a separate calibration input. `LocalModel.measure_attention` returns the per-position profile used to construct that input. `--calibration-strength` sets a value in [0, 1]. The correction enters the native attention interface as a key-position log bias, preserving the backbone's causal and sliding-window masks.

Calibration coordinates follow the complete input sequence across prompt chunks and sliding-window caches. Evaluation binds this convention to the saved calibration run.

For an API backend, install `.[api]`, set `LLM_SERVER` to the OpenAI-compatible server URL, and use `--backend api`. Set `LLM_KEY` when the server requires authentication. `--tokenizer` identifies the matching released tokenizer when the served model uses an alias. The server must use the same tokenizer and chat template and report exact prompt and completion token counts. Attention measurements and reader-side calibration use the local backend.
