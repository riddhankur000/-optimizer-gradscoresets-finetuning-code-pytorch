# CoLM Architecture & Data Selection Flow - Visual Guide

## 1. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MAIN ENTRY POINTS                              │
├─────────────────┬──────────────────────────────────────────────┬────────┤
│  train.py       │  train_multitask.py                          │ run.py │
│  (Single task)  │  (Multi-task with YAML config)              │        │
└────────┬────────┴──────────────────────┬───────────────────────┴────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
              ┌──────────▼─────────────┐
              │   ArgumentParser       │
              │ (HfArgumentParser)     │
              ├───────────────────────┤
              │ ModelArguments        │
              │ DataArguments         │
              │ TrainingArguments     │
              └────────┬──────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐  ┌─────▼───┐  ┌──────▼─────────┐
   │ Model   │  │Tokenizer│  │Dataset         │
   │Loading  │  │Loading  │  │get_training_   │
   │         │  │         │  │dataset()       │
   │+ LoRA   │  │+ Padding│  │                │
   └────┬────┘  └────┬────┘  └────┬───────────┘
        │            │             │
        │     ┌──────┴─────────────┘
        │     │
        └─────┼──────────────────────────────┐
              │                              │
         ┌────▼─────────────────────────┐   │
         │    Trainer Class Selection    │   │
         ├───────────────────────────────┤   │
         │                               │   │
         │  data_selection = "none"      │   │
         │  ↓ Standard HF Trainer        │   │
         │                               │   │
         │  data_selection = "greats"    │   │
         │  ↓ SubsetTrainer              │   │
         │    (rich selection)           │   │
         │                               │   │
         │  data_selection = "fairot"    │   │
         │  ↓ SubsetTrainer              │   │
         │                               │   │
         │  data_selection = "spot"      │   │
         │  ↓ SubsetTrainer              │   │
         │                               │   │
         │  efficient_mezo = true        │   │
         │  ↓ SubsetTrainerEfficient     │   │
         └────┬─────────────────────────┘   │
              │                             │
              └──────────┬──────────────────┘
                         │
              ┌──────────▼─────────────────┐
              │  trainer.train()           │
              │  _inner_training_loop()    │
              └──────────┬─────────────────┘
                         │
         ┌───────────────┴────────────────────┐
         │  [See data selection loop below]   │
         └────────────────────────────────────┘
```

---

## 2. SUBSET TRAINER DATA SELECTION LOOP (DETAILED)

### Simplified Flow

```
┌─ EPOCH LOOP ──────────────────────────────────────────────────────┐
│                                                                     │
│  ┌─ BATCH LOOP ────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  total_batched_samples = 0                                   │  │
│  │  total_reps = []                                             │  │
│  │  input_list = []                                             │  │
│  │                                                               │  │
│  │  FOR batch_idx, inputs IN enumerate(dataloader):             │  │
│  │                                                               │  │
│  │    total_batched_samples += 1                                │  │
│  │                                                               │  │
│  │    ┌─── ACCUMULATION CHECK ────────────────────────┐        │  │
│  │    │ if (total_batched_samples % grad_acc_steps != 0)      │  │
│  │    │   AND not last_batch:                        │        │  │
│  │    │                                               │        │  │
│  │    │    [ACCUMULATE PHASE]                         │        │  │
│  │    │    rep = save_select(model, inputs)           │        │  │
│  │    │         ↑                                      │        │  │
│  │    │         └─ Extract feature vector             │        │  │
│  │    │            (hidden_state or MeZO grad)       │        │  │
│  │    │                                               │        │  │
│  │    │    total_reps.append(rep)  # Shape: [hidden] │        │  │
│  │    │    input_list.append(inputs)                  │        │  │
│  │    │    continue  # Skip to next batch             │        │  │
│  │    │                                               │        │  │
│  │    └───────────────────────────────────────────────┘        │  │
│  │                                                               │  │
│  │    ELSE: [FULL BATCH REACHED - PROCEED TO SELECTION]        │  │
│  │                                                               │  │
│  │    ┌─── COLLECTION PHASE ──────────────────────────┐        │  │
│  │    │ rep = save_select(model, inputs)              │        │  │
│  │    │ total_reps.append(rep)                        │        │  │
│  │    │                                                │        │  │
│  │    │ total_reps now contains grad_acc_steps reps   │        │  │
│  │    │ input_list contains grad_acc_steps inputs     │        │  │
│  │    │                                                │        │  │
│  │    │ Shape of total_reps:                           │        │  │
│  │    │   - If rep: [grad_acc_steps, hidden_dim]     │        │  │
│  │    │   - If mezo: [grad_acc_steps, param_dim]     │        │  │
│  │    │                                                │        │  │
│  │    └────────────────────────────────────────────────┘        │  │
│  │                                                               │  │
│  │    ┌─── FILTERING PHASE ────────────────────────────┐       │  │
│  │    │ Filter out NaN and zero-norm representations  │       │  │
│  │    │ filtered_reps = [rep for rep if not isnan]   │       │  │
│  │    │ filtered_inputs = corresponding inputs        │       │  │
│  │    └────────────────────────────────────────────────┘       │  │
│  │                                                               │  │
│  │    ┌─── DISTRIBUTED GATHERING (DDP) ──────────────┐        │  │
│  │    │ On rank r: send {rank: total_reps}           │        │  │
│  │    │ dist.all_gather_object() → gathered_reps     │        │  │
│  │    │                                                │        │  │
│  │    │ Result: all_reps_rank0 = [                    │        │  │
│  │    │   cat(rank_0_reps),                          │        │  │
│  │    │   cat(rank_1_reps),                          │        │  │
│  │    │   ...                                         │        │  │
│  │    │ ]                                             │        │  │
│  │    │ Shape: [grad_acc_steps * world_size, dim]   │        │  │
│  │    └────────────────────────────────────────────────┘       │  │
│  │                                                               │  │
│  │    IF rank == 0:  [SELECTION ON RANK 0 ONLY]               │  │
│  │                                                               │  │
│  │      ┌─── NORMALIZATION PHASE ───────────────────┐         │  │
│  │      │ all_reps:                                 │         │  │
│  │      │   - Shape [B, feature_dim]               │         │  │
│  │      │                                            │         │  │
│  │      │ Apply transform:                          │         │  │
│  │      │   mezo_transform="self_normalize":        │         │  │
│  │      │     all_reps = all_reps /                 │         │  │
│  │      │       norm(all_reps, dim=1, keepdim=True)│         │  │
│  │      │                                            │         │  │
│  │      │   mezo_transform="clip_full":             │         │  │
│  │      │     if norm(all_reps) > max_grad_norm:    │         │  │
│  │      │       all_reps *= max_grad_norm/norm     │         │  │
│  │      │                                            │         │  │
│  │      └────────────────────────────────────────────┘        │  │
│  │                                                               │  │
│  │      ┌─ FEATURE MASKING PHASE ────────────────────┐        │  │
│  │      │ if mezo_topk == "random":                  │        │  │
│  │      │   Select zo_dim random dimensions         │        │  │
│  │      │   all_reps = all_reps[:, random_indices]  │        │  │
│  │      │                                             │        │  │
│  │      │ else: call select_masking()                │        │  │
│  │      │   Applies per-source masking if needed     │        │  │
│  │      └─────────────────────────────────────────────┘       │  │
│  │                                                               │  │
│  │      ┌──── CORE SELECTION ALGORITHM ──────────────┐        │  │
│  │      │ Call select_data():                        │        │  │
│  │      │                                             │        │  │
│  │      │ selected_idx, weights =                    │        │  │
│  │      │   select_data(all_reps, K, source_list)   │        │  │
│  │      │                                             │        │  │
│  │      │ Input: all_reps [B, masked_dim]           │        │  │
│  │      │ Output: [K indices], [K weights]          │        │  │
│  │      │                                             │        │  │
│  │      │ Methods (data_selection_method):           │        │  │
│  │      │   - "greats": grad-based coreset          │        │  │
│  │      │   - "fairot": fair outlier truncation    │        │  │
│  │      │   - "spot": SPOT greedy                  │        │  │
│  │      │   - "submodlib": facility location       │        │  │
│  │      └─────────────────────────────────────────────┘       │  │
│  │                                                               │  │
│  │      ┌───── INDEX MAPPING PHASE ──────────────────┐        │  │
│  │      │ Map selected K back to original indices    │        │  │
│  │      │ and combine with always-keep sources       │        │  │
│  │      │                                             │        │  │
│  │      │ selected_idx_final = [                     │        │  │
│  │      │   list_idx_keep +                          │        │  │
│  │      │   sampling_indices[selected_idx]           │        │  │
│  │      │ ]                                          │        │  │
│  │      └─────────────────────────────────────────────┘       │  │
│  │    END IF rank==0                                           │  │
│  │                                                               │  │
│  │    ┌─── BROADCAST PHASE (DDP) ─────────────────────────────┐ │  │
│  │    │ dist.broadcast(selected_idx_tensor, src=0)            │ │  │
│  │    │ dist.broadcast(selected_weights_tensor, src=0)        │ │  │
│  │    │                                                         │ │  │
│  │    │ Now all ranks have same selected_idx & weights         │ │  │
│  │    └─────────────────────────────────────────────────────────┘ │  │
│  │                                                               │  │
│  │    ┌─── SHARD BY RANK ──────────────────────────────┐       │  │
│  │    │ Each rank r processes its shard:              │       │  │
│  │    │   rank_r_slice = selected_inputs[             │       │  │
│  │    │     r*train_on_each : (r+1)*train_on_each     │       │  │
│  │    │   ]                                            │       │  │
│  │    │                                                │       │  │
│  │    │ train_on_each = grad_acc_steps * small_ratio  │       │  │
│  │    │                                                │       │  │
│  │    └────────────────────────────────────────────────┘       │  │
│  │                                                               │  │
│  │    ┌─── TRAINING PHASE (PER RANK) ─────────────────┐       │  │
│  │    │ FOR inner_step, selected_input IN             │       │  │
│  │    │     enumerate(rank_r_slice):                  │       │  │
│  │    │                                                │       │  │
│  │    │   with accelerator.accumulate(model):         │       │  │
│  │    │     loss_step = training_step(                │       │  │
│  │    │       model,                                  │       │  │
│  │    │       selected_input,                         │       │  │
│  │    │       weight=selected_weights[inner_step]     │       │  │
│  │    │     )                                         │       │  │
│  │    │                                                │       │  │
│  │    │     tr_loss += loss_step                      │       │  │
│  │    │                                                │       │  │
│  │    │ optimizer.step()                              │       │  │
│  │    │ lr_scheduler.step()                           │       │  │
│  │    │ model.zero_grad()                             │       │  │
│  │    │                                                │       │  │
│  │    └────────────────────────────────────────────────┘       │  │
│  │                                                               │  │
│  │    Reset for next accumulation:                             │  │
│  │    total_reps = []                                          │  │
│  │    input_list = []                                          │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────┘  │  │
│                                                                  │  │
│  [Periodic evaluation & logging]                                │  │
│                                                                  │  │
└──────────────────────────────────────────────────────────────────┘  │
```

---

## 3. REPRESENTATION EXTRACTION (`save_select()`)

```
┌─ save_select(model, inputs) ──────────────────────────────┐
│                                                             │
│ data_selection_unit?                                       │
│                                                             │
├─ "rep" (Hidden states) ────────────────────────────────┐  │
│ │ hidden_states = model(input_ids,                     │  │
│ │                   output_hidden_states=True)         │  │
│ │ res = hidden_states[-1][batch_indices, last_pos]    │  │
│ │ SHAPE: [batch_size, hidden_dim]                      │  │
│ └─────────────────────────────────────────────────────┘  │
│                                                             │
├─ "mezo" (Zeroth-order gradient) ──────────────────────┐  │
│ │ # Compute loss at θ + ε*z                            │  │
│ │ z ~ N(0, 1)                                          │  │
│ │ θ_perturb1 = θ + ε*z                                │  │
│ │ loss1 = forward(model at θ_perturb1)                │  │
│ │                                                      │  │
│ │ # Compute loss at θ - ε*z                            │  │
│ │ θ_perturb2 = θ - ε*z                                │  │
│ │ loss2 = forward(model at θ_perturb2)                │  │
│ │                                                      │  │
│ │ # Finite difference gradient                         │  │
│ │ grad_hat = (loss1 - loss2) / (2ε)                   │  │
│ │                                                      │  │
│ │ # For each parameter, compute grad_hat * z           │  │
│ │ res = [grad_hat * z_i for each param i]             │  │
│ │ SHAPE: [total_params,]                               │  │
│ │      or [total_params * batch_size,]                │  │
│ └─────────────────────────────────────────────────────┘  │
│                                                             │
├─ "masked_grad" (Full backprop) ───────────────────────┐  │
│ │ Freeze non-target layers (requires_grad = False)    │  │
│ │ loss = training_step(model, inputs)  # backward()   │  │
│ │ grads = [param.grad.flatten()                        │  │
│ │          for param in target_params]                │  │
│ │ res = concatenate(grads)                             │  │
│ │ SHAPE: [target_param_dim,]                           │  │
│ │      or [target_param_dim * batch_size,]            │  │
│ └─────────────────────────────────────────────────────┘  │
│                                                             │
├─ "completion_length" (Sequence length) ──────────────┐  │
│ │ res = inputs["completion_lengths"][0]                │  │
│ │ SHAPE: scalar (int)                                  │  │
│ └─────────────────────────────────────────────────────┘  │
│                                                             │
└─ "length_loss_weighted" (Loss × length) ──────────────┐  │
  │ completion_len = max(attention_mask.sum(dim=1))    │  │
  │ loss = model(inputs).loss                          │  │
  │ res = completion_len * loss / 10                   │  │
  │ SHAPE: scalar (float)                              │  │
  └──────────────────────────────────────────────────────┘  │
```

---

## 4. SELECTION ALGORITHMS (`select_data()`)

```
┌─ select_data(all_reps, max_samples, source_list) ─────────────┐
│                                                                 │
│ Input:  all_reps [B, feature_dim]                             │
│         source_list [B] (optional, for source-wise selection)  │
│ Output: selected_idx [K], weights [K]                         │
│                                                                 │
│ Method: data_selection_method                                  │
│                                                                 │
├─ "greats" (Gradient-based coreset selection) ──────────────┐  │
│ │ # Diversity: similarity within all samples                 │  │
│ │ sims = compute_cost_matrix(all_reps, all_reps)            │  │
│ │                                                             │  │
│ │ # Representativeness: distance to small eval set           │  │
│ │ eval_reps = sample_k_random_items(2)                      │  │
│ │ sims_cross = compute_cost_matrix(all_reps, eval_reps)     │  │
│ │                                                             │  │
│ │ # Greedy coreset selection                                │  │
│ │ idx = greats.greedy_selection(                             │  │
│ │   sims_cross.mean(1),  # representativeness              │  │
│ │   sims,                # diversity                        │  │
│ │   max_samples          # K                                │  │
│ │ )                                                          │  │
│ │ weights = ones(K)                                          │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                 │
├─ "fairot" (Fair Outlier Truncation) ──────────────────────┐  │
│ │ # Compute pairwise similarities                           │  │
│ │ dist, sims = compute_cost_matrix(all_reps, all_reps)      │  │
│ │                                                            │  │
│ │ # Fair greedy outlier truncation                          │  │
│ │ idx = fairot2.greedy_fairot(                              │  │
│ │   sims,              # similarity matrix                  │  │
│ │   max_samples,       # K                                  │  │
│ │   dist=dist,         # distance matrix                    │  │
│ │   iters=500,         # optimization iterations            │  │
│ │   reg=1e-1           # regularization                     │  │
│ │ )                                                         │  │
│ │ weights = ones(K) / K  # uniform                         │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                 │
├─ "spot" (SPOT Greedy Subset Selection) ───────────────────┐  │
│ │ # Only uses distance matrix                               │  │
│ │ dist = compute_cost_matrix(all_reps, all_reps)            │  │
│ │ target_marginals = ones(B) / (some normalization)         │  │
│ │                                                            │  │
│ │ # SPOT algorithm                                          │  │
│ │ idx = SPOT_GreedySubsetSelection(                         │  │
│ │   dist, target_marginals, max_samples                     │  │
│ │ )                                                         │  │
│ │ weights = ones(K) / K  # uniform                         │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                 │
├─ "submodlib" / "weightedsubmodlib" ─────────────────────────┐  │
│ │ # Facility location-based (submodular)                     │  │
│ │ greedy_indices = get_orders_and_weights(                   │  │
│ │   max_samples,                                             │  │
│ │   all_reps,                                                │  │
│ │   metric=facility_similarity,  # ["cosine","euclidean"]  │  │
│ │   y=source_list,               # For source-wise strategy│  │
│ │   strategy=source_wise_selection  # ["none","proportional","balanced"]│
│ │ )                                                          │  │
│ │ idx, weights = greedy_indices                              │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                 │
├─ "fairot_multisource" ────────────────────────────────────┐  │
│ │ # Combines fairot with per-source balancing               │  │
│ │ lambda_func = lambda S, k, dist=None:                      │  │
│ │   fairot2.greedy_fairot(S, k, dist=dist, reg=1e-1, ...)  │  │
│ │                                                            │  │
│ │ idx, weights = select_data_facloc(                        │  │
│ │   all_reps,                                                │  │
│ │   max_samples,                                             │  │
│ │   source_list,                                             │  │
│ │   optim=lambda_func                                        │  │
│ │ )                                                          │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. WANDB METRIC LOGGING PIPELINE

```
┌─ Trainer Execution ──────────────────────────┐
│                                               │
│ For each step:                                │
│   if step % logging_steps == 0:               │
│     ├─ Compute loss metrics                   │
│     ├─ Call on_log() callbacks                │
│     └─ Broadcast to wandb                     │
│                                               │
│ For each eval step:                           │
│   if step % eval_steps == 0:                  │
│     ├─ compute_metrics() on eval set          │
│     ├─ Call on_evaluate() callbacks           │
│     └─ Log to wandb                           │
└───────────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼─────────────┐   ┌─────▼──────────────┐
    │ TrainOutput      │   │ MonitoringCallback │
    ├──────────────────┤   ├────────────────────┤
    │ train_loss       │   │ on_log():          │
    │ metrics = {      │   │  - Add perplexity  │
    │   loss           │   │  - Grad norms      │
    │   learning_rate  │   │  - GPU stats       │
    │   epoch          │   │  - CPU stats       │
    │ }                │   │                    │
    │                  │   │ on_evaluate():     │
    │ trainer.         │   │  - Eval perplexity │
    │ log_metrics()    │   │  - Per-task loss   │
    │ trainer.         │   │  - Resource use    │
    │ save_metrics()   │   │                    │
    └────┬─────────────┘   └────┬───────────────┘
         │                      │
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │      WandB          │
         ├─────────────────────┤
         │ Project: project    │
         │ Entity: entity      │
         │ Run name: run_name  │
         │                     │
         │ Logged:             │
         │ - Loss curves       │
         │ - Learning rate     │
         │ - Perplexity        │
         │ - Grad norms        │
         │ - GPU memory        │
         │ - CPU usage         │
         │ - Per-task metrics  │
         └─────────────────────┘
```

---

## 6. PARAMETER IMPORTANCE FOR DATA SELECTION

```
┌─────────────────────────────────────────────────────┐
│         KEY SELECTION PARAMETERS                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 1. data_selection_method                           │
│    Values: none, greats, fairot, spot, submodlib  │
│    Effect: Which algorithm to use                  │
│    Default: "none" (no selection)                  │
│                                                     │
│ 2. data_selection_unit                             │
│    Values: rep, mezo, masked_grad, ...             │
│    Effect: WHAT to select (which representation)   │
│    Default: "mezo"                                 │
│                                                     │
│ 3. small_batch_ratio                               │
│    Values: 0.0 - 1.0                               │
│    Effect: K/B select ratio (0.1 = select 10%)    │
│    Default: 1.0 (no reduction)                     │
│    *** CRITICAL: Controls reduction amount ***     │
│                                                     │
│ 4. gradient_accumulation_steps                      │
│    Values: positive integer                        │
│    Effect: B batch size (accum buffer size)        │
│    Default: 1                                      │
│    *** SETS B: K = B × small_batch_ratio ***      │
│                                                     │
│ 5. mezo_eps / mezo_topk / mezo_optim                │
│    Effect: Customize MeZO representation           │
│    When: data_selection_unit="mezo"                │
│                                                     │
│ 6. facility_similarity                             │
│    Values: cosine, euclidean, l1                   │
│    Effect: Distance metric for facility location   │
│    When: data_selection_method=submodlib           │
│                                                     │
│ 7. source_wise_selection                           │
│    Values: none, proportional, balanced            │
│    Effect: Balance samples across sources/tasks    │
│    Requires: source_list in inputs                 │
│                                                     │
│ 8. keep_sources                                     │
│    Values: "0_1_3_5_7..."  (underscore-sep)       │
│    Effect: Always keep certain sources (no remove) │
│    Use: Ensure important tasks always trained      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 7. SEQUENTIAL TASK TRAINING INJECTION POINTS

```
┌─────────────────────────────────────────────────────────────┐
│            INJECTION POINT 1: Task-Aware Selection           │
│                                                              │
│ Location: select_data() [line 1358]                          │
│                                                              │
│ Current: SelectsK from all B samples                         │
│ Modified: Select K only from current task                    │
│                                                              │
│ Implementation:                                              │
│   source_list now contains task IDs                          │
│   current_task = state.global_step // steps_per_task       │
│   task_mask = (source_list == current_task)                │
│   task_reps = all_reps[task_mask]                          │
│   task_idx, task_weights = algorithm(task_reps, K)         │
│   return task_idx[task_mask], task_weights                 │
└─────────────────────────────────────────────────────────────┘
                            │
┌────────────────────────────▼────────────────────────────────┐
│         INJECTION POINT 2: Per-Task Eval Loss Tracking       │
│                                                              │
│ Location: MonitoringCallback.on_evaluate() [line 170]       │
│                                                              │
│ Current: Logs aggregate eval loss                           │
│ Modified: Log per-task eval loss                            │
│                                                              │
│ Implementation:                                              │
│   for task in unique(eval_dataset['task']):               │
│     task_indices = where(eval['task'] == task)            │
│     task_eval_set = eval_dataset.select(task_indices)     │
│     task_metrics = trainer.evaluate(task_eval_set)        │
│     metrics[f'eval_loss_{task}'] = task_metrics['loss']  │
│     metrics[f'eval_perp_{task}'] = exp(metrics['loss'])   │
└─────────────────────────────────────────────────────────────┘
                            │
┌────────────────────────────▼────────────────────────────────┐
│      INJECTION POINT 3: Sequential Training Wrapper          │
│                                                              │
│ Location: main() function [line 400+]                       │
│                                                              │
│ Current: Mix all tasks in training dataset                  │
│ Modified: Train 1 task at a time sequentially              │
│                                                              │
│ Implementation:                                              │
│   if training_args.sequential_training:                   │
│     tasks = sorted(unique(train_dataset['task']))         │
│     for task in tasks:                                   │
│       task_dataset = train_dataset.filter(task)           │
│       trainer.train_dataset = task_dataset                │
│       task_result = trainer.train(...)                    │
│       log(f"Task {task}: loss={result.loss}")             │
│   else:                                                   │
│     # Normal mixed training                              │
│     train_result = trainer.train(...)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. LOSS TRACKING ARCHITECTURE

```
┌─ Training Step ─────────────────────────────────────────┐
│                                                          │
│  loss = compute_loss(model, inputs)                    │
│        │                                                │
│        ├─ Returned by model.forward()                  │
│        ├─ Applies label smoothing if configured        │
│        └─ Shape: scalar (no reduction across batch)   │
│                                                         │
└────────┬───────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────┐
    │  Every gradient_accumulation_steps:              │
    │                                                  │
    │  tr_loss += loss_step                           │
    │  optimizer.step()                              │
    │  lr_scheduler.step()                           │
    │  state.global_step += 1                        │
    └────┬─────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────┐
    │ Every logging_steps:                          │
    │                                               │
    │ train_loss = tr_loss / steps_since_log        │
    │ metrics = {loss: train_loss, ...}             │
    │ Call callbacks: on_log()                      │
    │  ├─ Compute perplexity = exp(train_loss)     │
    │  ├─ Compute grad norms                       │
    │  └─ Gather GPU/CPU stats                     │
    │ log_metrics("train", metrics)                │
    └────┬──────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────┐
    │ Every eval_steps:                             │
    │                                               │
    │ eval_metrics = evaluate()                     │
    │  ├─ Run model.eval() on eval_dataset          │
    │  └─ Compute eval_loss                         │
    │ Call callbacks: on_evaluate()                 │
    │  ├─ Compute eval_perplexity                   │
    │  ├─ Per-task eval loss [INJECTION POINT 2]   │
    │  └─ Gather stats                             │
    │ log_metrics("eval", eval_metrics)            │
    └────┬──────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────────┐
    │ WandB Dashboard:                              │
    │                                               │
    │ Training Curves:                              │
    │ - loss (smoothed)                             │
    │ - train_perplexity                            │
    │ - learning_rate                               │
    │ - grad_norm                                   │
    │                                               │
    │ Evaluation Curves:                            │
    │ - eval_loss (at eval steps)                   │
    │ - eval_perplexity                             │
    │ - eval_loss_task1                    [NEW]   │
    │ - eval_loss_task2                    [NEW]   │
    │ - eval_loss_task3                    [NEW]   │
    │                                               │
    │ Resource Curves:                              │
    │ - gpu_memory_used_gb                          │
    │ - gpu_memory_util_%                           │
    │ - cpu_percent                                 │
    └────────────────────────────────────────────┘
```

