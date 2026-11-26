# Fault-Tolerant Browser Training — Design Doc

2026-11-25

## Overview

We will build a system where anyone opens a link, their browser tab joins a training run on its own GPU through WebGPU, and training still converges while tabs randomly crash, close, join late, or run slowly. Version 1 ships in 2 weeks on a small model (MNIST), with charts proving accuracy holds up under failure.

**Problem.** Browser volunteer training was explored in 2018–2021 (Pando, JSDoop) on WebGL or plain JavaScript, which were too slow for ML. WebGPU now ships in all major browsers, but no current project combines it with training that survives workers disappearing. A 2026 survey still lists fault tolerance under open, changing membership as unsolved.

**Goals**

1. Train a model across many browser tabs, coordinated by one server.
2. Keep training correct when any worker dies at any moment: no lost work, no corrupted model.
3. Measure and chart how accuracy and speed change as the failure rate rises.
4. Make every design decision explainable, with the rejected alternative.

**Non-goals for version 1:** large models, malicious workers, privacy guarantees, coordinator failover, real internet-scale deployment.

**Success criteria**

- With 30% of workers randomly killed, final test accuracy is within 2 percentage points of the single-tab baseline.
- No training round ever stalls permanently, whatever workers fail.
- A README with at least two charts, a design section, and a limitations section.

## Background and prior art

The niche is quiet: browser training work stopped around 2021, and active decentralized-training efforts target native GPUs, not browser tabs.

| Work | Year | What it did | Gap we address |
|---|---|---|---|
| [Pando](https://arxiv.org/pdf/1803.08426) | 2018 | Volunteer computing in browsers | Pre-WebGPU; general compute, not ML training |
| [JSDoop + TensorFlow.js](https://arxiv.org/pdf/1910.07402) | 2019 | Volunteer neural network training in browsers | WebGL-era speed; limited failure handling |
| [Volunteer computing-like deep learning](https://arxiv.org/pdf/2103.08894) | 2021 | Deep learning on volunteer machines | Not browser-based |
| [Distributed Deep Learning in Open Collaborations](https://arxiv.org/pdf/2106.10207) (Hivemind) | 2021 | Volunteers training together over the internet | Native code, not browsers |
| [OpenDiLoCo](https://primeintellect.ai/blog/opendiloco) (Prime Intellect) | 2024 | Low-communication training across data centres | Real GPUs, managed machines |
| [Edge LLM training survey](https://arxiv.org/pdf/2610.32215) | 2026 | Lists open problems | Names fault tolerance under changing membership as unsolved |

**Why now:** [WebGPU ships in all major browsers](https://web.dev/blog/webgpu-supported-major-browsers), giving tabs real GPU compute. Read the JSDoop and Pando papers before day 2 and note what they measured.

## Scope

Version 1 is the fault-tolerance core plus the evidence; everything else is cut or deferred.

| Area | Version 1 (2 weeks) | Deferred |
|---|---|---|
| Model and data | Small multilayer perceptron (784 → 128 → 10) on MNIST | Small convolutional network; tiny character-level transformer |
| Browser compute | TensorFlow.js with its WebGPU backend | Own WebGPU kernels (matrix multiply, ReLU, softmax, backward passes) |
| Coordinator | Single Go server, in-memory state, checkpoints to disk | Replicated coordinator |
| Failure handling | Heartbeats, leases, reassignment, stale-update handling, worker rejoin | Malicious workers (Byzantine robustness) |
| Testing | Playwright chaos harness on one machine | Real volunteers over the internet |
| Dashboard | One page: worker grid plus live accuracy chart | Public demo link |
| Experiments | Failure rate, worker count, drop vs. down-weight | Compression, adaptive round timing |

**Explicitly out of scope:** privacy guarantees, incentives for volunteers, authentication, and anything larger than a few million parameters.

**Stretch goals, in priority order:** (1) own WebGPU kernels, (2) public demo link where friends join from phones and laptops, (3) tiny transformer, (4) replicated coordinator.

## Architecture

Three browser-side parts talk to one Go coordinator; inside it, a single state loop owns every version, lease, and round.

_[Architecture diagram: see the original design doc.]_

_system architecture · browser parts, coordinator internals, run logs_

Tabs reach the coordinator only through WebSockets, and only the state loop changes state, so a tab dying mid-message can never leave the model half-updated.

## Training protocol

Training runs in synchronous rounds of local SGD (stochastic gradient descent): each worker trains on its own slice of data, then the coordinator averages the weight changes it receives.

**Model and data**

- Multilayer perceptron: 784 inputs → 128 (ReLU) → 10 (softmax). 101,770 parameters, about 407 KB as float32.
- MNIST: 60,000 training images split into 100 shards of 600; 10,000 test images kept on the coordinator.
- Weights travel as one flat Float32Array in a fixed layer order (W1, b1, W2, b2). Fixed random seed for initialization.

**One round (version r)**

1. The coordinator holds global weights for version r and assigns one shard to each idle worker, with a lease (see Fault tolerance).
2. Each worker downloads the version-r weights (skipped if already cached), trains one pass over its shard (batch size 32, about 19 steps, plain SGD, learning rate 0.05), and computes the delta: its local weights minus the version-r weights.
3. The worker sends back the delta, the version it started from, and its sample count.
4. The round closes when every assigned shard has returned, or when the round deadline passes (default 10 s) and at least a quorum of shards (default 50%) has returned.
5. The coordinator aggregates, bumps the version to r + 1, evaluates on the test set, checkpoints, and starts the next round.
6. Shards not returned by the close are reassigned in the next round. One epoch = all 100 shards returned once.

**Aggregation.** Weighted average of accepted deltas, where n is a worker's sample count and s is its staleness weight:

```math
W_{r+1} = W_r + \frac{\sum_i n_i \, s(\tau_i) \, \Delta_i}{\sum_i n_i \, s(\tau_i)}
```

**Staleness.** An update's staleness τ is the current version minus the version it started from. Updates that arrive after their round closed are buffered into the next aggregation. Two strategies, chosen by config flag, are compared in the experiments:

| Strategy | Rule | Expected trade-off |
|---|---|---|
| Drop | Accept only τ = 0; discard the rest, reassign the shard | Cleaner updates, more wasted work |
| Down-weight | s(τ) = 1 / (1 + τ); discard if τ > 2 | Uses more work, risks noisier updates |

**Evaluation.** The coordinator runs the forward pass in Go (gonum) on the 10,000 test images after every round and logs test accuracy and loss. This keeps the source of truth on the server, independent of any tab.

## Fault tolerance

Workers are disposable: any of them can vanish at any moment, and the coordinator never waits on one forever. Four mechanisms make that true.

**1. Heartbeats.** Each worker sends a heartbeat every 2 s over its WebSocket. A worker with no heartbeat for 6 s (3 missed) is marked dead, and its connection is closed. A closed WebSocket marks it dead immediately.

**2. Leases.** Every shard assignment is a lease with an ID, a shard, a base version, and an expiry (round deadline + 5 s grace). When a lease expires or its worker dies, the lease is revoked and the shard returns to the pending queue. Results are accepted only for a live lease, or as a late update under the staleness rules.

**3. Exactly-once results.** Each shard counts at most once per version. If a revoked lease's worker later returns a result for a shard that has already been reassigned and returned, the duplicate is discarded.

**4. Checkpoints.** After every aggregation, the coordinator writes weights, version, and shard state to disk: write to a temp file, fsync, then atomic rename, so a crash mid-write never corrupts the checkpoint. Restarting the coordinator resumes from the last checkpoint; workers reconnect automatically with backoff (1 s, 2 s, 4 s, capped at 10 s).

**Update sanity check.** Before aggregating, reject any delta containing NaN or infinity, or whose norm exceeds 10× the median of that round's accepted deltas. This catches bugs, not attackers.

**Failure matrix**

| Failure | Detected by | Response | Lost work |
|---|---|---|---|
| Tab closed or crashes mid-training | WebSocket close, or 6 s heartbeat timeout | Revoke lease, requeue shard | That shard's training |
| Tab is slow (weak GPU, throttled) | Lease expiry | Requeue shard; late result follows staleness rules | None if accepted late |
| Tab disconnects briefly, then returns | Reconnect | Joins as a new worker; old lease already revoked | That shard's training |
| Worker sends NaN or a huge delta | Sanity check | Discard, requeue shard, log it | That shard's training |
| All workers die | No live workers | Round stays open; resumes when any worker connects | Nothing beyond open leases |
| Coordinator crashes | Workers lose connection | Manual restart resumes from checkpoint | At most one round |

**Timing defaults** (all config flags): heartbeat 2 s, dead after 6 s, round deadline 10 s, lease grace 5 s, quorum 50%, max staleness 2.

## Coordinator design (Go)

One goroutine owns all training state and processes events one at a time, in the style of an Erlang process with a mailbox. Connection goroutines never touch state directly; they only send events on a channel. This removes locks and makes every state change easy to log and replay.

**Goroutines**

- **State loop** (one): reads from a single `events` channel and a 1 s ticker; owns everything below.
- **Per worker, reader and writer** (two each): the reader parses WebSocket frames into events; the writer drains that worker's outbound channel.
- **HTTP server:** serves the worker page, dashboard, and MNIST shards.

**Core state**

```go
type Coordinator struct {
    version   int            // global model version
    weights   []float32      // flat, fixed layer order
    workers   map[string]*Worker
    leases    map[string]*Lease  // by lease ID
    pending   []int          // shard IDs waiting for a worker
    round     *Round         // open round: assigned and returned leases, deadline
    lateBuf   []Update       // late updates for the next aggregation
    cfg       Config
}

type Worker struct {
    id        string
    lastBeat  time.Time
    lease     *Lease         // nil when idle
    out       chan []byte    // outbound frames
}

type Lease struct {
    id, workerID string
    shard, baseVersion int
    expires   time.Time
}
```

**Events handled by the state loop:** worker joined, heartbeat, update received, worker disconnected, tick. On each tick: mark dead workers, revoke expired leases, close the round if its conditions are met, and assign pending shards to idle workers.

**HTTP and WebSocket endpoints**

| Endpoint | Purpose |
|---|---|
| `GET /` | Worker page (opening it joins the run) |
| `GET /dashboard` | Dashboard page |
| `GET /shards/{id}` | One MNIST shard: 600 images as uint8 plus labels, binary |
| `WS /ws/worker` | Worker protocol |
| `WS /ws/dashboard` | Live status and metrics stream |

**Libraries:** `github.com/coder/websocket` for WebSockets, `gonum.org/v1/gonum/mat` for test-set evaluation.

**Logs:** every event and every round's metrics go to a JSON Lines file (`runs/<run-id>/events.jsonl`, `metrics.jsonl`). The experiment scripts read only these files, so every chart can be regenerated from a run.

## Browser worker design

The worker is stateless: it holds nothing the coordinator cannot rebuild, so killing it at any point loses at most one shard of work.

**Stack:** TypeScript, Vite, TensorFlow.js with `@tensorflow/tfjs-backend-webgpu`. On startup the worker calls `tf.setBackend('webgpu')`; if WebGPU is unavailable it falls back to WebGL and reports its backend to the coordinator, so experiments can filter by it.

**Lifecycle**

1. Connect to `/ws/worker`, send `hello` with backend and a self-reported device label. Receive a worker ID.
2. Start the heartbeat timer (every 2 s).
3. On `assign`: fetch the shard (cached in memory after first fetch), load weights for the base version (reuse if cached, else request them), train one pass over the shard, compute the delta, send `result`.
4. Go back to waiting for the next `assign`.
5. On disconnect: stop work, drop the in-progress shard, reconnect with backoff, and start again at step 1.

**Training step** (one shard): build the model once with fixed layer shapes; per assignment, set weights from the flat array; run `model.fit` with batch size 32, 1 epoch, SGD learning rate 0.05; read weights back to a flat Float32Array; delta = local minus base.

**Memory hygiene.** Wrap per-step work in `tf.tidy` and dispose tensors after each assignment. Log `tf.memory().numTensors` in each heartbeat so the dashboard shows a leak as a rising line.

**Chaos hooks** (used only by the harness): a `?slow=<ms>` URL parameter adds a delay per training step to simulate a weak device; a `?worker=<label>` parameter names the tab in logs.

## Message protocol

Control messages are JSON text frames; weights and deltas are binary frames, since sending 101,770 floats as JSON would be about 5× larger and slow to parse.

**JSON messages** (every message has a `type` field)

| Type | Direction | Fields |
|---|---|---|
| `hello` | Worker → coordinator | `backend` (webgpu or webgl), `label` |
| `welcome` | Coordinator → worker | `workerId`, `heartbeatMs`, `config` (batch size, learning rate) |
| `heartbeat` | Worker → coordinator | `numTensors`, `busy` |
| `assign` | Coordinator → worker | `leaseId`, `shard`, `baseVersion`, `expiresAt` |
| `get_weights` | Worker → coordinator | `version` |
| `revoke` | Coordinator → worker | `leaseId` (stop work on it) |
| `error` | Either | `code`, `message` |

**Binary frames** (little-endian)

| Bytes | Field |
|---|---|
| 1 | Frame kind: 1 = weights, 2 = result |
| 4 | Version (uint32): the weights' version, or the base version of a result |
| 16 | Lease ID (UUID bytes; zeros for weights) |
| 4 | Sample count (uint32; zero for weights) |
| 4 × 101,770 | Float32 values: weights, or the delta for a result |

**Dashboard stream** (`/ws/dashboard`, JSON, about once per second): workers with ID, label, state (idle, training, dead) and current shard; round number, version, and returned/assigned counts; latest test accuracy and loss; totals for revoked leases, dropped updates, and late-accepted updates.

**Versioning rule:** any field addition is backward compatible; the worker ignores unknown fields.

## Chaos harness and experiments

The harness turns "it survives failures" from a claim into a measured result: it runs the system unattended, kills tabs on a schedule, and every run leaves logs that regenerate its charts.

**Harness** (TypeScript, Playwright with Chromium)

- Launch the coordinator with a run ID and config, then open N worker tabs, each in its own browser context.
- **Kill model:** every 10 s, each live tab is closed with probability p/6, so about p of the pool is killed per minute. A replacement tab opens 5 s later, keeping the pool near N.
- **Slow model:** a chosen fraction of tabs opens with `?slow=<ms>` to simulate weak devices.
- Stop after a fixed number of epochs (default 5), then save the config next to the logs.
- Launch Chromium with WebGPU enabled (`--enable-unsafe-webgpu`). If headless mode cannot get a real GPU, run headed; record the backend in every run either way.

**Experiments** (3 seeds each; report mean and range)

| ID | Question | Setup | Main metric |
|---|---|---|---|
| E1 | What is the baseline? | 1 tab, no failures | Test accuracy after 5 epochs |
| E2 | Do more tabs train faster? | N = 1, 5, 10, 20, no failures | Wall-clock time to 95% test accuracy |
| E3 | Does training survive failures? | N = 10, kill rate 0%, 10%, 30%, 50% per minute | Final accuracy vs. E1; time to 95% |
| E4 | Drop or down-weight stale updates? | N = 10, 30% kill rate, both strategies | Final accuracy; wasted shards |
| E5 | What do slow workers cost? | N = 10, 30% of tabs slowed 3× | Time to 95%; late updates accepted |

**Charts for the README**

1. Test accuracy vs. epoch, one line per kill rate (E3), with E1 as a dashed baseline.
2. Time to 95% accuracy vs. number of tabs (E2).
3. One run's accuracy over time with each worker death marked as a tick on the time axis.
4. Drop vs. down-weight: final accuracy and wasted shards side by side (E4).

**Scripts:** a Python script (`experiments/plot.py`, pandas + matplotlib) reads `metrics.jsonl` and `events.jsonl` from each run folder and writes PNGs to `docs/`. No result goes in the README unless a run folder backs it.

## Dashboard

One page that makes the demo video: workers dying on the left, accuracy still climbing on the right.

- **Stack:** React + Vite, Recharts for the line chart, fed by `/ws/dashboard`.
- **Worker grid:** one tile per worker showing its label and current shard. Green = training, grey = idle, red = dead (fades out after 10 s). A small badge marks WebGPU vs. WebGL.
- **Accuracy chart:** test accuracy per round, with a red tick on the time axis for each worker death.
- **Counters:** round, version, live workers, shards revoked, updates dropped, late updates accepted.
- **Out of scope:** controls (start, stop, kill buttons), history across runs, authentication.

**Demo video (60 s):** start 10 tabs, run the harness at a 30% kill rate, record the dashboard as tiles go red and accuracy keeps rising, end on the final accuracy next to the single-tab baseline.

## Repository structure and tech stack

One repository, four parts, one command to run a local demo.

```text
browser-train/
  coordinator/          Go: state loop, leases, aggregation, eval, checkpoints
    cmd/coordinator/    main.go, flags
    internal/state/     Coordinator, Worker, Lease, Round; event handling
    internal/protocol/  JSON messages, binary frame encode/decode
    internal/model/     flat weight layout, gonum forward pass for eval
    internal/data/      MNIST loading and sharding
  worker/               TypeScript + Vite + TensorFlow.js (WebGPU)
  dashboard/            React + Vite + Recharts
  harness/              TypeScript + Playwright chaos runs
  experiments/          plot.py, run configs
  runs/                 run logs (git-ignored, except runs used in the README)
  docs/                 charts, design notes
  Makefile              make demo, make test, make experiment E=E3
```

| Part | Language | Key libraries |
|---|---|---|
| Coordinator | Go | coder/websocket, gonum |
| Worker | TypeScript | TensorFlow.js, tfjs-backend-webgpu |
| Dashboard | TypeScript | React, Recharts |
| Harness | TypeScript | Playwright |
| Plots | Python | pandas, matplotlib |

**Tests that must exist:**

- Go unit tests for lease expiry, duplicate-result rejection, staleness weights, and aggregation math (weighted average of known deltas).
- A round-trip test that a weights frame encoded in Go decodes identically in TypeScript.
- An end-to-end smoke test: the harness runs 3 tabs for 1 epoch with one kill and asserts accuracy above 85%.

## Two-week timeline

The build runs in eight milestones over 14 days; if time runs short, stopping after day 9 still leaves a complete, honest project.

| Days | Milestone | Done when |
|---|---|---|
| 1 | Write the design doc yourself | You explain the whole system aloud, no notes |
| 2–3 | One tab trains MNIST on WebGPU | About 95% test accuracy in a single tab |
| 4–5 | Coordinator plus multiple tabs | 5 tabs match the single-tab accuracy |
| 6–7 | Heartbeats, leases, staleness | Close a tab mid-round and training goes on |
| 8–9 | Chaos harness | Runs unattended, kills and accuracy logged |
| 10–11 | Experiments E1 to E5 | All four README charts generated |
| 12 | Dashboard and demo video | A 60-second video of tabs dying, accuracy rising |
| 13–14 | README, write-up, pinned repo | A stranger understands it in 2 minutes |

The 1-week version ends after day 9.

Each milestone starts only when the previous one meets its "done" line. The 1-week version is days 1 to 9 plus a basic README with one chart of accuracy vs. kill rate.

## Key decisions and rejected alternatives

These are the choices an interviewer will probe; know each reason well enough to defend it without notes.

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Training scheme | Synchronous rounds of local SGD, averaging weight deltas | Fully asynchronous parameter server (apply each gradient on arrival) | Rounds make versions and staleness explicit and easy to reason about; local steps cut messages per shard from about 19 to 1 |
| What workers send | Weight delta after a full shard | Gradients after every batch | About 19× less traffic; tolerates slow connections |
| Failure detection | Heartbeats plus leases with expiry | Waiting on TCP or WebSocket close alone | A frozen tab can keep its socket open; only a timeout catches it |
| Stragglers | Round closes at deadline once a quorum returns | Wait for every worker | One slow or dead tab would stall everyone |
| Late updates | Compare drop vs. down-weight experimentally | Pick one up front | This is the project's main open question; measuring it is the contribution |
| Coordinator concurrency | One state goroutine with an event channel | Shared state with mutexes | No lock ordering bugs; every change is one logged event |
| Browser compute (v1) | TensorFlow.js WebGPU backend | Own kernels from day 1 | Novelty is fault tolerance; own kernels are stretch goal 1 |
| Wire format | JSON for control, binary for weights | All JSON | 407 KB of floats per message; JSON is about 5× larger |
| Evaluation | Coordinator in Go | A dedicated evaluator tab | The source of truth must not depend on a killable worker |
| Coordinator failure | Checkpoint + manual restart | Replicated coordinator (Raft) | Doubles the scope; stated as a limitation |

## Limitations and risks

State these in the README; interviewers trust a project more when it names its own limits.

**Known limitations (by design)**

- The coordinator is a single point of failure; a crash pauses training until a manual restart.
- No defence against malicious workers; the sanity check only catches bugs.
- No privacy guarantee; workers see the data shards they are sent.
- Tested on one machine, not across real networks with real latency.
- Proven only on a 100K-parameter model; larger models hit browser memory limits.

**Risks to the 2-week plan**

| Risk | Effect | Fallback |
|---|---|---|
| Headless Chromium cannot use WebGPU | Harness runs on WebGL or CPU | Run headed, or accept WebGL and note it; fault-tolerance results still hold |
| Many tabs on one GPU contend for memory | Tabs crash or slow down | Cap N at what the machine handles; report that limit |
| Averaging diverges at high kill rates | Accuracy collapses | Lower the learning rate, raise the quorum; report the failure honestly |
| TensorFlow.js memory leak | Tabs slow over time | Track `numTensors` in heartbeats; fix with `tf.tidy` and `dispose` |
| Timeline slips | No experiments by day 14 | Cut to the 1-week version: days 1–9 plus one chart of accuracy vs. kill rate |

**Interview readiness check:** before calling it done, explain out loud, without notes, (1) what happens step by step when a tab dies mid-shard, (2) why rounds instead of fully asynchronous updates, and (3) what you would change to support a 100-million-parameter model.
