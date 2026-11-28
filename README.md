# Fault-Tolerant Browser Training

Open a link, and your browser tab joins a training run on its own GPU through WebGPU. Training keeps converging while tabs crash, close, join late, or run slowly.

Version 1 trains a small multilayer perceptron (784 → 128 → 10) on MNIST across many tabs, coordinated by one Go server, and measures how accuracy holds up as the failure rate rises.

> **Status:** milestone 2 of 8. A single tab trains MNIST end to end: the coordinator loads and shards
> MNIST and serves it over HTTP, and the worker decodes shards, trains the model on its GPU and
> evaluates on the test set. The weight layout, binary wire format and aggregation math are
> implemented and tested too. What is left is stubbed with `TODO(milestone N)` markers that match the
> [build plan](docs/design.md#two-week-timeline): the WebSocket protocol and state loop, leases and
> heartbeats, the chaos harness, the experiments and the dashboard.
>
> Measured so far: **96.73% test accuracy after 5 epochs** (60,000 images per epoch, batch 32,
> SGD 0.05), with the tensor count flat across epochs. That run used the CPU backend under Node
> against a live coordinator, not a browser tab on WebGPU, so it stands for the data path, weight
> layout and hyperparameters only. `worker/src/mnist.e2e.test.ts` reproduces it.

## How it works

- **Synchronous rounds of local SGD (stochastic gradient descent).** Each worker trains one pass over a 600-image shard and sends back a weight delta. The coordinator averages the deltas, weighted by sample count and staleness.
- **Workers are disposable.** Heartbeats every 2 s, leases with expiry, and exactly-once results per shard mean a tab can die at any moment without losing more than its own shard or corrupting the model.
- **Rounds never stall.** A round closes once every shard returns, or at its deadline once a quorum has returned. Late updates are either dropped or down-weighted (`s(τ) = 1 / (1 + τ)`), and comparing the two is one of the experiments.
- **One state loop.** Inside the coordinator, a single goroutine owns all state and processes events from a channel, so there are no locks and every change is one logged event.

The full design, with rejected alternatives and the failure matrix, is in [docs/design.md](docs/design.md).

## Repository layout

```text
coordinator/          Go: state loop, leases, aggregation, eval, checkpoints
  cmd/coordinator/    main.go, flags, HTTP endpoints
  cmd/fetch-mnist/    one-off MNIST download into data/
  internal/state/     Coordinator, Worker, Lease, Round; config; aggregation
  internal/protocol/  JSON messages, binary frame encode/decode
  internal/model/     flat weight layout (101,770 params), init
  internal/data/      MNIST loading and sharding
worker/               TypeScript + Vite + TensorFlow.js (WebGPU backend)
dashboard/            React + Vite + Recharts
harness/              TypeScript + Playwright chaos runs
experiments/          plot.py, run configs E1–E5
runs/                 run logs (git-ignored, except runs used in this README)
docs/                 design doc, charts
```

## Getting started

Requirements: Go 1.24+, Node 22+, Python 3.10+, and a browser with WebGPU (recent Chrome or Edge).

```bash
make install   # npm and pip dependencies
make data      # download MNIST into data/ (47 MB, once)
make test      # Go unit tests + worker tests
make demo      # build everything and start the coordinator on :8080
make experiment E=E3
make plots
```

Then open <http://localhost:8080> and the tab starts training. Until the coordinator protocol
lands (milestone 3) the tab walks every shard itself; `?epochs=`, `?shards=`, `?slow=` (per-step
delay, in ms) and `?worker=` (a label for the logs) control the run.

## Experiments

| ID | Question | Setup |
|---|---|---|
| E1 | What is the baseline? | 1 tab, no failures |
| E2 | Do more tabs train faster? | 1, 5, 10, 20 tabs |
| E3 | Does training survive failures? | 10 tabs, kill rate 0–50% per minute |
| E4 | Drop or down-weight stale updates? | 10 tabs, 30% kill rate, both strategies |
| E5 | What do slow workers cost? | 10 tabs, 30% slowed 3× |

Results and charts will appear here once a run folder backs them.

## Limitations

- The coordinator is a single point of failure; a crash pauses training until a manual restart from the last checkpoint.
- No defence against malicious workers; the sanity check only catches bugs.
- No privacy guarantee; workers see the data shards they are sent.
- Tested on one machine, not across real networks.
- Proven only on a ~100K-parameter model.
