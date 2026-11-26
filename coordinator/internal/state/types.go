package state

import "time"

type Worker struct {
	ID       string
	Label    string
	Backend  string
	LastBeat time.Time
	Lease    *Lease      // nil when idle
	Out      chan []byte // outbound frames, drained by the connection's writer goroutine
}

type Lease struct {
	ID          string
	WorkerID    string
	Shard       int
	BaseVersion int
	Expires     time.Time
}

type Round struct {
	Version  int
	Deadline time.Time
	Assigned map[int]*Lease // by shard
	Returned map[int]Update // by shard; each shard counts at most once per version
}

type Update struct {
	LeaseID     string
	WorkerID    string
	Shard       int
	BaseVersion int
	Samples     int
	Delta       []float32
}

// Coordinator is owned by the state loop goroutine.
type Coordinator struct {
	Version int
	Weights []float32
	Workers map[string]*Worker
	Leases  map[string]*Lease // by lease ID
	Pending []int             // shard IDs waiting for a worker
	Round   *Round
	LateBuf []Update // late updates for the next aggregation
	Cfg     Config
}

// Events sent by connection goroutines to the state loop.
type (
	EvWorkerJoined struct{ W *Worker }
	EvHeartbeat    struct {
		WorkerID   string
		NumTensors int
		Busy       bool
	}
	EvUpdate       struct{ U Update }
	EvDisconnected struct{ WorkerID string }
	EvTick         struct{ Now time.Time }
)
