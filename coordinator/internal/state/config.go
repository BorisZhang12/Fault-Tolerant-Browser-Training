// Package state owns all training state. One goroutine runs the state loop and
// handles events one at a time; nothing else touches these structs.
package state

import (
	"flag"
	"time"
)

type Strategy string

const (
	StrategyDrop       Strategy = "drop"        // accept only staleness 0
	StrategyDownWeight Strategy = "down-weight" // s(t) = 1/(1+t), discard if t > MaxStaleness
)

type Config struct {
	HeartbeatInterval time.Duration
	DeadAfter         time.Duration
	RoundDeadline     time.Duration
	LeaseGrace        time.Duration
	Quorum            float64 // fraction of assigned shards needed to close at the deadline
	MaxStaleness      int
	Strategy          Strategy
	NormLimit         float64 // reject deltas with norm > NormLimit * median accepted norm

	NumShards    int
	BatchSize    int
	LearningRate float64
	Seed         int64
}

func DefaultConfig() Config {
	return Config{
		HeartbeatInterval: 2 * time.Second,
		DeadAfter:         6 * time.Second,
		RoundDeadline:     10 * time.Second,
		LeaseGrace:        5 * time.Second,
		Quorum:            0.5,
		MaxStaleness:      2,
		Strategy:          StrategyDownWeight,
		NormLimit:         10,
		NumShards:         100,
		BatchSize:         32,
		LearningRate:      0.05,
		Seed:              42,
	}
}

// RegisterFlags binds every config field to a command-line flag.
func (c *Config) RegisterFlags(fs *flag.FlagSet) {
	fs.DurationVar(&c.HeartbeatInterval, "heartbeat", c.HeartbeatInterval, "worker heartbeat interval")
	fs.DurationVar(&c.DeadAfter, "dead-after", c.DeadAfter, "mark a worker dead after this long without a heartbeat")
	fs.DurationVar(&c.RoundDeadline, "round-deadline", c.RoundDeadline, "round deadline")
	fs.DurationVar(&c.LeaseGrace, "lease-grace", c.LeaseGrace, "lease expiry = round deadline + grace")
	fs.Float64Var(&c.Quorum, "quorum", c.Quorum, "fraction of shards needed to close a round at its deadline")
	fs.IntVar(&c.MaxStaleness, "max-staleness", c.MaxStaleness, "discard updates staler than this (down-weight strategy)")
	fs.Func("strategy", "late-update strategy: drop or down-weight (default down-weight)", func(s string) error {
		c.Strategy = Strategy(s)
		return nil
	})
	fs.Float64Var(&c.NormLimit, "norm-limit", c.NormLimit, "reject deltas whose norm exceeds this multiple of the round median")
	fs.IntVar(&c.NumShards, "shards", c.NumShards, "number of training shards")
	fs.IntVar(&c.BatchSize, "batch-size", c.BatchSize, "worker batch size")
	fs.Float64Var(&c.LearningRate, "lr", c.LearningRate, "worker SGD learning rate")
	fs.Int64Var(&c.Seed, "seed", c.Seed, "weight initialization seed")
}
