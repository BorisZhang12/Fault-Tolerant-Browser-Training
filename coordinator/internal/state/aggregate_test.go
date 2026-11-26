package state

import (
	"math"
	"testing"
)

func TestStalenessWeight(t *testing.T) {
	cfg := DefaultConfig()
	cases := []struct {
		strat Strategy
		tau   int
		w     float64
		ok    bool
	}{
		{StrategyDrop, 0, 1, true},
		{StrategyDrop, 1, 1, false},
		{StrategyDownWeight, 0, 1, true},
		{StrategyDownWeight, 1, 0.5, true},
		{StrategyDownWeight, 2, 1.0 / 3, true},
		{StrategyDownWeight, 3, 0, false},
		{StrategyDownWeight, -1, 0, false},
	}
	for _, c := range cases {
		cfg.Strategy = c.strat
		w, ok := StalenessWeight(cfg, c.tau)
		if ok != c.ok || (ok && math.Abs(w-c.w) > 1e-12) {
			t.Errorf("%s tau=%d: got (%v,%v) want (%v,%v)", c.strat, c.tau, w, ok, c.w, c.ok)
		}
	}
}

func TestAggregateWeightedAverage(t *testing.T) {
	cfg := DefaultConfig() // down-weight
	w := []float32{0, 0}
	ups := []Update{
		{BaseVersion: 5, Samples: 600, Delta: []float32{1, 2}},   // tau 0, weight 600
		{BaseVersion: 4, Samples: 600, Delta: []float32{4, -2}},  // tau 1, weight 300
		{BaseVersion: 2, Samples: 600, Delta: []float32{99, 99}}, // tau 3, discarded
	}
	used := Aggregate(w, 5, ups, cfg)
	if len(used) != 2 {
		t.Fatalf("used %d updates, want 2", len(used))
	}
	// (600*1 + 300*4)/900 = 2 ; (600*2 + 300*-2)/900 = 2/3
	if math.Abs(float64(w[0])-2) > 1e-6 || math.Abs(float64(w[1])-2.0/3) > 1e-6 {
		t.Fatalf("got %v", w)
	}
}

func TestAggregateNoUpdatesLeavesWeights(t *testing.T) {
	w := []float32{1, 2}
	if used := Aggregate(w, 0, nil, DefaultConfig()); used != nil || w[0] != 1 || w[1] != 2 {
		t.Fatal("weights changed with no updates")
	}
}

func TestSanityFilter(t *testing.T) {
	ups := []Update{
		{Shard: 0, Delta: []float32{1, 0}},
		{Shard: 1, Delta: []float32{0, 1}},
		{Shard: 2, Delta: []float32{1, 1}},
		{Shard: 3, Delta: []float32{float32(math.NaN()), 0}},
		{Shard: 4, Delta: []float32{100, 0}},
	}
	kept, rej := SanityFilter(ups, 10)
	if len(kept) != 3 || len(rej) != 2 {
		t.Fatalf("kept %d rejected %d, want 3 and 2", len(kept), len(rej))
	}
}
