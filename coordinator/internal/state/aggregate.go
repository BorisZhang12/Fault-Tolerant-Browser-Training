package state

import (
	"math"
	"sort"
)

// StalenessWeight returns s(tau) for an update and whether it is accepted.
func StalenessWeight(cfg Config, tau int) (float64, bool) {
	if tau < 0 {
		return 0, false
	}
	switch cfg.Strategy {
	case StrategyDrop:
		return 1, tau == 0
	default:
		if tau > cfg.MaxStaleness {
			return 0, false
		}
		return 1 / (1 + float64(tau)), true
	}
}

// Norm is the L2 norm of a delta, or +Inf if it contains NaN or Inf.
func Norm(d []float32) float64 {
	var s float64
	for _, v := range d {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return math.Inf(1)
		}
		s += f * f
	}
	return math.Sqrt(s)
}

// SanityFilter drops deltas with NaN/Inf or a norm above limit * the median norm
// of the finite ones. It catches bugs, not attackers. It returns kept and rejected.
func SanityFilter(updates []Update, limit float64) (kept, rejected []Update) {
	norms := make([]float64, len(updates))
	var finite []float64
	for i, u := range updates {
		norms[i] = Norm(u.Delta)
		if !math.IsInf(norms[i], 1) {
			finite = append(finite, norms[i])
		}
	}
	median := math.Inf(1)
	if len(finite) > 0 {
		sort.Float64s(finite)
		median = finite[len(finite)/2]
		if len(finite)%2 == 0 {
			median = (finite[len(finite)/2-1] + finite[len(finite)/2]) / 2
		}
	}
	for i, u := range updates {
		if math.IsInf(norms[i], 1) || norms[i] > limit*median {
			rejected = append(rejected, u)
		} else {
			kept = append(kept, u)
		}
	}
	return kept, rejected
}

// Aggregate applies W_{r+1} = W_r + sum(n_i s_i Delta_i) / sum(n_i s_i) in place,
// with s_i = StalenessWeight(version - BaseVersion_i). It returns the updates used.
func Aggregate(weights []float32, version int, updates []Update, cfg Config) []Update {
	acc := make([]float64, len(weights))
	var total float64
	var used []Update
	for _, u := range updates {
		s, ok := StalenessWeight(cfg, version-u.BaseVersion)
		if !ok || u.Samples <= 0 || len(u.Delta) != len(weights) {
			continue
		}
		w := float64(u.Samples) * s
		for i, d := range u.Delta {
			acc[i] += w * float64(d)
		}
		total += w
		used = append(used, u)
	}
	if total == 0 {
		return nil
	}
	for i := range weights {
		weights[i] += float32(acc[i] / total)
	}
	return used
}
