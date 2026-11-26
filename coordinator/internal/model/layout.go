// Package model defines the flat weight layout shared by the coordinator and
// the browser worker, plus (later) the gonum forward pass used for evaluation.
package model

import (
	"math"
	"math/rand"
)

// Multilayer perceptron: 784 -> 128 (ReLU) -> 10 (softmax).
const (
	Inputs  = 784
	Hidden  = 128
	Outputs = 10
)

// Tensor is one named slice of the flat weight vector.
type Tensor struct {
	Name   string
	Shape  []int
	Offset int
	Size   int
}

// Layout lists the tensors in their fixed wire order: W1, b1, W2, b2.
var Layout = func() []Tensor {
	specs := []struct {
		name  string
		shape []int
	}{
		{"W1", []int{Inputs, Hidden}},
		{"b1", []int{Hidden}},
		{"W2", []int{Hidden, Outputs}},
		{"b2", []int{Outputs}},
	}
	out := make([]Tensor, 0, len(specs))
	off := 0
	for _, s := range specs {
		n := 1
		for _, d := range s.shape {
			n *= d
		}
		out = append(out, Tensor{Name: s.name, Shape: s.shape, Offset: off, Size: n})
		off += n
	}
	return out
}()

// NumParams is the length of the flat weight vector (101,770).
var NumParams = func() int {
	last := Layout[len(Layout)-1]
	return last.Offset + last.Size
}()

// InitWeights returns Glorot-uniform weights and zero biases from a fixed seed,
// so every run with the same seed starts from the same model.
func InitWeights(seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	w := make([]float32, NumParams)
	for _, t := range Layout {
		if len(t.Shape) != 2 {
			continue // biases stay zero
		}
		fanIn, fanOut := float64(t.Shape[0]), float64(t.Shape[1])
		limit := math.Sqrt(6 / (fanIn + fanOut))
		for i := 0; i < t.Size; i++ {
			w[t.Offset+i] = float32((rng.Float64()*2 - 1) * limit)
		}
	}
	return w
}
