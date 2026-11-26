package model

import "testing"

func TestNumParams(t *testing.T) {
	if NumParams != 101770 {
		t.Fatalf("NumParams = %d, want 101770", NumParams)
	}
}

func TestInitWeightsDeterministic(t *testing.T) {
	a, b := InitWeights(42), InitWeights(42)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("weights differ at %d with the same seed", i)
		}
	}
	b1 := Layout[1]
	for i := 0; i < b1.Size; i++ {
		if a[b1.Offset+i] != 0 {
			t.Fatal("biases should start at zero")
		}
	}
}
