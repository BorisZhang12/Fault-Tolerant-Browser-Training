package protocol

import "testing"

func TestFrameRoundTrip(t *testing.T) {
	in := Frame{Kind: KindResult, Version: 7, SampleCount: 600, Values: []float32{1.5, -2, 0, 3.25}}
	in.LeaseID[0], in.LeaseID[15] = 0xab, 0xcd
	out, err := Decode(Encode(in), len(in.Values))
	if err != nil {
		t.Fatal(err)
	}
	if out.Kind != in.Kind || out.Version != in.Version || out.SampleCount != in.SampleCount || out.LeaseID != in.LeaseID {
		t.Fatalf("header mismatch: %+v vs %+v", out, in)
	}
	for i := range in.Values {
		if out.Values[i] != in.Values[i] {
			t.Fatalf("value %d: got %v want %v", i, out.Values[i], in.Values[i])
		}
	}
}

func TestDecodeRejectsWrongLength(t *testing.T) {
	buf := Encode(Frame{Kind: KindWeights, Values: make([]float32, 3)})
	if _, err := Decode(buf, 4); err == nil {
		t.Fatal("expected length error")
	}
	if _, err := Decode(buf[:10], 3); err != ErrShortFrame {
		t.Fatalf("expected ErrShortFrame, got %v", err)
	}
}
