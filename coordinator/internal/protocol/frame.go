// Package protocol holds the JSON control messages and the binary frame format
// for weights and deltas. All binary fields are little-endian.
//
//	bytes  field
//	1      kind: 1 = weights, 2 = result
//	4      version (uint32): the weights' version, or a result's base version
//	16     lease ID (UUID bytes; zeros for weights)
//	4      sample count (uint32; zero for weights)
//	4*N    float32 values: weights, or the delta for a result
package protocol

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

type FrameKind uint8

const (
	KindWeights FrameKind = 1
	KindResult  FrameKind = 2
)

const HeaderSize = 1 + 4 + 16 + 4

type Frame struct {
	Kind        FrameKind
	Version     uint32
	LeaseID     [16]byte
	SampleCount uint32
	Values      []float32
}

func Encode(f Frame) []byte {
	buf := make([]byte, HeaderSize+4*len(f.Values))
	buf[0] = byte(f.Kind)
	binary.LittleEndian.PutUint32(buf[1:5], f.Version)
	copy(buf[5:21], f.LeaseID[:])
	binary.LittleEndian.PutUint32(buf[21:25], f.SampleCount)
	for i, v := range f.Values {
		binary.LittleEndian.PutUint32(buf[HeaderSize+4*i:], math.Float32bits(v))
	}
	return buf
}

var ErrShortFrame = errors.New("protocol: frame shorter than header")

// Decode parses a frame and checks it carries exactly wantValues floats.
func Decode(buf []byte, wantValues int) (Frame, error) {
	if len(buf) < HeaderSize {
		return Frame{}, ErrShortFrame
	}
	f := Frame{
		Kind:        FrameKind(buf[0]),
		Version:     binary.LittleEndian.Uint32(buf[1:5]),
		SampleCount: binary.LittleEndian.Uint32(buf[21:25]),
	}
	if f.Kind != KindWeights && f.Kind != KindResult {
		return Frame{}, fmt.Errorf("protocol: unknown frame kind %d", f.Kind)
	}
	copy(f.LeaseID[:], buf[5:21])
	body := buf[HeaderSize:]
	if len(body) != 4*wantValues {
		return Frame{}, fmt.Errorf("protocol: got %d body bytes, want %d", len(body), 4*wantValues)
	}
	f.Values = make([]float32, wantValues)
	for i := range f.Values {
		f.Values[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[4*i:]))
	}
	return f, nil
}
