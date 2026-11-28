package data

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// idxImages builds an IDX3 file holding count images whose every pixel is the
// image's own index, so tests can tell images apart by content.
func idxImages(count int) []byte {
	buf := make([]byte, 16+count*Pixels)
	binary.BigEndian.PutUint32(buf[0:4], magicImages)
	binary.BigEndian.PutUint32(buf[4:8], uint32(count))
	binary.BigEndian.PutUint32(buf[8:12], 28)
	binary.BigEndian.PutUint32(buf[12:16], 28)
	for i := 0; i < count; i++ {
		for p := 0; p < Pixels; p++ {
			buf[16+i*Pixels+p] = byte(i)
		}
	}
	return buf
}

func idxLabels(count int) []byte {
	buf := make([]byte, 8+count)
	binary.BigEndian.PutUint32(buf[0:4], magicLabels)
	binary.BigEndian.PutUint32(buf[4:8], uint32(count))
	for i := 0; i < count; i++ {
		buf[8+i] = byte(i % 10)
	}
	return buf
}

func TestParseImages(t *testing.T) {
	images, count, err := ParseImages(idxImages(3))
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 || len(images) != 3*Pixels {
		t.Fatalf("got count %d and %d bytes, want 3 and %d", count, len(images), 3*Pixels)
	}
	if images[0] != 0 || images[Pixels] != 1 || images[2*Pixels] != 2 {
		t.Errorf("images are out of order: %d %d %d", images[0], images[Pixels], images[2*Pixels])
	}
}

func TestParseImagesRejectsBadInput(t *testing.T) {
	t.Run("short header", func(t *testing.T) {
		if _, _, err := ParseImages([]byte{0, 0, 8}); !errors.Is(err, ErrShortFile) {
			t.Errorf("got %v, want ErrShortFile", err)
		}
	})
	t.Run("wrong magic", func(t *testing.T) {
		buf := idxImages(1)
		buf[3] = 0x09
		if _, _, err := ParseImages(buf); err == nil {
			t.Error("accepted a file with the wrong magic number")
		}
	})
	t.Run("wrong image size", func(t *testing.T) {
		buf := idxImages(1)
		binary.BigEndian.PutUint32(buf[8:12], 27)
		if _, _, err := ParseImages(buf); err == nil {
			t.Error("accepted 27x28 images")
		}
	})
	t.Run("truncated body", func(t *testing.T) {
		buf := idxImages(2)
		if _, _, err := ParseImages(buf[:16+Pixels]); !errors.Is(err, ErrShortFile) {
			t.Errorf("got %v, want ErrShortFile", err)
		}
	})
}

func TestParseLabels(t *testing.T) {
	labels, err := ParseLabels(idxLabels(12))
	if err != nil {
		t.Fatal(err)
	}
	if len(labels) != 12 || labels[11] != 1 {
		t.Fatalf("got %d labels ending %d, want 12 ending 1", len(labels), labels[11])
	}
}

func TestParseLabelsRejectsOutOfRange(t *testing.T) {
	buf := idxLabels(2)
	buf[9] = 10
	if _, err := ParseLabels(buf); err == nil {
		t.Error("accepted label 10")
	}
}

// writeIDX lays out a data directory the way Load expects to find one.
func writeIDX(t *testing.T, train, test int) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string][]byte{
		trainImagesFile: idxImages(train),
		trainLabelsFile: idxLabels(train),
		testImagesFile:  idxImages(test),
		testLabelsFile:  idxLabels(test),
	}
	for name, buf := range files {
		if err := os.WriteFile(filepath.Join(dir, name), buf, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestLoadShardsEvenly(t *testing.T) {
	m, err := Load(writeIDX(t, 50, 7), 5)
	if err != nil {
		t.Fatal(err)
	}
	if m.Train.Len() != 50 || m.Test.Len() != 7 {
		t.Fatalf("got train %d test %d, want 50 and 7", m.Train.Len(), m.Test.Len())
	}
	if len(m.Shards) != 5 {
		t.Fatalf("got %d shards, want 5", len(m.Shards))
	}
	for i, s := range m.Shards {
		if s.Len() != 10 {
			t.Fatalf("shard %d has %d images, want 10", i, s.Len())
		}
		// Shard i must start at image 10*i: pixels carry the image index.
		if got := s.Images[0]; got != byte(10*i) {
			t.Errorf("shard %d starts at image %d, want %d", i, got, 10*i)
		}
	}
}

func TestLoadDropsUnevenRemainder(t *testing.T) {
	m, err := Load(writeIDX(t, 53, 4), 5)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range m.Shards {
		if s.Len() != 10 {
			t.Errorf("shard %d has %d images, want every shard at 10", i, s.Len())
		}
	}
}

func TestLoadRejectsTooManyShards(t *testing.T) {
	if _, err := Load(writeIDX(t, 4, 2), 5); err == nil {
		t.Error("accepted 5 shards for 4 images")
	}
	if _, err := Load(writeIDX(t, 4, 2), 0); err == nil {
		t.Error("accepted 0 shards")
	}
}

func TestLoadRejectsMismatchedCounts(t *testing.T) {
	dir := writeIDX(t, 10, 5)
	if err := os.WriteFile(filepath.Join(dir, trainLabelsFile), idxLabels(9), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(dir, 1); err == nil {
		t.Error("accepted 10 images with 9 labels")
	}
}

func TestLoadReportsMissingFiles(t *testing.T) {
	if _, err := Load(t.TempDir(), 1); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("got %v, want an os.ErrNotExist", err)
	}
}

func TestShardRoundTrip(t *testing.T) {
	m, err := Load(writeIDX(t, 20, 3), 2)
	if err != nil {
		t.Fatal(err)
	}
	buf := EncodeShard(1, m.Shards[1])
	if want := ShardHeaderSize + 10*Pixels + 10; len(buf) != want {
		t.Fatalf("encoded shard is %d bytes, want %d", len(buf), want)
	}
	id, s, err := DecodeShard(buf)
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Errorf("got shard ID %d, want 1", id)
	}
	if s.Len() != 10 {
		t.Fatalf("got %d images, want 10", s.Len())
	}
	if string(s.Images) != string(m.Shards[1].Images) || string(s.Labels) != string(m.Shards[1].Labels) {
		t.Error("decoded shard does not match the original")
	}
}

func TestDecodeShardRejectsBadInput(t *testing.T) {
	t.Run("short header", func(t *testing.T) {
		if _, _, err := DecodeShard(make([]byte, 4)); !errors.Is(err, ErrShortFile) {
			t.Errorf("got %v, want ErrShortFile", err)
		}
	})
	t.Run("wrong pixel count", func(t *testing.T) {
		buf := EncodeShard(0, Set{Images: make([]byte, Pixels), Labels: []byte{3}})
		binary.LittleEndian.PutUint32(buf[4:8], 783)
		if _, _, err := DecodeShard(buf); err == nil {
			t.Error("accepted 783 pixels per image")
		}
	})
	t.Run("truncated body", func(t *testing.T) {
		buf := EncodeShard(0, Set{Images: make([]byte, Pixels), Labels: []byte{3}})
		if _, _, err := DecodeShard(buf[:len(buf)-1]); err == nil {
			t.Error("accepted a shard missing its last byte")
		}
	})
}

func TestEncodeTestSetUsesTestShardID(t *testing.T) {
	id, _, err := DecodeShard(EncodeShard(TestShardID, Set{Images: make([]byte, Pixels), Labels: []byte{7}}))
	if err != nil {
		t.Fatal(err)
	}
	if id != TestShardID {
		t.Errorf("got shard ID %#x, want %#x", id, TestShardID)
	}
}

// TestLoadRealMNIST runs only when the downloaded IDX files are present.
func TestLoadRealMNIST(t *testing.T) {
	dir := filepath.Join("..", "..", "..", "data", "mnist")
	if _, err := os.Stat(filepath.Join(dir, trainImagesFile)); err != nil {
		t.Skip("no downloaded MNIST; run `make data`")
	}
	m, err := Load(dir, 100)
	if err != nil {
		t.Fatal(err)
	}
	if m.Train.Len() != 60000 || m.Test.Len() != 10000 {
		t.Errorf("got train %d test %d, want 60000 and 10000", m.Train.Len(), m.Test.Len())
	}
	if len(m.Shards) != 100 || m.Shards[0].Len() != 600 {
		t.Errorf("got %d shards of %d, want 100 of 600", len(m.Shards), m.Shards[0].Len())
	}
}
