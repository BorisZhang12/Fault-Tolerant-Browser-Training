// Package data loads MNIST from the original IDX files and splits the training
// set into fixed shards. The coordinator keeps the test set for evaluation and
// serves training shards to workers over HTTP in the binary form below.
//
// Shard wire format (little-endian; images are row-major uint8, 0-255):
//
//	bytes    field
//	4        count (uint32): images in this shard
//	4        pixels (uint32): pixels per image, always 784
//	4        shard (uint32): shard ID, or 0xFFFFFFFF for the test set
//	count*784 image bytes
//	count    labels (uint8, 0-9)
package data

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Pixels is the number of pixels in one MNIST image (28x28).
const Pixels = 784

// TestShardID marks an encoded shard as the test set rather than a training shard.
const TestShardID = 0xFFFFFFFF

// ShardHeaderSize is the fixed header length of an encoded shard.
const ShardHeaderSize = 12

// Set is a collection of MNIST images with their labels. Images holds
// len(Labels)*Pixels bytes, image i starting at i*Pixels.
type Set struct {
	Images []byte
	Labels []byte
}

// Len returns the number of images in the set.
func (s Set) Len() int { return len(s.Labels) }

// Slice returns images [from,to) as a new Set sharing the underlying arrays.
func (s Set) Slice(from, to int) Set {
	return Set{Images: s.Images[from*Pixels : to*Pixels], Labels: s.Labels[from:to]}
}

// IDX file magic numbers: 0x0801 = 1-D uint8 (labels), 0x0803 = 3-D uint8 (images).
const (
	magicLabels = 0x0801
	magicImages = 0x0803
)

// ErrShortFile reports an IDX file that ends before its header or its data does.
var ErrShortFile = errors.New("data: IDX file shorter than its header declares")

// ParseImages reads an IDX3 image file: magic, count, rows, cols, then pixels.
func ParseImages(buf []byte) ([]byte, int, error) {
	if len(buf) < 16 {
		return nil, 0, ErrShortFile
	}
	if got := binary.BigEndian.Uint32(buf[0:4]); got != magicImages {
		return nil, 0, fmt.Errorf("data: image magic %#x, want %#x", got, magicImages)
	}
	count := int(binary.BigEndian.Uint32(buf[4:8]))
	rows := int(binary.BigEndian.Uint32(buf[8:12]))
	cols := int(binary.BigEndian.Uint32(buf[12:16]))
	if rows*cols != Pixels {
		return nil, 0, fmt.Errorf("data: images are %dx%d, want 28x28", rows, cols)
	}
	body := buf[16:]
	if len(body) < count*Pixels {
		return nil, 0, ErrShortFile
	}
	return body[:count*Pixels], count, nil
}

// ParseLabels reads an IDX1 label file: magic, count, then one byte per label.
func ParseLabels(buf []byte) ([]byte, error) {
	if len(buf) < 8 {
		return nil, ErrShortFile
	}
	if got := binary.BigEndian.Uint32(buf[0:4]); got != magicLabels {
		return nil, fmt.Errorf("data: label magic %#x, want %#x", got, magicLabels)
	}
	count := int(binary.BigEndian.Uint32(buf[4:8]))
	body := buf[8:]
	if len(body) < count {
		return nil, ErrShortFile
	}
	for i, l := range body[:count] {
		if l > 9 {
			return nil, fmt.Errorf("data: label %d at index %d is out of range", l, i)
		}
	}
	return body[:count], nil
}

// MNIST holds the training set, its shards, and the held-out test set.
type MNIST struct {
	Train  Set
	Test   Set
	Shards []Set // Shards[i] is shard i of the training set
}

// IDX file names inside the data directory.
const (
	trainImagesFile = "train-images-idx3-ubyte"
	trainLabelsFile = "train-labels-idx1-ubyte"
	testImagesFile  = "t10k-images-idx3-ubyte"
	testLabelsFile  = "t10k-labels-idx1-ubyte"
)

// Files lists the IDX files Load expects in its data directory.
func Files() []string {
	return []string{trainImagesFile, trainLabelsFile, testImagesFile, testLabelsFile}
}

// Load reads the four IDX files from dir and splits the training set into
// numShards equal shards. Any images left over by an uneven split are dropped,
// so every shard has the same sample count.
func Load(dir string, numShards int) (*MNIST, error) {
	if numShards <= 0 {
		return nil, fmt.Errorf("data: numShards is %d, want at least 1", numShards)
	}
	train, err := loadSet(dir, trainImagesFile, trainLabelsFile)
	if err != nil {
		return nil, err
	}
	test, err := loadSet(dir, testImagesFile, testLabelsFile)
	if err != nil {
		return nil, err
	}
	per := train.Len() / numShards
	if per == 0 {
		return nil, fmt.Errorf("data: %d training images cannot fill %d shards", train.Len(), numShards)
	}
	shards := make([]Set, numShards)
	for i := range shards {
		shards[i] = train.Slice(i*per, (i+1)*per)
	}
	return &MNIST{Train: train, Test: test, Shards: shards}, nil
}

func loadSet(dir, imagesFile, labelsFile string) (Set, error) {
	rawImages, err := os.ReadFile(filepath.Join(dir, imagesFile))
	if err != nil {
		return Set{}, fmt.Errorf("data: %w (run `make data` to download MNIST)", err)
	}
	rawLabels, err := os.ReadFile(filepath.Join(dir, labelsFile))
	if err != nil {
		return Set{}, fmt.Errorf("data: %w (run `make data` to download MNIST)", err)
	}
	images, count, err := ParseImages(rawImages)
	if err != nil {
		return Set{}, err
	}
	labels, err := ParseLabels(rawLabels)
	if err != nil {
		return Set{}, err
	}
	if len(labels) != count {
		return Set{}, fmt.Errorf("data: %s has %d images but %s has %d labels", imagesFile, count, labelsFile, len(labels))
	}
	return Set{Images: images, Labels: labels}, nil
}

// EncodeShard serializes a set in the wire format documented above.
func EncodeShard(shardID uint32, s Set) []byte {
	n := s.Len()
	buf := make([]byte, ShardHeaderSize+n*Pixels+n)
	binary.LittleEndian.PutUint32(buf[0:4], uint32(n))
	binary.LittleEndian.PutUint32(buf[4:8], Pixels)
	binary.LittleEndian.PutUint32(buf[8:12], shardID)
	copy(buf[ShardHeaderSize:], s.Images[:n*Pixels])
	copy(buf[ShardHeaderSize+n*Pixels:], s.Labels[:n])
	return buf
}

// DecodeShard parses the wire format, mainly so tests can check the encoder
// against the TypeScript decoder's expectations.
func DecodeShard(buf []byte) (shardID uint32, s Set, err error) {
	if len(buf) < ShardHeaderSize {
		return 0, Set{}, ErrShortFile
	}
	n := int(binary.LittleEndian.Uint32(buf[0:4]))
	pixels := int(binary.LittleEndian.Uint32(buf[4:8]))
	shardID = binary.LittleEndian.Uint32(buf[8:12])
	if pixels != Pixels {
		return 0, Set{}, fmt.Errorf("data: shard says %d pixels per image, want %d", pixels, Pixels)
	}
	if want := ShardHeaderSize + n*Pixels + n; len(buf) != want {
		return 0, Set{}, fmt.Errorf("data: shard is %d bytes, want %d for %d images", len(buf), want, n)
	}
	return shardID, Set{
		Images: buf[ShardHeaderSize : ShardHeaderSize+n*Pixels],
		Labels: buf[ShardHeaderSize+n*Pixels:],
	}, nil
}
