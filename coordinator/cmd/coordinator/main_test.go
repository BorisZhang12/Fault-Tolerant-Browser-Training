package main

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/BorisZhang12/fault-tolerant-browser-training/coordinator/internal/data"
)

// testMNIST builds a two-shard set in memory: shard i's pixels are all i.
func testMNIST(t *testing.T) *data.MNIST {
	t.Helper()
	const perShard = 3
	shards := make([]data.Set, 2)
	for i := range shards {
		images := make([]byte, perShard*data.Pixels)
		for p := range images {
			images[p] = byte(i)
		}
		shards[i] = data.Set{Images: images, Labels: []byte{1, 2, 3}}
	}
	return &data.MNIST{
		Shards: shards,
		Test:   data.Set{Images: make([]byte, data.Pixels), Labels: []byte{7}},
	}
}

func get(t *testing.T, h http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	return rec
}

func TestShardHandlerServesTrainingShard(t *testing.T) {
	h := shardHandler(testMNIST(t))
	rec := get(t, h, "/shards/1")
	if rec.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/octet-stream" {
		t.Errorf("got content type %q, want application/octet-stream", ct)
	}
	id, s, err := data.DecodeShard(rec.Body.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Errorf("got shard ID %d, want 1", id)
	}
	if s.Len() != 3 || s.Images[0] != 1 {
		t.Errorf("got %d images starting %d, want 3 starting 1", s.Len(), s.Images[0])
	}
}

func TestShardHandlerServesTestSet(t *testing.T) {
	rec := get(t, shardHandler(testMNIST(t)), "/shards/test")
	if rec.Code != http.StatusOK {
		t.Fatalf("got status %d, want 200", rec.Code)
	}
	id, s, err := data.DecodeShard(rec.Body.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if id != data.TestShardID {
		t.Errorf("got shard ID %#x, want the test ID %#x", id, data.TestShardID)
	}
	if s.Len() != 1 || s.Labels[0] != 7 {
		t.Errorf("got %d images labelled %v, want 1 labelled [7]", s.Len(), s.Labels)
	}
}

func TestShardHandlerRejectsUnknownShards(t *testing.T) {
	h := shardHandler(testMNIST(t))
	for _, path := range []string{"/shards/2", "/shards/-1", "/shards/", "/shards/abc", "/shards/1.5"} {
		if rec := get(t, h, path); rec.Code != http.StatusNotFound {
			t.Errorf("GET %s: got status %d, want 404", path, rec.Code)
		}
	}
}

func TestShardHandlerRejectsWrites(t *testing.T) {
	rec := httptest.NewRecorder()
	shardHandler(testMNIST(t)).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/shards/0", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("got status %d, want 405", rec.Code)
	}
}
