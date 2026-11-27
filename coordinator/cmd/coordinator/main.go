// Command coordinator runs the training coordinator: a single state loop that
// owns every version, lease and round, plus HTTP and WebSocket endpoints.
package main

import (
	"errors"
	"flag"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/BorisZhang12/fault-tolerant-browser-training/coordinator/internal/data"
	"github.com/BorisZhang12/fault-tolerant-browser-training/coordinator/internal/model"
	"github.com/BorisZhang12/fault-tolerant-browser-training/coordinator/internal/state"
)

func main() {
	cfg := state.DefaultConfig()
	addr := flag.String("addr", ":8080", "listen address")
	workerDir := flag.String("worker-dir", "worker/dist", "built worker page to serve")
	dataDir := flag.String("data-dir", "data/mnist", "directory holding the MNIST IDX files")
	runID := flag.String("run-id", "dev", "run ID; logs go to runs/<run-id>/")
	cfg.RegisterFlags(flag.CommandLine)
	flag.Parse()

	mnist, err := data.Load(*dataDir, cfg.NumShards)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("run %s: %d parameters, strategy %s, %d shards of %d images, %d test images",
		*runID, model.NumParams, cfg.Strategy, len(mnist.Shards), mnist.Shards[0].Len(), mnist.Test.Len())

	mux := http.NewServeMux()
	// TODO(milestone 3): GET /dashboard, WS /ws/worker, WS /ws/dashboard, and the
	// state loop goroutine that owns every version, lease and round.
	// TODO(milestone 4): heartbeat timeouts, lease expiry and the staleness rules.
	mux.Handle("/shards/", shardHandler(mnist))
	mux.Handle("/", http.FileServer(http.Dir(*workerDir)))

	srv := &http.Server{
		Addr:              *addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("listening on %s", *addr)
	log.Fatal(srv.ListenAndServe())
}

// shardHandler serves GET /shards/{id} for a training shard, and GET
// /shards/test for the held-out test set, both in the binary shard format.
// Shards are immutable for the life of the process, so they cache forever.
func shardHandler(mnist *data.MNIST) http.Handler {
	// Encode once: 100 shards of 600 images is 47 MB, and every worker in a run
	// asks for the same bytes.
	encoded := make([][]byte, len(mnist.Shards))
	for i, s := range mnist.Shards {
		encoded[i] = data.EncodeShard(uint32(i), s)
	}
	testSet := data.EncodeShard(data.TestShardID, mnist.Test)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		name := strings.TrimPrefix(r.URL.Path, "/shards/")
		body, err := shardBytes(name, encoded, testSet)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		if r.Method == http.MethodHead {
			return
		}
		if _, err := w.Write(body); err != nil {
			log.Printf("shard %s: %v", name, err)
		}
	})
}

var errNoSuchShard = errors.New("no such shard")

func shardBytes(name string, encoded [][]byte, testSet []byte) ([]byte, error) {
	if name == "test" {
		return testSet, nil
	}
	id, err := strconv.Atoi(name)
	if err != nil || id < 0 || id >= len(encoded) {
		return nil, errNoSuchShard
	}
	return encoded[id], nil
}
