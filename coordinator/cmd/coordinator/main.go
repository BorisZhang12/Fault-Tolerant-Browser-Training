// Command coordinator runs the training coordinator: a single state loop that
// owns every version, lease and round, plus HTTP and WebSocket endpoints.
package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/BorisZhang12/fault-tolerant-browser-training/coordinator/internal/model"
	"github.com/BorisZhang12/fault-tolerant-browser-training/coordinator/internal/state"
)

func main() {
	cfg := state.DefaultConfig()
	addr := flag.String("addr", ":8080", "listen address")
	workerDir := flag.String("worker-dir", "worker/dist", "built worker page to serve")
	runID := flag.String("run-id", "dev", "run ID; logs go to runs/<run-id>/")
	cfg.RegisterFlags(flag.CommandLine)
	flag.Parse()

	log.Printf("run %s: %d parameters, strategy %s", *runID, model.NumParams, cfg.Strategy)

	mux := http.NewServeMux()
	// TODO(milestone 3): GET / (worker page), GET /dashboard, GET /shards/{id},
	// WS /ws/worker, WS /ws/dashboard, and the state loop goroutine.
	mux.Handle("/", http.FileServer(http.Dir(*workerDir)))
	log.Printf("listening on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
