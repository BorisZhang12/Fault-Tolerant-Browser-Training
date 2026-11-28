// Command fetch-mnist downloads the four MNIST IDX files into a data directory
// and gunzips them, so the project has no shell or Python dependency for setup.
// Files that already exist are left alone.
package main

import (
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/BorisZhang12/fault-tolerant-browser-training/coordinator/internal/data"
)

// mirror serves the original IDX files; yann.lecun.com now refuses automated clients.
const mirror = "https://storage.googleapis.com/cvdf-datasets/mnist/"

func main() {
	dir := flag.String("dir", "data/mnist", "directory to download into")
	flag.Parse()

	if err := os.MkdirAll(*dir, 0o755); err != nil {
		log.Fatal(err)
	}
	client := &http.Client{Timeout: 5 * time.Minute}
	for _, name := range data.Files() {
		path := filepath.Join(*dir, name)
		if _, err := os.Stat(path); err == nil {
			log.Printf("have %s", name)
			continue
		}
		if err := fetch(client, name, path); err != nil {
			log.Fatalf("%s: %v", name, err)
		}
	}
	log.Printf("MNIST ready in %s", *dir)
}

// fetch downloads name.gz, decompresses it, and writes it to path via a temp
// file and rename, so an interrupted run never leaves a half-written file.
func fetch(client *http.Client, name, path string) error {
	url := mirror + name + ".gz"
	log.Printf("downloading %s", url)
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gz.Close()

	tmp, err := os.CreateTemp(filepath.Dir(path), ".fetch-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	n, err := io.Copy(tmp, gz)
	if err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return err
	}
	log.Printf("wrote %s (%d bytes)", path, n)
	return nil
}
