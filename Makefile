.PHONY: demo test experiment coordinator worker dashboard harness plots install data

E ?= E3

install:
	cd worker && npm install
	cd dashboard && npm install
	cd harness && npm install
	pip install -r experiments/requirements.txt

# MNIST IDX files, downloaded once into data/ (git-ignored).
data:
	cd coordinator && go run ./cmd/fetch-mnist -dir ../data/mnist

coordinator:
	cd coordinator && go build -o bin/coordinator ./cmd/coordinator

worker:
	cd worker && npm run build

dashboard:
	cd dashboard && npm run build

# Local demo: build the browser parts, then run the coordinator, which serves them.
demo: data worker dashboard coordinator
	./coordinator/bin/coordinator -addr :8080 -run-id demo

test:
	cd coordinator && go test ./...
	cd worker && npm test --if-present

experiment: data worker dashboard coordinator
	cd harness && npm run experiment -- --config ../experiments/configs/$(E).json

plots:
	python3 experiments/plot.py --runs runs --out docs
