.PHONY: build run test lint clean worker worker-install inspect

# === Go Backend ===
build:
	go build -o m2a.exe ./cmd/m2a/

run: build
	./m2a.exe run --pages sample.png --tier t1

test:
	go test ./... -v

lint:
	go vet ./...

# === Python Worker ===
worker-install:
	cd workers && pip install -r requirements.txt

worker:
	python -m uvicorn workers.server:app --host 127.0.0.1 --port 8100 --reload

# === Maintenance ===
clean:
	rm -rf m2a.exe runs/

inspect:
	./m2a.exe inspect --run-dir $(RUN_DIR)
