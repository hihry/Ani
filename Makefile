.PHONY: build run test test-py lint clean worker worker-install inspect

BIN ?= m2a.exe

# === Go Backend ===
build:
	go build -o $(BIN) ./cmd/m2a/

run: build
	./$(BIN) run --pages data/eval/sample_page_1.png,data/eval/sample_page_2.png,data/eval/sample_page_3.png --tier t1 --lang en --auto-approve

test:
	go test ./... -v

test-py:
	python -m pytest tests/test_pipeline.py -v

lint:
	go vet ./...

# === Python Worker ===
worker-install:
	cd workers && pip install -r requirements.txt

worker:
	python -m uvicorn workers.server:app --host 127.0.0.1 --port 8100 --reload

# === Maintenance ===
clean:
	rm -rf $(BIN) m2a runs/

inspect:
	./$(BIN) inspect --run-dir $(RUN_DIR)
