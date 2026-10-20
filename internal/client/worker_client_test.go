package client_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/m2a/manhwa2anime/internal/client"
	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
)

func TestWorkerClient_HTTPTransport(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/ingest":
			_ = json.NewEncoder(w).Encode(client.IngestResult{
				StitchedPath: "runs/test/stitched.png",
				Width:        800,
				Height:       2200,
			})
		case "/understand":
			_ = json.NewEncoder(w).Encode(client.UnderstandResult{
				PanelID:         "panel_001",
				Shot:            model.ShotCloseUp,
				Action:          "Jin ignites mana dagger",
				MotionPrompt:    "slow push-in",
				MotionIntensity: 0.7,
			})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer ts.Close()

	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatalf("parse server url: %v", err)
	}
	port, _ := strconv.Atoi(u.Port())

	cfg := &config.PipelineConfig{
		Worker: config.WorkerConfig{
			Host: u.Hostname(),
			Port: port,
		},
	}
	wc := client.NewWorkerClient(cfg)

	ing, err := wc.Ingest(context.Background(), []string{"p1.png"}, "runs/test/stitched.png")
	if err != nil {
		t.Fatalf("Ingest over HTTP failed: %v", err)
	}
	if ing.Width != 800 || ing.Height != 2200 {
		t.Errorf("unexpected ingest dimensions: %+v", ing)
	}

	und, err := wc.Understand(context.Background(), "panel_001.png", "panel_001", "ko", 1)
	if err != nil {
		t.Fatalf("Understand over HTTP failed: %v", err)
	}
	if und.Shot != model.ShotCloseUp || und.MotionIntensity != 0.7 {
		t.Errorf("unexpected understand response: %+v", und)
	}
}

func TestWorkerClient_NoDoubleExecutionOnHTTPError(t *testing.T) {
	httpCalls := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpCalls++
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"detail": "GPU OOM during render"}`))
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	port, _ := strconv.Atoi(u.Port())
	wc := client.NewWorkerClient(&config.PipelineConfig{
		Worker: config.WorkerConfig{Host: u.Hostname(), Port: port},
	})

	_, err := wc.RenderClip(context.Background(), "in.png", "out.avi", model.RouteKenBurns, model.ShotMedium, 2.5, 24, 720, 1280)
	if err == nil {
		t.Fatal("expected HTTP 500 error to be returned, got nil")
	}
	if httpCalls != 1 {
		t.Fatalf("expected exactly 1 HTTP call, got %d", httpCalls)
	}
}
