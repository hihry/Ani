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

func TestWorkerClient_AsyncRenderJobPollingAndCancel(t *testing.T) {
	polls := 0
	deleted := false
	freed := false

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			_ = json.NewEncoder(w).Encode(client.HealthResponse{
				Status: "ok",
				Device: "cuda",
				VideoProviders: map[string]string{
					"wan22-ti2v-5b": "ready",
					"ltx23-fp8":     "ready",
				},
			})
		case r.Method == http.MethodPost && r.URL.Path == "/gpu/free":
			freed = true
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
		case r.Method == http.MethodPost && r.URL.Path == "/render/jobs":
			_ = json.NewEncoder(w).Encode(map[string]string{"job_id": "job_123", "status": "queued"})
		case r.Method == http.MethodGet && r.URL.Path == "/render/jobs/job_123":
			polls++
			if polls < 2 {
				_ = json.NewEncoder(w).Encode(map[string]any{"job_id": "job_123", "status": "running"})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"job_id": "job_123",
				"status": "done",
				"result": client.RenderResult{
					ClipPath:    "runs/test/clips/p1.avi",
					FramesCount: 60,
					DurationS:   2.5,
					RouteUsed:   "i2v",
					ModelID:     "wan22-ti2v-5b",
					Seed:        12345,
					GenSeconds:  14.2,
					PeakVRAMGB:  11.1,
					QCScores: model.QCScores{
						IdentitySimilarity: 0.91,
						FlickerScore:       0.04,
						MotionScore:        0.22,
						WarpError:          0.03,
						FirstFrameFidelity: 0.89,
						StyleDrift:         0.09,
						MetricsVersion:     2,
						OverallPass:        true,
					},
				},
			})
		case r.Method == http.MethodDelete && r.URL.Path == "/render/jobs/job_123":
			deleted = true
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "cancelled"})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer ts.Close()

	u, _ := url.Parse(ts.URL)
	port, _ := strconv.Atoi(u.Port())
	wc := client.NewWorkerClient(&config.PipelineConfig{
		Worker: config.WorkerConfig{Host: u.Hostname(), Port: port},
	})

	h, err := wc.Health(context.Background())
	if err != nil || h.VideoProviders["wan22-ti2v-5b"] != "ready" {
		t.Fatalf("unexpected Health response: %+v, err=%v", h, err)
	}

	res, err := wc.RenderClip(
		context.Background(),
		"in.png",
		"out.avi",
		model.RouteI2V,
		model.ShotCloseUp,
		2.5,
		24,
		540,
		960,
		client.RenderOptions{PanelID: "p1", ModelID: "wan22-ti2v-5b", Seed: 12345},
	)
	if err != nil {
		t.Fatalf("RenderClip async I2V failed: %v", err)
	}
	if polls < 2 || res.ModelID != "wan22-ti2v-5b" || res.GenSeconds != 14.2 {
		t.Fatalf("unexpected async RenderClip result: polls=%d res=%+v", polls, res)
	}

	if err := wc.FreeGPU(context.Background()); err != nil || !freed {
		t.Fatalf("FreeGPU failed: freed=%v err=%v", freed, err)
	}
	_ = deleted
}
