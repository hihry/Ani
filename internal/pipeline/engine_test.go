package pipeline_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
	"github.com/m2a/manhwa2anime/internal/pipeline"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func testConfig() *config.PipelineConfig {
	return &config.PipelineConfig{
		General:       config.GeneralConfig{LogLevel: "info"},
		Perception:    config.PerceptionConfig{Detector: "opencv"},
		Understanding: config.UnderstandingConfig{VLM: "qwen"},
		Cleanup:       config.CleanupConfig{Inpainter: "lama"},
		Generation:    config.GenerationConfig{PrimaryModel: "wan", MaxClipDuration: 4.0},
		Composition:   config.CompositionConfig{OutputFormat: "mp4", Resolution: [2]int{720, 1280}, FPS: 24},
	}
}

func TestEngine_LifecycleWithCheckpoints(t *testing.T) {
	runDir := t.TempDir()
	logger := discardLogger()
	cfg := testConfig()
	cfg.General.RunDir = runDir

	state := model.NewRunState([]string{"p1.png"}, "ko", "t1")
	engine := pipeline.NewEngine(runDir, logger)
	pipeline.RegisterDefaultNodes(engine)

	// Step 1: Initial Run should stop at CHECKPOINT_1
	err := engine.Run(context.Background(), state, cfg)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if state.CurrentStage != model.StageCheckpoint1 {
		t.Fatalf("expected stage CHECKPOINT_1, got %s", state.CurrentStage)
	}
	if len(state.Panels) != 3 {
		t.Fatalf("expected 3 panels created by SegmentNode, got %d", len(state.Panels))
	}

	// Verify state saved to disk
	loadedState, err := model.LoadRunState(runDir)
	if err != nil {
		t.Fatalf("load run state from disk failed: %v", err)
	}
	if loadedState.CurrentStage != model.StageCheckpoint1 {
		t.Fatalf("loaded state expected CHECKPOINT_1, got %s", loadedState.CurrentStage)
	}

	// Step 2: First Resume should resume from CHECKPOINT_1 and pause at CHECKPOINT_2
	err = engine.Resume(context.Background(), loadedState, cfg)
	if err != nil {
		t.Fatalf("first resume failed: %v", err)
	}

	if loadedState.CurrentStage != model.StageCheckpoint2 {
		t.Fatalf("expected stage CHECKPOINT_2, got %s", loadedState.CurrentStage)
	}
	if len(loadedState.CleanPlatePaths) != 3 {
		t.Fatalf("expected 3 clean plates created by CleanPlatesNode, got %d", len(loadedState.CleanPlatePaths))
	}

	// Step 3: Second Resume should resume from CHECKPOINT_2 and complete at DONE
	err = engine.Resume(context.Background(), loadedState, cfg)
	if err != nil {
		t.Fatalf("second resume failed: %v", err)
	}

	if loadedState.CurrentStage != model.StageDone {
		t.Fatalf("expected stage DONE, got %s", loadedState.CurrentStage)
	}
	if len(loadedState.ClipCandidates) != 3 {
		t.Fatalf("expected 3 clip candidate entries, got %d", len(loadedState.ClipCandidates))
	}
}

func TestEngine_FallbackHandling(t *testing.T) {
	runDir := t.TempDir()
	logger := discardLogger()
	cfg := testConfig()
	cfg.General.RunDir = runDir

	state := model.NewRunState([]string{"p1.png"}, "ko", "t1")
	engine := pipeline.NewEngine(runDir, logger)
	pipeline.RegisterDefaultNodes(engine)

	fallbackCalled := false
	engine.RegisterNode(model.StageIngest, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		return errors.New("primary ingest failed")
	})
	engine.RegisterFallback(model.StageIngest, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		fallbackCalled = true
		time.Sleep(2 * time.Millisecond)
		return nil
	})

	err := engine.Run(context.Background(), state, cfg)
	if err != nil {
		t.Fatalf("engine run with fallback failed: %v", err)
	}
	if !fallbackCalled {
		t.Fatal("expected fallback handler to be invoked, but was not")
	}
	if _, ok := state.CostLedger.PerStage["INGEST_fallback"]; !ok {
		t.Errorf("expected cost ledger key for INGEST_fallback, got %v", state.CostLedger.PerStage)
	}
}

func TestEngine_StageFailureMarksFailed(t *testing.T) {
	runDir := t.TempDir()
	logger := discardLogger()
	cfg := testConfig()
	cfg.General.RunDir = runDir

	state := model.NewRunState([]string{"p1.png"}, "ko", "t1")
	engine := pipeline.NewEngine(runDir, logger)

	engine.RegisterNode(model.StageIngest, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		return errors.New("unrecoverable failure")
	})

	err := engine.Run(context.Background(), state, cfg)
	if err == nil {
		t.Fatal("expected error on unrecoverable failure, got nil")
	}

	if state.CurrentStage != model.StageFailed {
		t.Fatalf("expected stage to be marked FAILED, got %s", state.CurrentStage)
	}
	if len(state.Errors) == 0 {
		t.Fatal("expected error to be recorded in state.Errors")
	}

	// Verify saved state on disk reflects failure
	loadedState, err := model.LoadRunState(runDir)
	if err != nil {
		t.Fatalf("failed to load state: %v", err)
	}
	if loadedState.CurrentStage != model.StageFailed {
		t.Fatalf("expected saved state to be FAILED, got %s", loadedState.CurrentStage)
	}
}

func TestEngine_SubdirectoriesCreated(t *testing.T) {
	runDir := t.TempDir()
	logger := discardLogger()
	cfg := testConfig()
	cfg.General.RunDir = runDir

	state := model.NewRunState([]string{"p1.png"}, "ko", "t1")
	engine := pipeline.NewEngine(runDir, logger)
	pipeline.RegisterDefaultNodes(engine)

	_ = engine.Run(context.Background(), state, cfg)
	_ = engine.Resume(context.Background(), state, cfg)
	_ = engine.Resume(context.Background(), state, cfg)

	expectedDirs := []string{
		"pages",
		"panels",
		"understanding",
		"clean_plates",
		"routes",
		"clips",
		"qc",
		"audio",
		"output",
	}

	for _, d := range expectedDirs {
		p := filepath.Join(runDir, d)
		info, err := os.Stat(p)
		if err != nil || !info.IsDir() {
			t.Errorf("expected directory %q to be created in run folder", d)
		}
	}
}
