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

// registerMockNodes registers fast in-memory stage handlers for testing the Engine state machine.
func registerMockNodes(e *pipeline.Engine, runDir string) {
	mkDir := func(sub string) error {
		return os.MkdirAll(filepath.Join(runDir, sub), 0755)
	}
	e.RegisterNode(model.StageIngest, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		return mkDir("pages")
	})
	e.RegisterNode(model.StageSegment, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		if err := mkDir("panels"); err != nil {
			return err
		}
		s.Panels = []model.PanelJSON{
			model.NewPanelJSON("panel_001", model.BBox{X: 0, Y: 0, W: 720, H: 400}),
			model.NewPanelJSON("panel_002", model.BBox{X: 0, Y: 420, W: 720, H: 450}),
			model.NewPanelJSON("panel_003", model.BBox{X: 0, Y: 890, W: 720, H: 390}),
		}
		return nil
	})
	e.RegisterNode(model.StageUnderstand, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		return mkDir("understanding")
	})
	e.RegisterNode(model.StageCleanPlates, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		if err := mkDir("clean_plates"); err != nil {
			return err
		}
		for _, p := range s.Panels {
			s.CleanPlatePaths[p.PanelID] = filepath.ToSlash(filepath.Join(runDir, "clean_plates", p.PanelID+"_clean.png"))
		}
		return nil
	})
	e.RegisterNode(model.StageRoute, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		return mkDir("routes")
	})
	e.RegisterNode(model.StageTTS, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		return mkDir("audio")
	})
	e.RegisterNode(model.StageGenerate, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		if err := mkDir("clips"); err != nil {
			return err
		}
		for _, p := range s.Panels {
			s.ClipCandidates[p.PanelID] = []model.ClipCandidate{{
				PanelID:    p.PanelID,
				Path:       filepath.ToSlash(filepath.Join(runDir, "clips", p.PanelID+".avi")),
				TakeNumber: 1,
				QC: model.QCScores{
					IdentitySimilarity: 0.94,
					NewTextCount:       0,
					FlickerScore:       0.02,
					MotionScore:        0.08,
					OverallPass:        true,
				},
				Accepted: true,
			}}
		}
		return nil
	})
	e.RegisterNode(model.StageQC, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		return mkDir("qc")
	})
	e.RegisterNode(model.StageAssemble, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		return mkDir("output")
	})
}

func TestEngine_LifecycleWithCheckpoints(t *testing.T) {
	runDir := t.TempDir()
	logger := discardLogger()
	cfg := testConfig()
	cfg.General.RunDir = runDir

	state := model.NewRunState([]string{"p1.png"}, "ko", "t1")
	engine := pipeline.NewEngine(runDir, logger)
	registerMockNodes(engine, runDir)

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
	registerMockNodes(engine, runDir)

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
	registerMockNodes(engine, runDir)

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

// TestIngest_MissingPageFails verifies P0-01: a missing input page fails INGEST immediately instead of fabricating dummy panels.
func TestIngest_MissingPageFails(t *testing.T) {
	runDir := t.TempDir()
	logger := discardLogger()
	cfg := testConfig()
	cfg.General.RunDir = runDir

	state := model.NewRunState([]string{"does_not_exist.png"}, "ko", "t1")
	engine := pipeline.NewEngine(runDir, logger)
	pipeline.RegisterDefaultNodes(engine)

	err := engine.Run(context.Background(), state, cfg)
	if err == nil {
		t.Fatal("expected engine.Run to fail on missing source page, got nil")
	}
	if state.CurrentStage != model.StageFailed {
		t.Fatalf("expected stage FAILED on missing source page, got %s", state.CurrentStage)
	}
}

// TestQC_NoDuplicateKenBurnsReRender verifies P0-02/O5: a failed Ken Burns candidate does not trigger a duplicate re-render or force OverallPass=true.
func TestQC_NoDuplicateKenBurnsReRender(t *testing.T) {
	runDir := t.TempDir()
	cfg := testConfig()
	cfg.General.RunDir = runDir
	cfg.QC.IdentityThreshold = 0.85

	state := model.NewRunState([]string{"p1.png"}, "ko", "t1")
	p := model.NewPanelJSON("panel_001", model.BBox{X: 0, Y: 0, W: 720, H: 400})
	p.Route = model.RouteKenBurns
	state.Panels = []model.PanelJSON{p}
	state.ClipCandidates["panel_001"] = []model.ClipCandidate{{
		PanelID:    "panel_001",
		Path:       "runs/test/clips/panel_001_clip_01.avi",
		TakeNumber: 1,
		QC: model.QCScores{
			IdentitySimilarity: 0.78, // below 0.85 threshold
			NewTextCount:       0,
			FlickerScore:       0.02,
			MotionScore:        0.05,
			OverallPass:        false,
		},
		Accepted: false,
	}}

	if err := pipeline.QCNode(context.Background(), state, cfg); err != nil {
		t.Fatalf("QCNode failed: %v", err)
	}
	cands := state.ClipCandidates["panel_001"]
	if len(cands) != 1 {
		t.Fatalf("expected 1 candidate (no duplicate kenburns->kenburns re-render), got %d", len(cands))
	}
	if cands[0].QC.OverallPass {
		t.Errorf("expected OverallPass to remain false on sub-threshold candidate")
	}
}
