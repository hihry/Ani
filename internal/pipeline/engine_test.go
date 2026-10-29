package pipeline_test

import (
	"context"
	"encoding/json"
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
	if !cands[0].Accepted {
		t.Errorf("expected terminal fallback best candidate to be accepted for assembly")
	}
	if !state.Panels[0].NeedsReview {
		t.Errorf("expected state.Panels[0].NeedsReview to be set true on terminal fallback QC failure")
	}
	if len(state.Errors) == 0 {
		t.Errorf("expected state.Errors to record terminal fallback QC warning")
	}
}

// TestCheckpoint1HumanEditSyncOnResume verifies Phase 2 CP1: human edits to understanding/<panel_id>.json are loaded into state on Resume.
func TestCheckpoint1HumanEditSyncOnResume(t *testing.T) {
	runDir := t.TempDir()
	logger := discardLogger()
	cfg := testConfig()
	cfg.General.RunDir = runDir

	state := model.NewRunState([]string{"p1.png"}, "en", "t1")
	engine := pipeline.NewEngine(runDir, logger)
	registerMockNodes(engine, runDir)

	if err := engine.Run(context.Background(), state, cfg); err != nil {
		t.Fatalf("engine.Run to CP1 failed: %v", err)
	}
	if state.CurrentStage != model.StageCheckpoint1 {
		t.Fatalf("expected CHECKPOINT_1, got %s", state.CurrentStage)
	}

	// Simulate human reviewer editing understanding/panel_001.json while paused at CP1
	editedPanel := state.Panels[0]
	editedPanel.Dialogue = []model.DialogueLine{{
		SpeakerID:  "char_jin",
		Text:       "Edited at CP1 in Go!",
		Lang:       "en",
		Emotion:    "confident",
		Confidence: 1.0,
	}}
	panelBytes, _ := json.MarshalIndent(editedPanel, "", "  ")
	vlmDir := filepath.Join(runDir, "understanding")
	_ = os.MkdirAll(vlmDir, 0755)
	if err := os.WriteFile(filepath.Join(vlmDir, "panel_001.json"), panelBytes, 0644); err != nil {
		t.Fatalf("write panel_001.json failed: %v", err)
	}

	if err := engine.Resume(context.Background(), state, cfg); err != nil {
		t.Fatalf("engine.Resume from CP1 failed: %v", err)
	}
	if len(state.Panels[0].Dialogue) != 1 || state.Panels[0].Dialogue[0].Text != "Edited at CP1 in Go!" {
		t.Fatalf("expected CP1 human edit to sync into state.Panels[0], got %+v", state.Panels[0].Dialogue)
	}
}

// TestAdversarial_CP1CorruptAndOutOfBoundsEdits verifies that Resume from CP1 clamps out-of-range duration_s/motion_intensity
// and gracefully survives corrupted JSON in understanding/panel_002.json without losing panel_002.
func TestAdversarial_CP1CorruptAndOutOfBoundsEdits(t *testing.T) {
	runDir := t.TempDir()
	logger := discardLogger()
	cfg := testConfig()
	cfg.General.RunDir = runDir

	state := model.NewRunState([]string{"p1.png"}, "en", "t1")
	engine := pipeline.NewEngine(runDir, logger)
	registerMockNodes(engine, runDir)

	if err := engine.Run(context.Background(), state, cfg); err != nil {
		t.Fatalf("engine.Run to CP1 failed: %v", err)
	}

	vlmDir := filepath.Join(runDir, "understanding")
	_ = os.MkdirAll(vlmDir, 0755)

	// Panel 1: Out-of-bounds DurationS (88.0) and MotionIntensity (9.5) + unsafe SpeakerID with slashes
	editedP1 := state.Panels[0]
	editedP1.DurationS = 88.0
	editedP1.MotionIntensity = 9.5
	editedP1.Dialogue = []model.DialogueLine{{
		SpeakerID:  "Jin / Shadow:King..!",
		Text:       "Clamped and sanitized!",
		Lang:       "en",
		Emotion:    "fierce",
		Confidence: 0.99,
	}}
	p1Bytes, _ := json.MarshalIndent(editedP1, "", "  ")
	_ = os.WriteFile(filepath.Join(vlmDir, "panel_001.json"), p1Bytes, 0644)

	// Panel 2: Corrupted JSON file on disk
	_ = os.WriteFile(filepath.Join(vlmDir, "panel_002.json"), []byte("{corrupt_json_payload"), 0644)

	if err := engine.Resume(context.Background(), state, cfg); err != nil {
		t.Fatalf("engine.Resume from CP1 failed on adversarial edits: %v", err)
	}

	if state.Panels[0].DurationS != 10.0 {
		t.Fatalf("expected DurationS clamped to 10.0, got %.2f", state.Panels[0].DurationS)
	}
	if state.Panels[0].MotionIntensity != 1.0 {
		t.Fatalf("expected MotionIntensity clamped to 1.0, got %.2f", state.Panels[0].MotionIntensity)
	}
	if state.Panels[1].PanelID != "panel_002" {
		t.Fatalf("expected panel_002 preserved after corrupt JSON skip, got %q", state.Panels[1].PanelID)
	}
}

// TestCheckpoint2PreservesTTSDurationOnResume verifies P2-02: resuming from CHECKPOINT_2 does not clobber
// TTS-synchronized panel durations with pre-TTS 2.5s durations from RouteNode.
func TestCheckpoint2PreservesTTSDurationOnResume(t *testing.T) {
	runDir := t.TempDir()
	logger := discardLogger()
	cfg := testConfig()
	cfg.General.RunDir = runDir

	state := model.NewRunState([]string{"p1.png"}, "en", "t1")
	engine := pipeline.NewEngine(runDir, logger)
	registerMockNodes(engine, runDir)

	// Override Route to write 2.5s to routes/routing_plan.json (like RouteNode)
	engine.RegisterNode(model.StageRoute, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		routesDir := filepath.Join(runDir, "routes")
		_ = os.MkdirAll(routesDir, 0755)
		for i := range s.Panels {
			s.Panels[i].DurationS = 2.5
			s.Panels[i].Route = model.RouteKenBurns
		}
		b, _ := json.MarshalIndent(s.Panels, "", "  ")
		return os.WriteFile(filepath.Join(routesDir, "routing_plan.json"), b, 0644)
	})

	// Override TTS to extend DurationS to 6.4s and update routes/routing_plan.json (like fixed TTSNode)
	engine.RegisterNode(model.StageTTS, func(ctx context.Context, s *model.RunState, c *config.PipelineConfig) error {
		routesDir := filepath.Join(runDir, "routes")
		_ = os.MkdirAll(routesDir, 0755)
		s.Panels[0].DurationS = 6.4
		b, _ := json.MarshalIndent(s.Panels, "", "  ")
		return os.WriteFile(filepath.Join(routesDir, "routing_plan.json"), b, 0644)
	})

	_ = engine.Run(context.Background(), state, cfg)    // pauses at CP1
	_ = engine.Resume(context.Background(), state, cfg) // runs CLEAN_PLATES, ROUTE, TTS -> pauses at CP2
	if state.CurrentStage != model.StageCheckpoint2 {
		t.Fatalf("expected CHECKPOINT_2, got %s", state.CurrentStage)
	}
	if state.Panels[0].DurationS != 6.4 {
		t.Fatalf("expected DurationS=6.4 at CP2, got %.2f", state.Panels[0].DurationS)
	}

	// Resume from CP2 -> GENERATE -> QC -> ASSEMBLE -> DONE
	if err := engine.Resume(context.Background(), state, cfg); err != nil {
		t.Fatalf("Resume from CP2 failed: %v", err)
	}
	if state.Panels[0].DurationS != 6.4 {
		t.Fatalf("expected DurationS=6.4 preserved after CP2 resume, got %.2f", state.Panels[0].DurationS)
	}
}

func TestRouteNodeBudgetKnapsackAndHeroScore(t *testing.T) {
	runDir := t.TempDir()
	cfg := testConfig()
	cfg.General.RunDir = runDir
	cfg.Generation.PrimaryModel = "wan22-ti2v-5b"
	cfg.Generation.MaxI2VPanels = 2
	cfg.Cost.I2VBudgetS = 120.0

	state := model.NewRunState([]string{"p1.png"}, "ko", "t1")
	p1 := model.NewPanelJSON("panel_001", model.BBox{X: 0, Y: 0, W: 720, H: 900})
	p1.Shot = model.ShotCloseUp
	p1.MotionIntensity = 0.85
	p1.DurationS = 2.5

	p2 := model.NewPanelJSON("panel_002", model.BBox{X: 0, Y: 900, W: 720, H: 900})
	p2.Shot = model.ShotSplash
	p2.MotionIntensity = 0.90
	p2.DurationS = 2.5

	p3 := model.NewPanelJSON("panel_003", model.BBox{X: 0, Y: 1800, W: 720, H: 900})
	p3.Shot = model.ShotCloseUp
	p3.MotionIntensity = 0.75
	p3.DurationS = 2.5

	p4 := model.NewPanelJSON("panel_004", model.BBox{X: 0, Y: 2700, W: 720, H: 400})
	p4.Shot = model.ShotWide
	p4.MotionIntensity = 0.15
	p4.DurationS = 2.5

	state.Panels = []model.PanelJSON{p1, p2, p3, p4}

	if err := pipeline.RouteNode(context.Background(), state, cfg); err != nil {
		t.Fatalf("RouteNode failed: %v", err)
	}

	i2vCount := 0
	for _, p := range state.Panels {
		if p.Route == model.RouteI2V {
			i2vCount++
		}
	}
	if i2vCount != 2 {
		t.Fatalf("expected MaxI2VPanels=2 to cap I2V panels at 2 (out of 3 hero panels), got %d", i2vCount)
	}
	if state.Panels[3].Route != model.RouteKenBurns {
		t.Fatalf("expected low-motion wide panel_004 to route to kenburns, got %s", state.Panels[3].Route)
	}

	planBytes, err := os.ReadFile(filepath.Join(runDir, "routes", "routing_plan.json"))
	if err != nil {
		t.Fatalf("read routing_plan.json failed: %v", err)
	}
	var plan pipeline.RoutingPlan
	if err := json.Unmarshal(planBytes, &plan); err != nil {
		t.Fatalf("unmarshal RoutingPlan failed: %v", err)
	}
	if len(plan.Panels) != 4 || plan.MaxI2VPanels != 2 {
		t.Fatalf("unexpected RoutingPlan: %+v", plan)
	}
}
