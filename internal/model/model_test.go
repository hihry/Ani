package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewPanelJSON(t *testing.T) {
	bbox := BBox{X: 10, Y: 20, W: 300, H: 400}
	p := NewPanelJSON("panel_001", bbox)

	if p.PanelID != "panel_001" {
		t.Fatalf("expected PanelID panel_001, got %s", p.PanelID)
	}
	if p.BBox != bbox {
		t.Fatalf("expected bbox %+v, got %+v", bbox, p.BBox)
	}
	if p.StyleAnchor != "2D anime, flat colors, clean line art" {
		t.Errorf("unexpected StyleAnchor: %s", p.StyleAnchor)
	}
	if p.MotionIntensity != 0.3 {
		t.Errorf("expected MotionIntensity 0.3, got %f", p.MotionIntensity)
	}
	if p.Route != RouteKenBurns {
		t.Errorf("expected Route %s, got %s", RouteKenBurns, p.Route)
	}
	if p.DurationS != 4.0 {
		t.Errorf("expected DurationS 4.0, got %f", p.DurationS)
	}
	if p.Characters == nil || p.Dialogue == nil || p.SFX == nil {
		t.Errorf("expected non-nil slices for Characters, Dialogue, and SFX")
	}
}

func TestNewCharacterBible(t *testing.T) {
	cb := NewCharacterBible()
	if cb == nil {
		t.Fatal("expected non-nil CharacterBible")
	}
	if cb.Characters == nil {
		t.Fatal("expected initialized Characters map")
	}
	cb.Characters["char_1"] = CharacterEntry{
		Name:        "Jinwoo",
		Description: "Protagonist hunter",
		Voice: VoiceProfile{
			VoiceID: "voice_jinwoo",
			Style:   "calm",
		},
		Appearances: []string{"panel_001"},
	}

	if len(cb.Characters) != 1 {
		t.Errorf("expected 1 character entry, got %d", len(cb.Characters))
	}
}

func TestRunStateLifecycle(t *testing.T) {
	pages := []string{"page_01.png", "page_02.png"}
	state := NewRunState(pages, "ko", "t1")

	if state.Config.RunID == "" {
		t.Fatal("expected non-empty RunID")
	}
	if state.Config.SourceLanguage != "ko" {
		t.Errorf("expected lang 'ko', got '%s'", state.Config.SourceLanguage)
	}
	if state.Config.ComputeTier != "t1" {
		t.Errorf("expected tier 't1', got '%s'", state.Config.ComputeTier)
	}
	if state.CurrentStage != StageIngest {
		t.Errorf("expected current stage %s, got %s", StageIngest, state.CurrentStage)
	}
	if len(state.StageHistory) != 1 || state.StageHistory[0].Stage != StageIngest {
		t.Fatalf("expected initial stage transition to %s", StageIngest)
	}

	// Advance stage
	state.AdvanceTo(StageSegment)
	if state.CurrentStage != StageSegment {
		t.Errorf("expected stage %s, got %s", StageSegment, state.CurrentStage)
	}
	if len(state.StageHistory) != 2 {
		t.Fatalf("expected 2 stage history entries, got %d", len(state.StageHistory))
	}
	if state.StageHistory[0].EndedAt == nil {
		t.Errorf("expected previous stage to have EndedAt timestamp")
	}

	// Record Error
	errMsg := "failed to segment panel 4"
	state.RecordError(errMsg)
	if len(state.Errors) != 1 || state.Errors[0] != errMsg {
		t.Errorf("error not recorded in state.Errors")
	}
	if state.StageHistory[1].Error != errMsg {
		t.Errorf("error not recorded in active stage history entry")
	}

	// Record Costs
	state.RecordStageCost("SEGMENT", 12.5)
	state.RecordModelCost("wan22-ti2v-5b", 34.0)

	if state.CostLedger.PerStage["SEGMENT"] != 12.5 {
		t.Errorf("expected PerStage['SEGMENT'] == 12.5, got %f", state.CostLedger.PerStage["SEGMENT"])
	}
	if state.CostLedger.TotalWallSeconds != 12.5 {
		t.Errorf("expected TotalWallSeconds == 12.5, got %f", state.CostLedger.TotalWallSeconds)
	}
	if state.CostLedger.PerModel["wan22-ti2v-5b"] != 34.0 {
		t.Errorf("expected PerModel['wan22-ti2v-5b'] == 34.0, got %f", state.CostLedger.PerModel["wan22-ti2v-5b"])
	}
	if state.CostLedger.TotalGPUSeconds != 34.0 {
		t.Errorf("expected TotalGPUSeconds == 34.0, got %f", state.CostLedger.TotalGPUSeconds)
	}
}

func TestRunStateContentHash(t *testing.T) {
	state := NewRunState([]string{"p1.png"}, "ko", "t0")
	hash1 := state.ContentHash()
	if hash1 == "" {
		t.Fatal("expected non-empty content hash")
	}

	// Adding a panel should alter the content hash
	panel := NewPanelJSON("p_1", BBox{X: 0, Y: 0, W: 100, H: 100})
	state.Panels = append(state.Panels, panel)

	hash2 := state.ContentHash()
	if hash2 == hash1 {
		t.Errorf("expected content hash to change after adding panel")
	}
}

func TestRunStateSaveAndLoad(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "m2a-state-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	state := NewRunState([]string{"p1.png"}, "ko", "t0")
	panel := NewPanelJSON("p_1", BBox{X: 10, Y: 20, W: 100, H: 200})
	state.Panels = append(state.Panels, panel)
	state.CleanPlatePaths["p_1"] = "/plates/p_1_clean.png"
	state.ClipCandidates["p_1"] = []ClipCandidate{
		{
			PanelID:    "p_1",
			Path:       "/clips/p_1_take1.mp4",
			TakeNumber: 1,
			Accepted:   true,
			QC: QCScores{
				IdentitySimilarity: 0.92,
				NewTextCount:       0,
				FlickerScore:       0.05,
				MotionScore:        0.88,
				OverallPass:        true,
			},
		},
	}
	state.AudioTracks = append(state.AudioTracks, AudioTrack{
		PanelID:   "p_1",
		SpeakerID: "jinwoo",
		AudioPath: "/audio/p_1_jinwoo.wav",
		DurationS: 2.5,
		Emotion:   "calm",
	})

	if err := state.Save(tmpDir); err != nil {
		t.Fatalf("failed to save state: %v", err)
	}

	// Verify state.json exists
	if _, err := os.Stat(filepath.Join(tmpDir, "state.json")); os.IsNotExist(err) {
		t.Fatalf("expected state.json to exist in %s", tmpDir)
	}

	// Overwrite save to ensure atomic overwrite works
	state.AdvanceTo(StageUnderstand)
	if err := state.Save(tmpDir); err != nil {
		t.Fatalf("failed to overwrite save state: %v", err)
	}

	// Load state back
	loaded, err := LoadRunState(tmpDir)
	if err != nil {
		t.Fatalf("failed to load state: %v", err)
	}

	if loaded.Config.RunID != state.Config.RunID {
		t.Errorf("expected RunID %s, got %s", state.Config.RunID, loaded.Config.RunID)
	}
	if loaded.CurrentStage != StageUnderstand {
		t.Errorf("expected current stage %s, got %s", StageUnderstand, loaded.CurrentStage)
	}
	if len(loaded.Panels) != 1 || loaded.Panels[0].PanelID != "p_1" {
		t.Errorf("panel data mismatch: %+v", loaded.Panels)
	}
	if loaded.CleanPlatePaths["p_1"] != "/plates/p_1_clean.png" {
		t.Errorf("clean plate path mismatch")
	}
	if len(loaded.ClipCandidates["p_1"]) != 1 || !loaded.ClipCandidates["p_1"][0].Accepted {
		t.Errorf("clip candidate mismatch")
	}
	if len(loaded.AudioTracks) != 1 || loaded.AudioTracks[0].SpeakerID != "jinwoo" {
		t.Errorf("audio track mismatch")
	}
}

func TestRunStateSave_RejectsInvalidDurationAndLang(t *testing.T) {
	tmpDir := t.TempDir()
	state := NewRunState([]string{"p1.png"}, "ko", "t0")
	p := NewPanelJSON("p_1", BBox{X: 0, Y: 0, W: 100, H: 100})
	p.DurationS = 12.3 // exceeds MaxPanelDuration (10.0)
	state.Panels = []PanelJSON{p}

	if err := state.Save(tmpDir); err == nil {
		t.Fatal("expected Save to reject duration_s=12.3, got nil")
	}

	state.Panels[0].DurationS = ClampDuration(12.3)
	if state.Panels[0].DurationS != 10.0 {
		t.Fatalf("expected ClampDuration(12.3) == 10.0, got %f", state.Panels[0].DurationS)
	}
	if err := state.Save(tmpDir); err != nil {
		t.Fatalf("expected clamped duration to save cleanly, got %v", err)
	}

	state.Config.SourceLanguage = "zh"
	if err := state.Save(tmpDir); err == nil {
		t.Fatal("expected Save to reject unsupported language 'zh', got nil")
	}
}
