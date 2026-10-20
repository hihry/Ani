package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
)

// PipelineStage represents which stage the pipeline is currently in.
type PipelineStage string

const (
	StageIngest      PipelineStage = "INGEST"
	StageSegment     PipelineStage = "SEGMENT"
	StageUnderstand  PipelineStage = "UNDERSTAND"
	StageCheckpoint1 PipelineStage = "CHECKPOINT_1"
	StageCleanPlates PipelineStage = "CLEAN_PLATES"
	StageRoute       PipelineStage = "ROUTE"
	StageCheckpoint2 PipelineStage = "CHECKPOINT_2"
	StageGenerate    PipelineStage = "GENERATE"
	StageQC          PipelineStage = "QC"
	StageTTS         PipelineStage = "TTS"
	StageAssemble    PipelineStage = "ASSEMBLE"
	StageDone        PipelineStage = "DONE"
	StageFailed      PipelineStage = "FAILED"
)

// QCScores holds quality metrics for a generated clip.
type QCScores struct {
	IdentitySimilarity float64 `json:"identity_similarity"`
	NewTextCount       int     `json:"new_text_count"`
	FlickerScore       float64 `json:"flicker_score"`
	MotionScore        float64 `json:"motion_score"`
	OverallPass        bool    `json:"overall_pass"`
}

// ClipCandidate represents one attempt at generating a video clip for a panel.
type ClipCandidate struct {
	PanelID    string   `json:"panel_id"`
	Path       string   `json:"path"`
	TakeNumber int      `json:"take_number"`
	QC         QCScores `json:"qc_scores"`
	Accepted   bool     `json:"accepted"`
}

// AudioTrack represents a generated TTS audio file.
type AudioTrack struct {
	PanelID   string  `json:"panel_id"`
	SpeakerID string  `json:"speaker_id"`
	AudioPath string  `json:"audio_path"`
	DurationS float64 `json:"duration_s"`
	Emotion   string  `json:"emotion"`
}

// CostLedger tracks compute costs across models and pipeline stages.
type CostLedger struct {
	TotalGPUSeconds  float64            `json:"total_gpu_seconds"`
	TotalWallSeconds float64            `json:"total_wall_seconds"`
	PerModel         map[string]float64 `json:"per_model_seconds"`
	PerStage         map[string]float64 `json:"per_stage_seconds"`
}

// RunConfig holds immutable configuration for a pipeline run.
type RunConfig struct {
	RunID            string    `json:"run_id"`
	ConfigHash       string    `json:"config_hash"`
	SourcePages      []string  `json:"source_pages"`
	SourceLanguage   string    `json:"source_language"`   // "ko", "ja", "en"
	OutputResolution [2]int    `json:"output_resolution"` // [width, height], e.g. [720, 1280]
	ComputeTier      string    `json:"compute_tier"`      // "t0", "t1", "t2"
	CreatedAt        time.Time `json:"created_at"`
}

// RunState is the full mutable state of a pipeline run.
type RunState struct {
	Config          RunConfig                  `json:"config"`
	CurrentStage    PipelineStage              `json:"current_stage"`
	Panels          []PanelJSON                `json:"panels"`
	CharacterBible  *CharacterBible            `json:"character_bible,omitempty"`
	CleanPlatePaths map[string]string          `json:"clean_plate_paths"` // panel_id -> path
	ClipCandidates  map[string][]ClipCandidate `json:"clip_candidates"`  // panel_id -> candidates
	AudioTracks     []AudioTrack               `json:"audio_tracks"`
	CostLedger      CostLedger                 `json:"cost_ledger"`
	Errors          []string                   `json:"errors"`
	StageHistory    []StageTransition          `json:"stage_history"`
}

// StageTransition records when a stage started and ended.
type StageTransition struct {
	Stage     PipelineStage `json:"stage"`
	StartedAt time.Time     `json:"started_at"`
	EndedAt   *time.Time    `json:"ended_at,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// NewRunState creates a new run with a unique UUID, sets stage to INGEST, and initializes all collections.
func NewRunState(pages []string, lang string, tier string) *RunState {
	if lang == "" {
		lang = "ko"
	}
	if tier == "" {
		tier = "t0"
	}

	sourcePages := make([]string, len(pages))
	copy(sourcePages, pages)

	now := time.Now().UTC()
	runID := uuid.New().String()

	// Compute initial configuration hash
	h := sha256.New()
	for _, p := range sourcePages {
		h.Write([]byte(p))
	}
	h.Write([]byte(lang))
	h.Write([]byte(tier))
	configHash := hex.EncodeToString(h.Sum(nil))

	cfg := RunConfig{
		RunID:            runID,
		ConfigHash:       configHash,
		SourcePages:      sourcePages,
		SourceLanguage:   lang,
		OutputResolution: [2]int{720, 1280},
		ComputeTier:      tier,
		CreatedAt:        now,
	}

	return &RunState{
		Config:          cfg,
		CurrentStage:    StageIngest,
		Panels:          make([]PanelJSON, 0),
		CharacterBible:  NewCharacterBible(),
		CleanPlatePaths: make(map[string]string),
		ClipCandidates:  make(map[string][]ClipCandidate),
		AudioTracks:     make([]AudioTrack, 0),
		CostLedger: CostLedger{
			PerModel: make(map[string]float64),
			PerStage: make(map[string]float64),
		},
		Errors: make([]string, 0),
		StageHistory: []StageTransition{
			{
				Stage:     StageIngest,
				StartedAt: now,
			},
		},
	}
}

// Save atomically writes the RunState to state.json in runDir using a temporary file and rename.
func (s *RunState) Save(runDir string) error {
	if err := os.MkdirAll(runDir, 0755); err != nil {
		return fmt.Errorf("create run dir %q: %w", runDir, err)
	}

	targetPath := filepath.Join(runDir, "state.json")
	tmpPath := filepath.Join(runDir, fmt.Sprintf("state-%d.json.tmp", time.Now().UnixNano()))

	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal run state: %w", err)
	}

	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("write temp state file %q: %w", tmpPath, err)
	}

	// Atomic rename to targetPath
	if err := os.Rename(tmpPath, targetPath); err != nil {
		// On Windows, if destination exists or rename fails, remove and retry
		_ = os.Remove(targetPath)
		if retryErr := os.Rename(tmpPath, targetPath); retryErr != nil {
			_ = os.Remove(tmpPath)
			return fmt.Errorf("atomic rename from %q to %q: %w", tmpPath, targetPath, retryErr)
		}
	}

	return nil
}

// LoadRunState reads and deserializes state.json from runDir.
func LoadRunState(runDir string) (*RunState, error) {
	targetPath := filepath.Join(runDir, "state.json")
	data, err := os.ReadFile(targetPath)
	if err != nil {
		return nil, fmt.Errorf("read state file %q: %w", targetPath, err)
	}

	var state RunState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("unmarshal state file %q: %w", targetPath, err)
	}

	// Guarantee non-nil slices and maps for safe usage
	if state.CleanPlatePaths == nil {
		state.CleanPlatePaths = make(map[string]string)
	}
	if state.ClipCandidates == nil {
		state.ClipCandidates = make(map[string][]ClipCandidate)
	}
	if state.CostLedger.PerModel == nil {
		state.CostLedger.PerModel = make(map[string]float64)
	}
	if state.CostLedger.PerStage == nil {
		state.CostLedger.PerStage = make(map[string]float64)
	}
	if state.Panels == nil {
		state.Panels = make([]PanelJSON, 0)
	}
	if state.AudioTracks == nil {
		state.AudioTracks = make([]AudioTrack, 0)
	}
	if state.Errors == nil {
		state.Errors = make([]string, 0)
	}
	if state.StageHistory == nil {
		state.StageHistory = make([]StageTransition, 0)
	}

	return &state, nil
}

// ContentHash returns the SHA256 hex string of the JSON-serialized panels and config.
func (s *RunState) ContentHash() string {
	type hashPayload struct {
		Config RunConfig   `json:"config"`
		Panels []PanelJSON `json:"panels"`
	}

	payload := hashPayload{
		Config: s.Config,
		Panels: s.Panels,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return ""
	}

	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// AdvanceTo transitions the pipeline to a new stage, recording completion of the prior stage in history.
func (s *RunState) AdvanceTo(stage PipelineStage) {
	now := time.Now().UTC()
	if len(s.StageHistory) > 0 {
		lastIdx := len(s.StageHistory) - 1
		if s.StageHistory[lastIdx].EndedAt == nil {
			s.StageHistory[lastIdx].EndedAt = &now
		}
	}

	s.CurrentStage = stage
	s.StageHistory = append(s.StageHistory, StageTransition{
		Stage:     stage,
		StartedAt: now,
	})
}

// RecordError appends an error message to the run state and annotates the active stage history entry.
func (s *RunState) RecordError(msg string) {
	s.Errors = append(s.Errors, msg)
	if len(s.StageHistory) > 0 {
		lastIdx := len(s.StageHistory) - 1
		if s.StageHistory[lastIdx].Error == "" {
			s.StageHistory[lastIdx].Error = msg
		}
	}
}

// RecordStageCost adds wall-clock execution duration for a stage to the cost ledger.
func (s *RunState) RecordStageCost(stage string, seconds float64) {
	if s.CostLedger.PerStage == nil {
		s.CostLedger.PerStage = make(map[string]float64)
	}
	s.CostLedger.PerStage[stage] += seconds
	s.CostLedger.TotalWallSeconds += seconds
}

// RecordModelCost adds GPU execution duration for a model to the cost ledger.
func (s *RunState) RecordModelCost(model string, seconds float64) {
	if s.CostLedger.PerModel == nil {
		s.CostLedger.PerModel = make(map[string]float64)
	}
	s.CostLedger.PerModel[model] += seconds
	s.CostLedger.TotalGPUSeconds += seconds
}
