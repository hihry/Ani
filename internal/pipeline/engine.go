package pipeline

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
)

// NodeFunc is the signature for a pipeline stage handler.
// It receives the current context, mutable run state, and pipeline config.
type NodeFunc func(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error

// Engine orchestrates the pipeline execution state machine.
type Engine struct {
	nodes                  map[model.PipelineStage]NodeFunc
	fallbacks              map[model.PipelineStage]NodeFunc
	order                  []model.PipelineStage
	runDir                 string
	logger                 *slog.Logger
	autoApproveCheckpoints bool
}

// NewEngine initializes an Engine with the canonical stage order:
// INGEST → SEGMENT → UNDERSTAND → CHECKPOINT_1 → CLEAN_PLATES → ROUTE → TTS → CHECKPOINT_2 → GENERATE → QC → ASSEMBLE → DONE.
func NewEngine(runDir string, logger *slog.Logger) *Engine {
	if logger == nil {
		logger = slog.Default()
	}

	return &Engine{
		nodes:     make(map[model.PipelineStage]NodeFunc),
		fallbacks: make(map[model.PipelineStage]NodeFunc),
		order: []model.PipelineStage{
			model.StageIngest,
			model.StageSegment,
			model.StageUnderstand,
			model.StageCheckpoint1,
			model.StageCleanPlates,
			model.StageRoute,
			model.StageTTS,
			model.StageCheckpoint2,
			model.StageGenerate,
			model.StageQC,
			model.StageAssemble,
			model.StageDone,
		},
		runDir: runDir,
		logger: logger,
	}
}

// SetAutoApproveCheckpoints configures whether CHECKPOINT_1 and CHECKPOINT_2 automatically advance without pausing.
func (e *Engine) SetAutoApproveCheckpoints(auto bool) {
	e.autoApproveCheckpoints = auto
}

// RegisterNode registers a handler for a given pipeline stage.
func (e *Engine) RegisterNode(stage model.PipelineStage, fn NodeFunc) {
	e.nodes[stage] = fn
}

// RegisterFallback registers an optional fallback handler for a given stage if the primary node fails.
func (e *Engine) RegisterFallback(stage model.PipelineStage, fn NodeFunc) {
	e.fallbacks[stage] = fn
}

// Run executes all stages from the beginning.
func (e *Engine) Run(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	if state == nil {
		return errors.New("state cannot be nil")
	}
	if cfg == nil {
		return errors.New("config cannot be nil")
	}
	if len(e.order) == 0 {
		return errors.New("engine has no stages configured")
	}

	if state.CurrentStage != e.order[0] {
		state.AdvanceTo(e.order[0])
		if err := state.Save(e.runDir); err != nil {
			return fmt.Errorf("failed to save initial state: %w", err)
		}
	}

	return e.execute(ctx, state, cfg, 0)
}

// Resume resumes execution from state.CurrentStage, skipping already-completed stages.
func (e *Engine) Resume(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	if state == nil {
		return errors.New("state cannot be nil")
	}
	if cfg == nil {
		return errors.New("config cannot be nil")
	}

	if state.CurrentStage == model.StageDone {
		e.logger.Info("Pipeline run is already complete", "stage", state.CurrentStage)
		return nil
	}

	if state.CurrentStage == model.StageCheckpoint1 {
		idx := e.indexOfStage(model.StageCheckpoint1)
		if idx >= 0 && idx+1 < len(e.order) {
			nextStage := e.order[idx+1]
			e.logger.Info("Resuming from checkpoint 1; advancing to next stage", "next_stage", nextStage)
			state.AdvanceTo(nextStage)
			if err := state.Save(e.runDir); err != nil {
				return fmt.Errorf("failed to save state on resume from checkpoint 1: %w", err)
			}
			return e.execute(ctx, state, cfg, idx+1)
		}
	}

	if state.CurrentStage == model.StageCheckpoint2 {
		idx := e.indexOfStage(model.StageCheckpoint2)
		if idx >= 0 && idx+1 < len(e.order) {
			nextStage := e.order[idx+1]
			e.logger.Info("Resuming from checkpoint 2; advancing to next stage", "next_stage", nextStage)
			state.AdvanceTo(nextStage)
			if err := state.Save(e.runDir); err != nil {
				return fmt.Errorf("failed to save state on resume from checkpoint 2: %w", err)
			}
			return e.execute(ctx, state, cfg, idx+1)
		}
	}

	if state.CurrentStage == model.StageFailed {
		var retryStage model.PipelineStage
		for i := len(state.StageHistory) - 1; i >= 0; i-- {
			st := state.StageHistory[i].Stage
			if st != model.StageFailed && st != model.StageDone {
				retryStage = st
				break
			}
		}
		if retryStage == "" {
			retryStage = e.order[0]
		}
		idx := e.indexOfStage(retryStage)
		if idx < 0 {
			idx = 0
			retryStage = e.order[0]
		}
		e.logger.Info("Resuming failed run at retry stage", "stage", retryStage)
		state.AdvanceTo(retryStage)
		if err := state.Save(e.runDir); err != nil {
			return fmt.Errorf("failed to save state on resume retry: %w", err)
		}
		return e.execute(ctx, state, cfg, idx)
	}

	idx := e.indexOfStage(state.CurrentStage)
	if idx < 0 {
		return fmt.Errorf("unknown current stage in execution order: %q", state.CurrentStage)
	}

	e.logger.Info("Resuming pipeline from current stage", "stage", state.CurrentStage, "index", idx)
	return e.execute(ctx, state, cfg, idx)
}

func (e *Engine) execute(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig, startIdx int) error {
	for i := startIdx; i < len(e.order); i++ {
		stage := e.order[i]

		if err := ctx.Err(); err != nil {
			state.RecordError(fmt.Sprintf("pipeline cancelled at stage %s: %v", stage, err))
			_ = state.Save(e.runDir)
			return err
		}

		if stage == model.StageDone {
			e.logger.Info("Pipeline completed successfully", "stage", stage)
			state.CurrentStage = model.StageDone
			now := time.Now().UTC()
			if len(state.StageHistory) > 0 {
				lastIdx := len(state.StageHistory) - 1
				if state.StageHistory[lastIdx].EndedAt == nil {
					state.StageHistory[lastIdx].EndedAt = &now
				}
			}
			if err := state.Save(e.runDir); err != nil {
				return fmt.Errorf("failed to save completed state: %w", err)
			}
			return nil
		}

		if stage == model.StageCheckpoint1 || stage == model.StageCheckpoint2 {
			if e.autoApproveCheckpoints {
				e.logger.Info(fmt.Sprintf("Auto-approving checkpoint %s", stage), "stage", stage)
				if i+1 < len(e.order) {
					nextStage := e.order[i+1]
					state.AdvanceTo(nextStage)
					_ = state.Save(e.runDir)
				}
				continue
			}
			e.logger.Info(fmt.Sprintf("Reached checkpoint %s; pausing for human review", stage), "stage", stage)
			state.CurrentStage = stage
			if err := state.Save(e.runDir); err != nil {
				return fmt.Errorf("failed to save state at checkpoint %s: %w", stage, err)
			}
			return nil
		}

		e.logger.Info(fmt.Sprintf("Starting stage %s", stage), "stage", stage)

		fn, ok := e.nodes[stage]
		if !ok || fn == nil {
			err := fmt.Errorf("no node handler registered for stage %s", stage)
			state.RecordError(err.Error())
			state.AdvanceTo(model.StageFailed)
			_ = state.Save(e.runDir)
			return err
		}

		start := time.Now()
		err := fn(ctx, state, cfg)
		duration := time.Since(start)

		state.RecordStageCost(string(stage), duration.Seconds())

		if saveErr := state.Save(e.runDir); saveErr != nil {
			e.logger.Error("Failed to save state checkpoint", "stage", stage, "error", saveErr)
		}

		if err != nil {
			state.RecordError(fmt.Sprintf("stage %s failed: %v", stage, err))
			e.logger.Error("Stage execution failed", "stage", stage, "error", err)

			var fallbackErr error = err
			if fallbackFn, fbOk := e.fallbacks[stage]; fbOk && fallbackFn != nil {
				e.logger.Warn("Attempting fallback for stage", "stage", stage)
				fbStart := time.Now()
				if fbErr := fallbackFn(ctx, state, cfg); fbErr == nil {
					fallbackErr = nil
					fbDuration := time.Since(fbStart)
					state.RecordStageCost(string(stage)+"_fallback", fbDuration.Seconds())
					_ = state.Save(e.runDir)
					e.logger.Info("Stage fallback succeeded", "stage", stage, "duration_s", fbDuration.Seconds())
				} else {
					fallbackErr = fbErr
					state.RecordError(fmt.Sprintf("stage %s fallback failed: %v", stage, fbErr))
					e.logger.Error("Stage fallback failed", "stage", stage, "error", fbErr)
				}
			}

			if fallbackErr != nil {
				state.AdvanceTo(model.StageFailed)
				_ = state.Save(e.runDir)
				return fmt.Errorf("stage %s failed: %w", stage, fallbackErr)
			}
		}

		e.logger.Info(fmt.Sprintf("Completed stage %s in %.2f seconds", stage, duration.Seconds()),
			"stage", stage, "duration_s", duration.Seconds())

		if i+1 < len(e.order) {
			nextStage := e.order[i+1]
			state.AdvanceTo(nextStage)
			if saveErr := state.Save(e.runDir); saveErr != nil {
				e.logger.Error("Failed to save state after advancing stage", "stage", nextStage, "error", saveErr)
			}
		}
	}

	return nil
}

func (e *Engine) indexOfStage(stage model.PipelineStage) int {
	for i, s := range e.order {
		if s == stage {
			return i
		}
	}
	return -1
}
