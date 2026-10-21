package config

import (
	"bytes"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// PipelineConfig holds the full pipeline configuration, loaded from YAML.
type PipelineConfig struct {
	General       GeneralConfig       `yaml:"general"          json:"general"`
	Perception    PerceptionConfig    `yaml:"perception"       json:"perception"`
	Understanding UnderstandingConfig `yaml:"understanding"    json:"understanding"`
	Cleanup       CleanupConfig       `yaml:"cleanup"          json:"cleanup"`
	Generation    GenerationConfig    `yaml:"generation"       json:"generation"`
	Audio         AudioConfig         `yaml:"audio"            json:"audio"`
	Composition   CompositionConfig   `yaml:"composition"      json:"composition"`
	QC            QCConfig            `yaml:"qc"               json:"qc"`
	Cost          CostConfig          `yaml:"cost,omitempty"   json:"cost,omitempty"`
	Worker        WorkerConfig        `yaml:"worker,omitempty" json:"worker,omitempty"`
}

// CostConfig holds compute budget limits across pipeline stages.
type CostConfig struct {
	I2VBudgetS float64 `yaml:"i2v_budget_s,omitempty" json:"i2v_budget_s,omitempty"`
}

// GeneralConfig holds runtime directory and logging configurations.
type GeneralConfig struct {
	RunDir   string `yaml:"run_dir"   json:"run_dir"`
	LogLevel string `yaml:"log_level" json:"log_level"`
}

// PerceptionConfig holds panel and speech bubble detector choices.
type PerceptionConfig struct {
	Detector string `yaml:"detector" json:"detector"` // "magiv3", "opencv"
	Fallback string `yaml:"fallback" json:"fallback"`
}

// UnderstandingConfig holds VLM and parsing schema configurations.
type UnderstandingConfig struct {
	VLM           string `yaml:"vlm"            json:"vlm"`
	SchemaVersion int    `yaml:"schema_version" json:"schema_version"`
}

// CleanupConfig holds inpainter model settings.
type CleanupConfig struct {
	Inpainter string `yaml:"inpainter" json:"inpainter"` // "lama"
}

// GenerationConfig holds video generation model priorities and retry limits.
type GenerationConfig struct {
	PrimaryModel    string   `yaml:"primary_model"               json:"primary_model"`
	VideoPrimary    string   `yaml:"video_primary,omitempty"     json:"video_primary,omitempty"`
	VideoFallback   string   `yaml:"video_fallback,omitempty"    json:"video_fallback,omitempty"`
	FallbackModels  []string `yaml:"fallback_models"             json:"fallback_models"`
	MaxRetries      int      `yaml:"max_retries"                 json:"max_retries"`
	MaxClipDuration float64  `yaml:"max_clip_duration_s"         json:"max_clip_duration_s"`
	MaxI2VClipS     float64  `yaml:"max_i2v_clip_s,omitempty"    json:"max_i2v_clip_s,omitempty"`
	MaxI2VPanels    int      `yaml:"max_i2v_panels,omitempty"    json:"max_i2v_panels,omitempty"`
	MaxI2VTakes     int      `yaml:"max_i2v_takes,omitempty"     json:"max_i2v_takes,omitempty"`
	CalibrationFile string   `yaml:"calibration_file,omitempty"  json:"calibration_file,omitempty"`
	ComfyURL        string   `yaml:"comfy_url,omitempty"         json:"comfy_url,omitempty"`
}

// EffectivePrimaryModel returns VideoPrimary if set, falling back to PrimaryModel.
func (g GenerationConfig) EffectivePrimaryModel() string {
	if g.VideoPrimary != "" {
		return g.VideoPrimary
	}
	return g.PrimaryModel
}

// AudioConfig holds speech synthesis backend settings.
type AudioConfig struct {
	TTSPrimary  string `yaml:"tts_primary"  json:"tts_primary"`
	TTSFallback string `yaml:"tts_fallback" json:"tts_fallback"`
}

// CompositionConfig holds final video assembly parameters.
type CompositionConfig struct {
	OutputFormat   string `yaml:"output_format"   json:"output_format"`
	Resolution     [2]int `yaml:"resolution"      json:"resolution"`
	FPS            int    `yaml:"fps"             json:"fps"`
	TransitionType string `yaml:"transition_type" json:"transition_type"`
}

// QCRouteThresholds holds route-specific QC thresholds.
type QCRouteThresholds struct {
	MinIdentitySimilarity float64 `yaml:"min_identity_similarity" json:"min_identity_similarity"`
	MaxFlickerScore       float64 `yaml:"max_flicker_score"       json:"max_flicker_score"`
	MinMotionScore        float64 `yaml:"min_motion_score"        json:"min_motion_score"`
	MaxMotionScore        float64 `yaml:"max_motion_score"        json:"max_motion_score"`
	MaxWarpError          float64 `yaml:"max_warp_error"          json:"max_warp_error"`
	MinFirstFrameFidelity float64 `yaml:"min_first_frame_fidelity" json:"min_first_frame_fidelity"`
	MaxStyleDrift         float64 `yaml:"max_style_drift"         json:"max_style_drift"`
	MaxRetries            int     `yaml:"max_retries"             json:"max_retries"`
}

// QCConfig holds validation and quality assurance thresholds.
type QCConfig struct {
	IdentityThreshold     float64                      `yaml:"identity_threshold"               json:"identity_threshold"`
	FlickerThreshold      float64                      `yaml:"flicker_threshold"                json:"flicker_threshold"`
	MaxNewTextLines       int                          `yaml:"max_new_text_lines"               json:"max_new_text_lines"`
	MinMotion             float64                      `yaml:"min_motion,omitempty"             json:"min_motion,omitempty"`
	MaxMotion             float64                      `yaml:"max_motion,omitempty"             json:"max_motion,omitempty"`
	MaxWarpError          float64                      `yaml:"max_warp_error,omitempty"         json:"max_warp_error,omitempty"`
	MinFirstFrameFidelity float64                      `yaml:"min_first_frame_fidelity,omitempty" json:"min_first_frame_fidelity,omitempty"`
	MaxStyleDrift         float64                      `yaml:"max_style_drift,omitempty"        json:"max_style_drift,omitempty"`
	RequireTextCheck      bool                         `yaml:"require_text_check,omitempty"     json:"require_text_check,omitempty"`
	PerRoute              map[string]QCRouteThresholds `yaml:"per_route,omitempty"              json:"per_route,omitempty"`
}

// WorkerConfig holds Python AI worker daemon host and port.
type WorkerConfig struct {
	Host string `yaml:"host" json:"host"`
	Port int    `yaml:"port" json:"port"`
}

// Load loads defaultPath and merges the tier override at tierPath on top.
func Load(defaultPath string, tierPath string) (*PipelineConfig, error) {
	if defaultPath == "" {
		return nil, fmt.Errorf("default config path is required")
	}

	defaultData, err := os.ReadFile(defaultPath)
	if err != nil {
		return nil, fmt.Errorf("read default config %q: %w", defaultPath, err)
	}

	var cfg PipelineConfig
	dec := yaml.NewDecoder(bytes.NewReader(defaultData))
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshal default config %q: %w", defaultPath, err)
	}

	if tierPath != "" {
		tierData, err := os.ReadFile(tierPath)
		if err != nil {
			return nil, fmt.Errorf("read tier config %q: %w", tierPath, err)
		}
		if len(bytes.TrimSpace(tierData)) > 0 {
			tierDec := yaml.NewDecoder(bytes.NewReader(tierData))
			tierDec.KnownFields(true)
			if err := tierDec.Decode(&cfg); err != nil {
				return nil, fmt.Errorf("unmarshal tier config %q: %w", tierPath, err)
			}
		}
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("validate config: %w", err)
	}

	return &cfg, nil
}

// Validate performs basic validation to ensure required fields are set and values are valid.
func (c *PipelineConfig) Validate() error {
	if c.Perception.Detector == "" {
		return fmt.Errorf("perception detector must not be empty")
	}
	if c.Understanding.VLM == "" {
		return fmt.Errorf("understanding VLM must not be empty")
	}
	if c.Cleanup.Inpainter == "" {
		return fmt.Errorf("cleanup inpainter must not be empty")
	}
	if c.Generation.EffectivePrimaryModel() == "" {
		return fmt.Errorf("generation primary model must not be empty")
	}
	if c.Generation.PrimaryModel == "" {
		c.Generation.PrimaryModel = c.Generation.VideoPrimary
	}
	if c.Generation.MaxClipDuration <= 0 && c.Generation.MaxI2VClipS > 0 {
		c.Generation.MaxClipDuration = c.Generation.MaxI2VClipS
	}
	if c.Generation.MaxClipDuration <= 0 {
		return fmt.Errorf("generation max clip duration must be positive, got %f", c.Generation.MaxClipDuration)
	}
	if c.Composition.OutputFormat == "" {
		return fmt.Errorf("composition output format must not be empty")
	}
	if c.Composition.Resolution[0] <= 0 || c.Composition.Resolution[1] <= 0 {
		return fmt.Errorf("composition resolution must have positive dimensions, got %v", c.Composition.Resolution)
	}
	if c.Composition.FPS <= 0 {
		return fmt.Errorf("composition FPS must be positive, got %d", c.Composition.FPS)
	}
	if c.Composition.TransitionType != "" && c.Composition.TransitionType != "crossfade" && c.Composition.TransitionType != "cut" {
		return fmt.Errorf("unsupported transition_type %q (must be crossfade or cut)", c.Composition.TransitionType)
	}
	if c.QC.IdentityThreshold < 0 || c.QC.IdentityThreshold > 1 {
		return fmt.Errorf("qc.identity_threshold must be in [0, 1], got %f", c.QC.IdentityThreshold)
	}
	if c.QC.FlickerThreshold < 0 || c.QC.FlickerThreshold > 1 {
		return fmt.Errorf("qc.flicker_threshold must be in [0, 1], got %f", c.QC.FlickerThreshold)
	}
	if c.Worker.Port < 0 || c.Worker.Port > 65535 {
		return fmt.Errorf("worker.port must be in [0, 65535], got %d", c.Worker.Port)
	}
	return nil
}
