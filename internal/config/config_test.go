package config_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/m2a/manhwa2anime/internal/config"
)

func TestLoad_Default(t *testing.T) {
	defaultPath := filepath.Join("..", "..", "configs", "default.yaml")
	cfg, err := config.Load(defaultPath, "")
	if err != nil {
		t.Fatalf("failed to load default config: %v", err)
	}

	if cfg.Perception.Detector != "opencv" {
		t.Errorf("expected perception.detector opencv, got %s", cfg.Perception.Detector)
	}
	if cfg.Generation.PrimaryModel != "wan22-ti2v-5b" {
		t.Errorf("expected primary_model wan22-ti2v-5b, got %s", cfg.Generation.PrimaryModel)
	}
	if cfg.Composition.FPS != 24 {
		t.Errorf("expected fps 24, got %d", cfg.Composition.FPS)
	}
}

func TestLoad_TierOverride(t *testing.T) {
	defaultPath := filepath.Join("..", "..", "configs", "default.yaml")
	tierPath := filepath.Join("..", "..", "configs", "tiers", "t0.yaml")

	cfg, err := config.Load(defaultPath, tierPath)
	if err != nil {
		t.Fatalf("failed to load tier config: %v", err)
	}

	// Overridden by t0.yaml
	if cfg.Generation.PrimaryModel != "kenburns" {
		t.Errorf("expected primary_model kenburns, got %s", cfg.Generation.PrimaryModel)
	}
	if cfg.Understanding.VLM != "qwen3-vl-2b" {
		t.Errorf("expected vlm qwen3-vl-2b, got %s", cfg.Understanding.VLM)
	}

	// Preserved from default.yaml
	if cfg.Perception.Detector != "opencv" {
		t.Errorf("expected detector opencv to persist, got %s", cfg.Perception.Detector)
	}
	if cfg.Composition.FPS != 24 {
		t.Errorf("expected fps 24 to persist, got %d", cfg.Composition.FPS)
	}
}

func TestValidate_InvalidFields(t *testing.T) {
	c := config.PipelineConfig{}
	if err := c.Validate(); err == nil {
		t.Fatal("expected validation error for empty config, got nil")
	}

	// Partially valid
	c.Perception.Detector = "opencv"
	c.Understanding.VLM = "qwen3-vl-8b"
	c.Cleanup.Inpainter = "lama"
	c.Generation.PrimaryModel = "wan22"
	c.Generation.MaxClipDuration = 0 // Invalid <= 0

	if err := c.Validate(); err == nil {
		t.Fatal("expected validation error for zero clip duration, got nil")
	}

	c.Generation.MaxClipDuration = 5.0
	c.Composition.OutputFormat = "mp4"
	c.Composition.Resolution = [2]int{720, 1280}
	c.Composition.FPS = 24

	if err := c.Validate(); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestLoad_MissingFiles(t *testing.T) {
	_, err := config.Load("nonexistent-default.yaml", "")
	if err == nil {
		t.Fatal("expected error loading nonexistent default, got nil")
	}

	tmpDir := t.TempDir()
	validDef := filepath.Join(tmpDir, "def.yaml")
	err = os.WriteFile(validDef, []byte(`
perception:
  detector: "opencv"
understanding:
  vlm: "qwen"
cleanup:
  inpainter: "lama"
generation:
  primary_model: "wan"
  max_clip_duration_s: 4.0
composition:
  output_format: "mp4"
  resolution: [720, 1280]
  fps: 24
`), 0644)
	if err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	_, err = config.Load(validDef, "nonexistent-tier.yaml")
	if err == nil {
		t.Fatal("expected error loading nonexistent tier file, got nil")
	}
}
