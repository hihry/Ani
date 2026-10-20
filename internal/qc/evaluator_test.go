package qc_test

import (
	"testing"

	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
	"github.com/m2a/manhwa2anime/internal/qc"
)

func TestEvaluateClip_PassAndFail(t *testing.T) {
	cfg := config.QCConfig{
		IdentityThreshold: 0.85,
		FlickerThreshold:  0.15,
		MaxNewTextLines:   0,
	}

	good := model.QCScores{
		IdentitySimilarity: 0.94,
		NewTextCount:       0,
		FlickerScore:       0.04,
		MotionScore:        0.45,
	}
	passed, reasons := qc.EvaluateClip(good, cfg)
	if !passed || len(reasons) != 0 {
		t.Fatalf("expected good scores to pass QC, got passed=%v reasons=%v", passed, reasons)
	}

	bad := model.QCScores{
		IdentitySimilarity: 0.72,
		NewTextCount:       2,
		FlickerScore:       0.25,
		MotionScore:        0.001,
	}
	passed, reasons = qc.EvaluateClip(bad, cfg)
	if passed || len(reasons) != 4 {
		t.Fatalf("expected bad scores to fail all 4 checks, got passed=%v reasons=%v", passed, reasons)
	}
}

func TestNextFallbackRoute(t *testing.T) {
	if r := qc.NextFallbackRoute(model.RouteI2V); r != model.RouteParallax {
		t.Errorf("expected parallax fallback from i2v, got %s", r)
	}
	if r := qc.NextFallbackRoute(model.RouteParallax); r != model.RouteKenBurns {
		t.Errorf("expected kenburns fallback from parallax, got %s", r)
	}
	if r := qc.NextFallbackRoute(model.RouteKenBurns); r != model.RouteKenBurns {
		t.Errorf("expected kenburns to stay kenburns, got %s", r)
	}
}
