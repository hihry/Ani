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

func TestEvaluateClipPerRouteAndV2Metrics(t *testing.T) {
	cfg := config.QCConfig{
		IdentityThreshold: 0.85,
		FlickerThreshold:  0.15,
		MaxNewTextLines:   0,
		PerRoute: map[string]config.QCRouteThresholds{
			"i2v": {
				MinIdentitySimilarity: 0.75,
				MaxFlickerScore:       0.14,
				MinMotionScore:        0.01,
				MaxMotionScore:        0.98,
				MaxWarpError:          0.11,
				MinFirstFrameFidelity: 0.78,
				MaxStyleDrift:         0.25,
			},
			"parallax": {
				MinIdentitySimilarity: 0.85,
				MaxFlickerScore:       0.15,
				MinMotionScore:        0.01,
				MaxMotionScore:        0.98,
				MaxWarpError:          0.10,
				MinFirstFrameFidelity: 0.82,
				MaxStyleDrift:         0.20,
			},
		},
	}

	scores := model.QCScores{
		IdentitySimilarity: 0.79,
		NewTextCount:       0,
		FlickerScore:       0.06,
		MotionScore:        0.18,
		WarpError:          0.05,
		FirstFrameFidelity: 0.84,
		StyleDrift:         0.12,
		MetricsVersion:     2,
	}

	// Should pass for i2v (min identity 0.75), but fail for parallax (min identity 0.85)
	passI2V, reasonsI2V := qc.EvaluateClip(scores, cfg, model.RouteI2V)
	if !passI2V {
		t.Fatalf("expected scores to pass I2V route thresholds, got reasons=%v", reasonsI2V)
	}
	passPx, _ := qc.EvaluateClip(scores, cfg, model.RouteParallax)
	if passPx {
		t.Fatalf("expected scores with identity 0.79 to fail parallax threshold 0.85")
	}

	// High warp error should fail v2 metrics check
	scores.WarpError = 0.22
	passWarp, reasonsWarp := qc.EvaluateClip(scores, cfg, model.RouteI2V)
	if passWarp || len(reasonsWarp) == 0 {
		t.Fatalf("expected high warp_error=0.22 to fail v2 QC check")
	}
}

func TestFallbackModelLadder(t *testing.T) {
	fallbacks := []string{"ltx23-fp8", "parallax", "kenburns"}

	// Step 1: Primary wan22-ti2v-5b take 1 fails QC -> re-seed (take 2)
	r1, m1, t1, reason1, term1 := qc.NextModelOrRouteFallback(model.RouteI2V, "wan22-ti2v-5b", 1, 2, fallbacks, "")
	if r1 != model.RouteI2V || m1 != "wan22-ti2v-5b" || t1 != 2 || reason1 != "qc_reseed" || term1 {
		t.Fatalf("unexpected step 1 fallback: route=%s model=%s take=%d reason=%s term=%v", r1, m1, t1, reason1, term1)
	}

	// Step 2: Primary take 2 fails QC -> next I2V model ltx23-fp8 (take 1)
	r2, m2, t2, reason2, term2 := qc.NextModelOrRouteFallback(r1, m1, t1, 2, fallbacks, "")
	if r2 != model.RouteI2V || m2 != "ltx23-fp8" || t2 != 1 || reason2 != "qc_model_fallback" || term2 {
		t.Fatalf("unexpected step 2 fallback: route=%s model=%s take=%d reason=%s term=%v", r2, m2, t2, reason2, term2)
	}

	// Step 3: OOM on ltx23-fp8 take 1 -> skips reseed and drops to parallax!
	r3, m3, t3, _, term3 := qc.NextModelOrRouteFallback(r2, m2, t2, 2, fallbacks, "CUDA error: out of memory")
	if r3 != model.RouteParallax || m3 != "parallax" || t3 != 1 || term3 {
		t.Fatalf("expected OOM on last I2V model to drop to parallax, got route=%s model=%s take=%d", r3, m3, t3)
	}

	// Step 4: Parallax fails -> drops to kenburns
	r4, m4, _, _, term4 := qc.NextModelOrRouteFallback(r3, m3, t3, 2, fallbacks, "")
	if r4 != model.RouteKenBurns || m4 != "kenburns" || term4 {
		t.Fatalf("expected parallax to drop to kenburns, got route=%s model=%s", r4, m4)
	}

	// Step 5: Kenburns is terminal
	_, _, _, _, term5 := qc.NextModelOrRouteFallback(r4, m4, 1, 2, fallbacks, "")
	if !term5 {
		t.Fatalf("expected kenburns to be terminal")
	}
}
