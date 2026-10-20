package qc

import (
	"fmt"

	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
)

// EvaluateClip validates a candidate clip's QCScores against the configured QC thresholds.
// Returns (passed, failureReasons).
func EvaluateClip(scores model.QCScores, cfg config.QCConfig) (bool, []string) {
	var reasons []string

	idThresh := cfg.IdentityThreshold
	if idThresh <= 0 {
		idThresh = 0.85
	}
	flickerThresh := cfg.FlickerThreshold
	if flickerThresh <= 0 {
		flickerThresh = 0.15
	}

	if scores.IdentitySimilarity < idThresh {
		reasons = append(reasons, fmt.Sprintf("identity_similarity %.3f < threshold %.3f", scores.IdentitySimilarity, idThresh))
	}
	if scores.FlickerScore > flickerThresh {
		reasons = append(reasons, fmt.Sprintf("flicker_score %.3f > threshold %.3f", scores.FlickerScore, flickerThresh))
	}
	if scores.NewTextCount < 0 {
		reasons = append(reasons, "new_text_count not measured")
	} else if scores.NewTextCount > cfg.MaxNewTextLines {
		reasons = append(reasons, fmt.Sprintf("new_text_count %d > max %d", scores.NewTextCount, cfg.MaxNewTextLines))
	}

	minMotion := cfg.MinMotion
	if minMotion <= 0 {
		minMotion = 0.01
	}
	maxMotion := cfg.MaxMotion
	if maxMotion <= 0 {
		maxMotion = 0.98
	}
	if scores.MotionScore < minMotion || scores.MotionScore > maxMotion {
		reasons = append(reasons, fmt.Sprintf("motion_score %.3f outside sanity range [%.2f, %.2f]", scores.MotionScore, minMotion, maxMotion))
	}

	return len(reasons) == 0, reasons
}

// NextFallbackRoute returns the next cheaper/safer animation lane when QC or generation fails.
func NextFallbackRoute(current model.AnimationRoute) model.AnimationRoute {
	switch current {
	case model.RouteI2V:
		return model.RouteParallax
	case model.RouteParallax:
		return model.RouteKenBurns
	default:
		return model.RouteKenBurns
	}
}
