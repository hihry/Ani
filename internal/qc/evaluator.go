package qc

import (
	"fmt"
	"strings"

	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
)

// EvaluateClip validates a candidate clip's QCScores against the configured QC thresholds
// (including route-specific overrides in cfg.PerRoute when a route is provided).
// Returns (passed, failureReasons).
func EvaluateClip(scores model.QCScores, cfg config.QCConfig, route ...model.AnimationRoute) (bool, []string) {
	var reasons []string

	idThresh := cfg.IdentityThreshold
	if idThresh <= 0 {
		idThresh = 0.85
	}
	flickerThresh := cfg.FlickerThreshold
	if flickerThresh <= 0 {
		flickerThresh = 0.15
	}
	minMotion := cfg.MinMotion
	if minMotion <= 0 {
		minMotion = 0.01
	}
	maxMotion := cfg.MaxMotion
	if maxMotion <= 0 {
		maxMotion = 0.98
	}
	maxWarp := cfg.MaxWarpError
	minFirst := cfg.MinFirstFrameFidelity
	maxStyle := cfg.MaxStyleDrift

	if len(route) > 0 && route[0] != "" && cfg.PerRoute != nil {
		if rt, ok := cfg.PerRoute[string(route[0])]; ok {
			if rt.MinIdentitySimilarity > 0 {
				idThresh = rt.MinIdentitySimilarity
			}
			if rt.MaxFlickerScore > 0 {
				flickerThresh = rt.MaxFlickerScore
			}
			if rt.MinMotionScore > 0 {
				minMotion = rt.MinMotionScore
			}
			if rt.MaxMotionScore > 0 {
				maxMotion = rt.MaxMotionScore
			}
			if rt.MaxWarpError > 0 {
				maxWarp = rt.MaxWarpError
			}
			if rt.MinFirstFrameFidelity > 0 {
				minFirst = rt.MinFirstFrameFidelity
			}
			if rt.MaxStyleDrift > 0 {
				maxStyle = rt.MaxStyleDrift
			}
		}
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
	if scores.MotionScore < minMotion || scores.MotionScore > maxMotion {
		reasons = append(reasons, fmt.Sprintf("motion_score %.3f outside sanity range [%.2f, %.2f]", scores.MotionScore, minMotion, maxMotion))
	}

	if scores.MetricsVersion >= 2 {
		if maxWarp <= 0 {
			maxWarp = 0.12
		}
		if scores.WarpError > maxWarp {
			reasons = append(reasons, fmt.Sprintf("warp_error %.3f > max %.3f", scores.WarpError, maxWarp))
		}
		if minFirst > 0 && scores.FirstFrameFidelity > 0 && scores.FirstFrameFidelity < minFirst {
			reasons = append(reasons, fmt.Sprintf("first_frame_fidelity %.3f < min %.3f", scores.FirstFrameFidelity, minFirst))
		}
		if maxStyle > 0 && scores.StyleDrift > maxStyle {
			reasons = append(reasons, fmt.Sprintf("style_drift %.3f > max %.3f", scores.StyleDrift, maxStyle))
		}
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

// IsNonRetryableI2VError returns true if a render error is an OOM, missing node, or unavailable ComfyUI
// where re-running the exact same model with a new seed cannot succeed.
func IsNonRetryableI2VError(renderErr string) bool {
	lower := strings.ToLower(renderErr)
	return strings.Contains(lower, "out of memory") ||
		strings.Contains(lower, "oom") ||
		strings.Contains(lower, "missing") ||
		strings.Contains(lower, "node_error") ||
		strings.Contains(lower, "tensor size mismatch") ||
		strings.Contains(lower, "unreachable") ||
		strings.Contains(lower, "unavailable")
}

// NextModelOrRouteFallback implements the Phase 3 model-level + route-level fallback ladder:
//  1. Primary I2V model take 1 -> re-seed (up to maxI2VTakes) unless error is OOM / node_error
//  2. Next I2V model in fallbackModels (excluding non-I2V tokens like "parallax"/"kenburns")
//  3. RouteParallax (model="parallax")
//  4. RouteKenBurns (model="kenburns", terminal)
func NextModelOrRouteFallback(
	currentRoute model.AnimationRoute,
	currentModel string,
	modelTake int,
	maxI2VTakes int,
	fallbackModels []string,
	renderErr string,
) (nextRoute model.AnimationRoute, nextModel string, nextModelTake int, reason string, terminal bool) {
	if maxI2VTakes <= 0 {
		maxI2VTakes = 1
	}

	if currentRoute == model.RouteI2V {
		// 1. Try re-seeding the current I2V model if allowed and error is not OOM/node_error
		if modelTake < maxI2VTakes && !IsNonRetryableI2VError(renderErr) {
			return model.RouteI2V, currentModel, modelTake + 1, "qc_reseed", false
		}

		// 2. Try next I2V model in fallbackModels
		i2vFallbacks := make([]string, 0, len(fallbackModels))
		for _, m := range fallbackModels {
			mLower := strings.ToLower(strings.TrimSpace(m))
			if mLower != "" && mLower != "parallax" && mLower != "kenburns" && mLower != "hunyuan15" && mLower != "hunyuan15-fp8" {
				i2vFallbacks = append(i2vFallbacks, m)
			}
		}
		for idx, m := range i2vFallbacks {
			if m == currentModel {
				if idx+1 < len(i2vFallbacks) {
					r := "qc_model_fallback"
					if IsNonRetryableI2VError(renderErr) {
						r = "oom_or_node_model_fallback"
					}
					return model.RouteI2V, i2vFallbacks[idx+1], 1, r, false
				}
				break
			}
			// If currentModel wasn't in i2vFallbacks (it was the primary model), pick i2vFallbacks[0]
			if idx == 0 && currentModel != m {
				foundLater := false
				for _, fm := range i2vFallbacks {
					if fm == currentModel {
						foundLater = true
						break
					}
				}
				if !foundLater {
					r := "qc_model_fallback"
					if IsNonRetryableI2VError(renderErr) {
						r = "oom_or_node_model_fallback"
					}
					return model.RouteI2V, m, 1, r, false
				}
			}
		}

		// 3. Exhausted I2V models -> fall back to Parallax
		r := "i2v_exhausted_to_parallax"
		if IsNonRetryableI2VError(renderErr) {
			r = "i2v_error_to_parallax"
		}
		return model.RouteParallax, "parallax", 1, r, false
	}

	if currentRoute == model.RouteParallax {
		return model.RouteKenBurns, "kenburns", 1, "parallax_to_kenburns", false
	}

	return model.RouteKenBurns, "kenburns", modelTake, "terminal_kenburns", true
}
