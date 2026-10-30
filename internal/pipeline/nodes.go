package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/m2a/manhwa2anime/internal/client"
	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
	"github.com/m2a/manhwa2anime/internal/qc"
	"gopkg.in/yaml.v3"
)

var (
	workerClientsMu sync.Mutex
	workerClients   = make(map[string]*client.WorkerClient)
)

// getWorkerClient returns a shared WorkerClient instance for the configured worker address (M0-3).
func getWorkerClient(cfg *config.PipelineConfig) *client.WorkerClient {
	host := "127.0.0.1"
	port := 8100
	if cfg != nil {
		if cfg.Worker.Host != "" {
			host = cfg.Worker.Host
		}
		if cfg.Worker.Port > 0 {
			port = cfg.Worker.Port
		}
	}
	key := fmt.Sprintf("%s:%d|%s|%s", host, port, os.Getenv("M2A_WORKER_CLI"), os.Getenv("M2A_PYTHON"))
	workerClientsMu.Lock()
	defer workerClientsMu.Unlock()
	if wc, ok := workerClients[key]; ok {
		return wc
	}
	wc := client.NewWorkerClient(cfg)
	workerClients[key] = wc
	return wc
}

// PanelRoutingDecision captures the Phase 3 routing decision and cost estimate for a single panel.
type PanelRoutingDecision struct {
	PanelID        string               `json:"panel_id"`
	Route          model.AnimationRoute `json:"route"`
	ModelID        string               `json:"model_id,omitempty"`
	HeroScore      float64              `json:"hero_score"`
	EstGPUSeconds  float64              `json:"est_gpu_s"`
	DurationS      float64              `json:"duration_s"`
	MotionPrompt   string               `json:"motion_prompt,omitempty"`
	FallbackReason string               `json:"fallback_reason,omitempty"`
}

// RoutingPlan is persisted to routes/routing_plan.json during RouteNode and updated during TTSNode.
type RoutingPlan struct {
	BudgetSeconds   float64                `json:"budget_s"`
	EstimatedTotalS float64                `json:"estimated_total_s"`
	MaxI2VPanels    int                    `json:"max_i2v_panels"`
	SPerFrame       float64                `json:"s_per_frame"`
	Panels          []PanelRoutingDecision `json:"panels"`
}

// HeroScore computes a [0, 1] priority score for routing a panel to the GPU I2V lane (Phase 3 §5.6).
func HeroScore(p model.PanelJSON, idx int, total int, maskRatio float64) float64 {
	score := 0.0
	if p.MotionIntensity >= 0.70 {
		score += 0.35
	} else if p.MotionIntensity >= 0.40 {
		score += 0.18
	} else {
		score += 0.05
	}

	if len(p.Dialogue) > 0 {
		score += 0.15
	}

	hasEmotion := p.Shot == model.ShotCloseUp || p.Shot == model.ShotSplash
	for _, d := range p.Dialogue {
		if d.Emotion != "" && d.Emotion != "neutral" {
			hasEmotion = true
		}
	}
	for _, c := range p.Characters {
		if c.Emotion != "" && c.Emotion != "neutral" {
			hasEmotion = true
		}
	}
	if hasEmotion {
		score += 0.15
	}

	areaNorm := math.Min(1.0, float64(p.BBox.W*p.BBox.H)/(720.0*900.0))
	if areaNorm <= 0 {
		areaNorm = 0.5
	}
	score += 0.15 * areaNorm

	if maskRatio < 0.25 {
		score += 0.10
	}

	if idx == 0 || (total > 1 && idx == total-1) {
		score += 0.10
	}

	return math.Round(math.Min(1.0, score)*1000) / 1000
}

func loadCalibrationSPerFrame(cfg *config.PipelineConfig, modelID string) float64 {
	defaultSPF := 1.2
	if cfg == nil || cfg.Generation.CalibrationFile == "" {
		return defaultSPF
	}
	raw, err := os.ReadFile(cfg.Generation.CalibrationFile)
	if err != nil {
		return defaultSPF
	}
	var doc struct {
		SPerFrame float64            `yaml:"s_per_frame"`
		Models    map[string]struct {
			SPerFrame float64 `yaml:"s_per_frame"`
		} `yaml:"models"`
	}
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		return defaultSPF
	}
	if m, ok := doc.Models[modelID]; ok && m.SPerFrame > 0 {
		return m.SPerFrame
	}
	if doc.SPerFrame > 0 {
		return doc.SPerFrame
	}
	return defaultSPF
}

func estimateNativeFrames(durationS float64, maxClipS float64) int {
	effDur := durationS
	if maxClipS > 0 && effDur > maxClipS {
		effDur = maxClipS
	}
	raw := int(math.Round(effDur * 16.0))
	if raw < 5 {
		raw = 5
	}
	k := int(math.Ceil(float64(raw-1) / 4.0))
	if k < 1 {
		k = 1
	}
	snapped := k*4 + 1
	if snapped > 81 {
		snapped = 81
	}
	return snapped
}

func renderParams(cfg *config.PipelineConfig) (fps, w, h int) {
	fps, w, h = 24, 720, 1280
	if cfg != nil {
		if cfg.Composition.FPS > 0 {
			fps = cfg.Composition.FPS
		}
		if r := cfg.Composition.Resolution; r[0] > 0 && r[1] > 0 {
			w, h = r[0], r[1]
		}
	}
	return
}

func effectiveQCConfig(cfg *config.PipelineConfig) config.QCConfig {
	qcCfg := config.QCConfig{
		IdentityThreshold: 0.85,
		FlickerThreshold:  0.15,
		MaxNewTextLines:   0,
		MinMotion:         0.01,
		MaxMotion:         0.98,
	}
	if cfg != nil && (cfg.QC.IdentityThreshold > 0 || len(cfg.QC.PerRoute) > 0) {
		qcCfg = cfg.QC
		if qcCfg.IdentityThreshold <= 0 {
			qcCfg.IdentityThreshold = 0.85
		}
		if qcCfg.FlickerThreshold <= 0 {
			qcCfg.FlickerThreshold = 0.15
		}
		if qcCfg.MinMotion <= 0 {
			qcCfg.MinMotion = 0.01
		}
		if qcCfg.MaxMotion <= 0 {
			qcCfg.MaxMotion = 0.98
		}
	}
	return qcCfg
}

func maxPanelWorkers() int {
	n := runtime.NumCPU() / 2
	if n < 1 {
		return 1
	}
	if n > 4 {
		return 4
	}
	return n
}

func bestCandidateIndex(candidates []model.ClipCandidate) int {
	bestIdx := 0
	bestScore := -1.0
	for i, c := range candidates {
		if c.QC.IdentitySimilarity > bestScore {
			bestScore = c.QC.IdentitySimilarity
			bestIdx = i
		}
	}
	return bestIdx
}

// IngestNode handles loading, copying, validating and stitching input source pages.
func IngestNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info(fmt.Sprintf("Ingesting pages: %v", state.Config.SourcePages))

	if len(state.Config.SourcePages) == 0 {
		return errors.New("ingest: no source pages specified")
	}
	for _, p := range state.Config.SourcePages {
		if _, err := os.Stat(p); err != nil {
			return fmt.Errorf("ingest: source page %q not found: %w", p, err)
		}
	}

	runDir := resolveRunDir(state, cfg)
	pagesDir := filepath.Join(runDir, "pages")
	if err := os.MkdirAll(pagesDir, 0755); err != nil {
		return fmt.Errorf("create pages dir %q: %w", pagesDir, err)
	}

	for idx, src := range state.Config.SourcePages {
		ext := filepath.Ext(src)
		if ext == "" {
			ext = ".png"
		}
		dst := filepath.Join(pagesDir, fmt.Sprintf("page_%03d%s", idx+1, ext))
		if err := copyFile(src, dst); err != nil {
			return fmt.Errorf("ingest: copy %q to %q: %w", src, dst, err)
		}
	}

	stitchedPath := filepath.Join(runDir, "stitched.png")
	wc := getWorkerClient(cfg)
	if _, err := wc.Ingest(ctx, state.Config.SourcePages, stitchedPath); err != nil {
		return fmt.Errorf("ingest stitch pages: %w", err)
	}
	slog.Info("Successfully stitched pages", "stitched", stitchedPath)

	return nil
}

// SegmentNode detects panel bounding boxes from source pages.
func SegmentNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Segmenting panels...")

	runDir := resolveRunDir(state, cfg)
	panelsDir := filepath.Join(runDir, "panels")
	if err := os.MkdirAll(panelsDir, 0755); err != nil {
		return fmt.Errorf("create panels dir %q: %w", panelsDir, err)
	}

	stitchedPath := filepath.Join(runDir, "stitched.png")
	if !fileExists(stitchedPath) {
		return fmt.Errorf("segment: %q missing (INGEST did not produce a stitched strip)", stitchedPath)
	}

	detector := "opencv"
	if cfg != nil && cfg.Perception.Detector != "" {
		detector = cfg.Perception.Detector
	}

	wc := getWorkerClient(cfg)
	res, err := wc.Segment(ctx, stitchedPath, panelsDir, detector)
	if err != nil {
		return fmt.Errorf("segment panels: %w", err)
	}
	if len(res.Panels) == 0 {
		return errors.New("segment: detector returned 0 panels")
	}

	panels := make([]model.PanelJSON, 0, len(res.Panels))
	for idx, dp := range res.Panels {
		p := model.NewPanelJSON(dp.PanelID, dp.BBox)
		switch model.ShotType(dp.ShotType) {
		case model.ShotCloseUp, model.ShotMedium, model.ShotWide, model.ShotSplash:
			p.Shot = model.ShotType(dp.ShotType)
		}
		if idx%2 == 0 {
			p.Route = model.RouteKenBurns
		} else {
			p.Route = model.RouteParallax
		}
		p.DurationS = 2.5
		panels = append(panels, p)
	}
	state.Panels = panels
	slog.Info(fmt.Sprintf("Segmented %d panels from source strip", len(state.Panels)))
	return nil
}

// UnderstandNode runs vision-language models / semantic analyzers to extract characters, dialogue, and semantics.
func UnderstandNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Running VLM understanding...")

	runDir := resolveRunDir(state, cfg)
	vlmDir := filepath.Join(runDir, "understanding")
	if err := os.MkdirAll(vlmDir, 0755); err != nil {
		return fmt.Errorf("create understanding dir %q: %w", vlmDir, err)
	}
	cropsDir := filepath.Join(vlmDir, "crops")
	if err := os.MkdirAll(cropsDir, 0755); err != nil {
		return fmt.Errorf("create crops dir %q: %w", cropsDir, err)
	}

	if state.CharacterBible == nil {
		state.CharacterBible = model.NewCharacterBible()
	}

	wc := getWorkerClient(cfg)
	type understandItem struct {
		res      *client.UnderstandResult
		cropPath string
		err      error
	}
	results := make([]understandItem, len(state.Panels))

	sem := make(chan struct{}, maxPanelWorkers())
	var wg sync.WaitGroup
	for i := range state.Panels {
		i := i
		p := state.Panels[i]
		cropPath := filepath.Join(runDir, "panels", p.PanelID+".png")
		if !fileExists(cropPath) {
			results[i] = understandItem{err: fmt.Errorf("understand: panel image %q not found", cropPath)}
			continue
		}
		annotatedPath := filepath.Join(vlmDir, p.PanelID+"_annotated.png")
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			res, err := wc.Understand(ctx, cropPath, p.PanelID, state.Config.SourceLanguage, i+1, cropsDir, annotatedPath)
			results[i] = understandItem{res: res, cropPath: cropPath, err: err}
		}()
	}
	wg.Wait()

	for i := range state.Panels {
		item := results[i]
		if item.err != nil {
			return fmt.Errorf("understand panel %s: %w", state.Panels[i].PanelID, item.err)
		}
		p := &state.Panels[i]
		res := item.res
		if res != nil {
			if res.Shot != "" {
				p.Shot = res.Shot
			}
			if res.Action != "" {
				p.Action = res.Action
			}
			if res.MotionPrompt != "" {
				p.MotionPrompt = res.MotionPrompt
			}
			if res.MotionIntensity > 0 {
				p.MotionIntensity = res.MotionIntensity
			}
			p.Dialogue = res.Dialogue
			p.SFX = res.SFX
			p.Characters = res.Characters

			for _, c := range p.Characters {
				refCrop := filepath.ToSlash(item.cropPath)
				if c.CropPath != "" {
					refCrop = filepath.ToSlash(c.CropPath)
				}
				entry, exists := state.CharacterBible.Characters[c.ID]
				if !exists {
					state.CharacterBible.Characters[c.ID] = model.CharacterEntry{
						Name:           c.Name,
						Description:    fmt.Sprintf("Character %s (%s)", c.Name, c.Pose),
						ReferenceCrops: []string{refCrop},
						Voice: model.VoiceProfile{
							VoiceID: "voice_" + c.ID,
							Style:   c.Emotion,
						},
						Appearances: []string{p.PanelID},
					}
				} else {
					if !containsStr(entry.Appearances, p.PanelID) {
						entry.Appearances = append(entry.Appearances, p.PanelID)
					}
					if !containsStr(entry.ReferenceCrops, refCrop) {
						entry.ReferenceCrops = append(entry.ReferenceCrops, refCrop)
					}
					state.CharacterBible.Characters[c.ID] = entry
				}
			}
		}

		if panelBytes, err := json.MarshalIndent(p, "", "  "); err == nil {
			_ = os.WriteFile(filepath.Join(vlmDir, p.PanelID+".json"), panelBytes, 0644)
		}
	}

	if bibleBytes, err := json.MarshalIndent(state.CharacterBible, "", "  "); err == nil {
		_ = os.WriteFile(filepath.Join(vlmDir, "character_bible.json"), bibleBytes, 0644)
	}

	return nil
}

// CleanPlatesNode creates speech-bubble and text-free clean plate images.
func CleanPlatesNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Creating clean plates...")

	runDir := resolveRunDir(state, cfg)
	cleanPlatesDir := filepath.Join(runDir, "clean_plates")
	if err := os.MkdirAll(cleanPlatesDir, 0755); err != nil {
		return fmt.Errorf("create clean_plates dir %q: %w", cleanPlatesDir, err)
	}

	if state.CleanPlatePaths == nil {
		state.CleanPlatePaths = make(map[string]string)
	}

	wc := getWorkerClient(cfg)
	errs := make([]error, len(state.Panels))
	cleanPaths := make([]string, len(state.Panels))

	sem := make(chan struct{}, maxPanelWorkers())
	var wg sync.WaitGroup
	for i, p := range state.Panels {
		i, p := i, p
		cleanPath := filepath.Join(cleanPlatesDir, p.PanelID+"_clean.png")
		cleanPaths[i] = cleanPath
		cropPath := filepath.Join(runDir, "panels", p.PanelID+".png")
		if !fileExists(cropPath) {
			errs[i] = fmt.Errorf("clean_plates: panel image %q not found", cropPath)
			continue
		}

		var bboxes []model.BBox
		for _, d := range p.Dialogue {
			if d.BBox != nil {
				bboxes = append(bboxes, *d.BBox)
			}
		}
		for _, s := range p.SFX {
			if s.BBox != nil {
				bboxes = append(bboxes, *s.BBox)
			}
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			_, err := wc.Inpaint(ctx, cropPath, cleanPath, bboxes)
			errs[i] = err
		}()
	}
	wg.Wait()

	for i, p := range state.Panels {
		if errs[i] != nil {
			return fmt.Errorf("inpaint panel %s: %w", p.PanelID, errs[i])
		}
		state.CleanPlatePaths[p.PanelID] = filepath.ToSlash(cleanPaths[i])
	}

	return nil
}

// RouteNode assigns animation routes and camera motion prompts to each panel using
// HeroScore ranking, calibrated GPU s_per_frame estimation, and I2V budget enforcement.
func RouteNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Routing panels...")

	runDir := resolveRunDir(state, cfg)
	routesDir := filepath.Join(runDir, "routes")
	if err := os.MkdirAll(routesDir, 0755); err != nil {
		return fmt.Errorf("create routes dir %q: %w", routesDir, err)
	}

	primaryModel := "wan22-ti2v-5b"
	maxI2VPanels := 3
	maxClipS := 5.0
	budgetS := 180.0
	if cfg != nil {
		if m := cfg.Generation.EffectivePrimaryModel(); m != "" {
			primaryModel = m
		}
		if cfg.Generation.MaxI2VPanels > 0 {
			maxI2VPanels = cfg.Generation.MaxI2VPanels
		}
		if cfg.Generation.MaxI2VClipS > 0 {
			maxClipS = cfg.Generation.MaxI2VClipS
		} else if cfg.Generation.MaxClipDuration > 0 {
			maxClipS = cfg.Generation.MaxClipDuration
		}
		if cfg.Cost.I2VBudgetS > 0 {
			budgetS = cfg.Cost.I2VBudgetS
		}
	}

	allowI2V := cfg != nil &&
		primaryModel != "kenburns" &&
		primaryModel != "parallax" &&
		state.Config.ComputeTier != "t0"

	comfyFallbackReason := ""
	selectedI2VModel := primaryModel

	// Check /health if worker daemon is reachable to see if I2V provider is explicitly unavailable
	if allowI2V {
		wc := getWorkerClient(cfg)
		if health, err := wc.Health(ctx); err == nil && health != nil && len(health.VideoProviders) > 0 {
			st, exists := health.VideoProviders[primaryModel]
			if exists && st != "ready" {
				// Check if any configured fallback I2V model is ready
				foundReadyFallback := ""
				if cfg != nil {
					if fb := cfg.Generation.VideoFallback; fb != "" && health.VideoProviders[fb] == "ready" {
						foundReadyFallback = fb
					}
					for _, fm := range cfg.Generation.FallbackModels {
						if health.VideoProviders[fm] == "ready" {
							foundReadyFallback = fm
							break
						}
					}
				}
				if foundReadyFallback != "" {
					selectedI2VModel = foundReadyFallback
				} else {
					allowI2V = false
					comfyFallbackReason = "comfy_unavailable"
				}
			}
		}
	}

	sPerFrame := loadCalibrationSPerFrame(cfg, selectedI2VModel)
	nPanels := len(state.Panels)

	type scoredPanel struct {
		idx       int
		heroScore float64
		estGPUS   float64
		eligible  bool
	}
	scored := make([]scoredPanel, nPanels)

	for i := range state.Panels {
		p := &state.Panels[i]
		// Estimate bubble mask ratio from dialogue/SFX bbox coverage
		panelArea := math.Max(1.0, float64(p.BBox.W*p.BBox.H))
		bubbleArea := 0.0
		for _, d := range p.Dialogue {
			if d.BBox != nil {
				bubbleArea += float64(d.BBox.W * d.BBox.H)
			}
		}
		maskRatio := math.Min(1.0, bubbleArea/panelArea)
		hScore := HeroScore(*p, i, nPanels, maskRatio)
		estFrames := estimateNativeFrames(p.DurationS, maxClipS)
		estGPUS := math.Round(float64(estFrames)*sPerFrame*100) / 100

		isHeroEligible := (p.Shot == model.ShotCloseUp || p.Shot == model.ShotSplash || p.MotionIntensity >= 0.55) && hScore >= 0.45
		scored[i] = scoredPanel{
			idx:       i,
			heroScore: hScore,
			estGPUS:   estGPUS,
			eligible:  isHeroEligible,
		}

		// Default non-I2V assignment
		if (p.Shot == model.ShotMedium || p.Shot == model.ShotCloseUp || p.Shot == model.ShotSplash) && (p.MotionIntensity >= 0.35 || i%2 == 1) {
			p.Route = model.RouteParallax
		} else {
			p.Route = model.RouteKenBurns
		}

		if p.MotionPrompt == "" {
			switch p.Shot {
			case model.ShotCloseUp:
				p.MotionPrompt = "dramatic slow push-in on character expression, subtle hair sway"
			case model.ShotSplash:
				p.MotionPrompt = "dynamic impact camera pullback, intense energy motion"
			case model.ShotWide:
				p.MotionPrompt = "slow establishing horizon pan, gentle atmospheric drift"
			default:
				p.MotionPrompt = "steady cinematic vertical track, layered depth parallax"
			}
		}
	}

	// Greedy knapsack over eligible hero panels sorted by heroScore descending
	orderIdx := make([]int, nPanels)
	for i := range orderIdx {
		orderIdx[i] = i
	}
	sort.SliceStable(orderIdx, func(a, b int) bool {
		return scored[orderIdx[a]].heroScore > scored[orderIdx[b]].heroScore
	})

	usedBudgetS := 0.0
	i2vCount := 0
	fallbackReasons := make([]string, nPanels)
	for _, panelIdx := range orderIdx {
		sp := scored[panelIdx]
		if !sp.eligible {
			continue
		}
		if !allowI2V {
			if comfyFallbackReason != "" {
				fallbackReasons[panelIdx] = comfyFallbackReason
				state.Panels[panelIdx].Route = model.RouteParallax
			}
			continue
		}
		if i2vCount < maxI2VPanels && (usedBudgetS+sp.estGPUS) <= budgetS {
			state.Panels[panelIdx].Route = model.RouteI2V
			usedBudgetS = math.Round((usedBudgetS+sp.estGPUS)*100) / 100
			i2vCount++
		} else {
			state.Panels[panelIdx].Route = model.RouteParallax
			if i2vCount >= maxI2VPanels {
				fallbackReasons[panelIdx] = "max_i2v_panels_exceeded"
			} else {
				fallbackReasons[panelIdx] = "i2v_budget_exceeded"
			}
		}
	}

	decisions := make([]PanelRoutingDecision, nPanels)
	for i, p := range state.Panels {
		mID := string(p.Route)
		if p.Route == model.RouteI2V {
			mID = selectedI2VModel
		}
		decisions[i] = PanelRoutingDecision{
			PanelID:        p.PanelID,
			Route:          p.Route,
			ModelID:        mID,
			HeroScore:      scored[i].heroScore,
			EstGPUSeconds:  scored[i].estGPUS,
			DurationS:      p.DurationS,
			MotionPrompt:   p.MotionPrompt,
			FallbackReason: fallbackReasons[i],
		}
	}

	plan := RoutingPlan{
		BudgetSeconds:   budgetS,
		EstimatedTotalS: usedBudgetS,
		MaxI2VPanels:    maxI2VPanels,
		SPerFrame:       sPerFrame,
		Panels:          decisions,
	}
	if planBytes, err := json.MarshalIndent(plan, "", "  "); err == nil {
		_ = os.WriteFile(filepath.Join(routesDir, "routing_plan.json"), planBytes, 0644)
	}

	return nil
}

// GenerateNode handles animation generation for all panels based on their assigned routes.
// Runs a serial GPU lane for RouteI2V (sorted by (model, prompt) to maximize ComfyUI cache hits)
// concurrently with a multi-worker CPU lane for RouteParallax and RouteKenBurns.
func GenerateNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Generating clips...")

	runDir := resolveRunDir(state, cfg)
	clipsDir := filepath.Join(runDir, "clips")
	if err := os.MkdirAll(clipsDir, 0755); err != nil {
		return fmt.Errorf("create clips dir %q: %w", clipsDir, err)
	}

	if state.ClipCandidates == nil {
		state.ClipCandidates = make(map[string][]model.ClipCandidate)
	}

	wc := getWorkerClient(cfg)
	fps, resW, resH := renderParams(cfg)
	qcCfg := effectiveQCConfig(cfg)

	primaryModel := "wan22-ti2v-5b"
	comfyURL := ""
	if cfg != nil {
		if m := cfg.Generation.EffectivePrimaryModel(); m != "" {
			primaryModel = m
		}
		comfyURL = cfg.Generation.ComfyURL
	}

	type genResult struct {
		cand      model.ClipCandidate
		routeUsed model.AnimationRoute
		reasons   []string
		err       error
	}
	results := make([]genResult, len(state.Panels))

	var gpuIndices []int
	var cpuIndices []int
	for i, p := range state.Panels {
		srcPath := state.CleanPlatePaths[p.PanelID]
		if srcPath == "" || !fileExists(srcPath) {
			srcPath = filepath.Join(runDir, "panels", p.PanelID+".png")
		}
		if !fileExists(srcPath) {
			return fmt.Errorf("generate %s: source image %q not found", p.PanelID, srcPath)
		}
		if p.Route == model.RouteI2V {
			gpuIndices = append(gpuIndices, i)
		} else {
			cpuIndices = append(cpuIndices, i)
		}
	}

	// Sort GPU indices by (model, motion_prompt, panel_id) to maximize ComfyUI prompt prefix cache hits
	sort.SliceStable(gpuIndices, func(a, b int) bool {
		pa := state.Panels[gpuIndices[a]]
		pb := state.Panels[gpuIndices[b]]
		if pa.MotionPrompt != pb.MotionPrompt {
			return pa.MotionPrompt < pb.MotionPrompt
		}
		return pa.PanelID < pb.PanelID
	})

	var stateMu sync.Mutex
	renderOne := func(i int) {
		p := state.Panels[i]
		srcPath := state.CleanPlatePaths[p.PanelID]
		if srcPath == "" || !fileExists(srcPath) {
			srcPath = filepath.Join(runDir, "panels", p.PanelID+".png")
		}
		clipPath := filepath.Join(clipsDir, p.PanelID+"_clip_01.avi")
		seed := model.ComputeDeterministicSeed(state.Config.ConfigHash, p.PanelID, 1)
		modelID := string(p.Route)
		if p.Route == model.RouteI2V {
			modelID = primaryModel
		}
		emotion := "neutral"
		if len(p.Dialogue) > 0 && p.Dialogue[0].Emotion != "" {
			emotion = p.Dialogue[0].Emotion
		}

		opt := client.RenderOptions{
			PanelID:      p.PanelID,
			ModelID:      modelID,
			Seed:         seed,
			ActionPrompt: p.Action,
			MotionType:   p.MotionPrompt,
			Emotion:      emotion,
			ComfyURL:     comfyURL,
		}

		t0 := time.Now()
		rendered, err := wc.RenderClip(ctx, srcPath, clipPath, p.Route, p.Shot, p.DurationS, fps, resW, resH, opt)
		if err != nil {
			// If RouteI2V failed (e.g. OOM, node error, or ComfyUI unavailable), record failed candidate
			// so QCNode can walk the model/route fallback ladder instead of aborting the entire run.
			if p.Route == model.RouteI2V {
				elapsedS := time.Since(t0).Seconds()
				cand := model.ClipCandidate{
					PanelID:     p.PanelID,
					Path:        filepath.ToSlash(clipPath),
					TakeNumber:  1,
					Seed:        seed,
					Route:       model.RouteI2V,
					ModelID:     modelID,
					GenSeconds:  elapsedS,
					RenderError: err.Error(),
					Accepted:    false,
				}
				stateMu.Lock()
				if elapsedS > 0 {
					state.RecordModelCost(modelID, elapsedS)
				}
				state.ClipCandidates[p.PanelID] = []model.ClipCandidate{cand}
				_ = state.Save(runDir)
				stateMu.Unlock()
				results[i] = genResult{
					cand:      cand,
					routeUsed: model.RouteI2V,
					reasons:   []string{"render_error: " + err.Error()},
				}
				return
			}
			results[i] = genResult{err: fmt.Errorf("generate %s (%s): %w", p.PanelID, p.Route, err)}
			return
		}

		routeUsed := p.Route
		if rendered.RouteUsed != "" {
			routeUsed = model.AnimationRoute(rendered.RouteUsed)
		}
		passed, reasons := qc.EvaluateClip(rendered.QCScores, qcCfg, routeUsed)
		rendered.QCScores.OverallPass = passed
		rendered.QCScores.Reasons = reasons

		effModel := rendered.ModelID
		if effModel == "" {
			effModel = modelID
		}
		effSeed := rendered.Seed
		if effSeed == 0 {
			effSeed = seed
		}

		cand := model.ClipCandidate{
			PanelID:        p.PanelID,
			Path:           filepath.ToSlash(rendered.ClipPath),
			RawClipPath:    filepath.ToSlash(rendered.RawClipPath),
			TakeNumber:     1,
			Seed:           effSeed,
			Route:          routeUsed,
			ModelID:        effModel,
			PositivePrompt: rendered.PositivePrompt,
			NegativePrompt: rendered.NegativePrompt,
			NativeWidth:    rendered.NativeWidth,
			NativeHeight:   rendered.NativeHeight,
			NativeFrames:   rendered.NativeFrames,
			NativeFPS:      rendered.NativeFPS,
			GenSeconds:     rendered.GenSeconds,
			PeakVRAMGB:     rendered.PeakVRAMGB,
			Cached:         rendered.Cached,
			QC:             rendered.QCScores,
			Accepted:       passed,
		}

		if p.Route == model.RouteI2V {
			stateMu.Lock()
			if rendered.GenSeconds > 0 {
				state.RecordModelCost(effModel, rendered.GenSeconds)
			}
			state.ClipCandidates[p.PanelID] = []model.ClipCandidate{cand}
			_ = state.Save(runDir)
			stateMu.Unlock()
		}

		results[i] = genResult{
			cand:      cand,
			routeUsed: routeUsed,
			reasons:   reasons,
		}
	}

	var wg sync.WaitGroup
	// 1. Launch serial GPU lane
	if len(gpuIndices) > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, idx := range gpuIndices {
				if ctx.Err() != nil {
					return
				}
				renderOne(idx)
			}
			_ = wc.FreeGPU(ctx)
		}()
	}

	// 2. Launch concurrent CPU lanes
	sem := make(chan struct{}, maxPanelWorkers())
	for _, idx := range cpuIndices {
		idx := idx
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			renderOne(idx)
		}()
	}
	wg.Wait()

	for i := range state.Panels {
		r := results[i]
		if r.err != nil {
			return r.err
		}
		p := &state.Panels[i]
		if r.routeUsed != "" {
			p.Route = r.routeUsed
		}
		state.ClipCandidates[p.PanelID] = []model.ClipCandidate{r.cand}
		if !r.cand.Accepted {
			slog.Warn("Panel clip did not pass initial QC thresholds", "panel", p.PanelID, "reasons", r.reasons)
		}
	}

	return nil
}

// QCNode verifies quality and consistency metrics on candidate video clips and walks the Phase 3
// model-level + route-level fallback ladder (`primary -> reseed -> fallback_models -> parallax -> kenburns`).
func QCNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Running quality checks...")

	runDir := resolveRunDir(state, cfg)
	qcDir := filepath.Join(runDir, "qc")
	if err := os.MkdirAll(qcDir, 0755); err != nil {
		return fmt.Errorf("create qc dir %q: %w", qcDir, err)
	}

	qcCfg := effectiveQCConfig(cfg)
	wc := getWorkerClient(cfg)
	fps, resW, resH := renderParams(cfg)

	maxRetries := 2
	maxI2VTakes := 2
	primaryModel := "wan22-ti2v-5b"
	var fallbackModels []string
	comfyURL := ""
	if cfg != nil {
		if cfg.Generation.MaxRetries > 0 {
			maxRetries = cfg.Generation.MaxRetries
		}
		if cfg.Generation.MaxI2VTakes > 0 {
			maxI2VTakes = cfg.Generation.MaxI2VTakes
		}
		if m := cfg.Generation.EffectivePrimaryModel(); m != "" {
			primaryModel = m
		}
		if cfg.Generation.VideoFallback != "" {
			fallbackModels = append(fallbackModels, cfg.Generation.VideoFallback)
		}
		for _, fm := range cfg.Generation.FallbackModels {
			if !containsStr(fallbackModels, fm) {
				fallbackModels = append(fallbackModels, fm)
			}
		}
		comfyURL = cfg.Generation.ComfyURL
	}

	for i := range state.Panels {
		p := &state.Panels[i]
		candidates := state.ClipCandidates[p.PanelID]
		hasAccepted := false

		for j := range candidates {
			cRoute := candidates[j].Route
			if cRoute == "" {
				cRoute = p.Route
				candidates[j].Route = cRoute
			}
			if candidates[j].RenderError != "" {
				candidates[j].QC.OverallPass = false
				candidates[j].Accepted = false
				continue
			}
			passed, reasons := qc.EvaluateClip(candidates[j].QC, qcCfg, cRoute)
			candidates[j].QC.OverallPass = passed
			candidates[j].QC.Reasons = reasons
			candidates[j].Accepted = passed
			if passed {
				hasAccepted = true
			}
		}
		state.ClipCandidates[p.PanelID] = candidates

		currRoute := p.Route
		currModel := string(p.Route)
		if currRoute == model.RouteI2V {
			currModel = primaryModel
		}
		currModelTake := 1
		lastRenderErr := ""
		if len(candidates) > 0 {
			last := candidates[len(candidates)-1]
			if last.Route != "" {
				currRoute = last.Route
			}
			if last.ModelID != "" {
				currModel = last.ModelID
			}
			lastRenderErr = last.RenderError
		}

		maxTotalAttempts := maxRetries + maxI2VTakes + len(fallbackModels) + 2
		for take := len(candidates) + 1; !hasAccepted && take <= maxTotalAttempts; take++ {
			nextRoute, nextModel, nextModelTake, fbReason, terminal := qc.NextModelOrRouteFallback(
				currRoute,
				currModel,
				currModelTake,
				maxI2VTakes,
				fallbackModels,
				lastRenderErr,
			)
			if terminal || (nextRoute == currRoute && nextRoute == model.RouteKenBurns) {
				p.NeedsReview = true
				allCands := state.ClipCandidates[p.PanelID]
				if len(allCands) > 0 {
					bestIdx := bestCandidateIndex(allCands)
					state.ClipCandidates[p.PanelID][bestIdx].Accepted = true
					if allCands[bestIdx].Route != "" {
						p.Route = allCands[bestIdx].Route
					}
					state.Errors = append(
						state.Errors,
						fmt.Sprintf(
							"panel %s: QC below threshold on terminal fallback lane (%s); flagged needs_review=true and accepted best take %d (identity=%.2f)",
							p.PanelID,
							p.Route,
							bestIdx+1,
							allCands[bestIdx].QC.IdentitySimilarity,
						),
					)
					slog.Warn("Panel QC below threshold on final fallback lane; flagging needs_review and accepting best take for assembly",
						"panel", p.PanelID, "lane", p.Route, "take", bestIdx+1,
						"identity", allCands[bestIdx].QC.IdentitySimilarity)
				}
				break
			}

			currRoute = nextRoute
			currModel = nextModel
			currModelTake = nextModelTake
			p.Route = nextRoute

			seed := model.ComputeDeterministicSeed(state.Config.ConfigHash, p.PanelID, take)
			fbClipPath := filepath.Join(runDir, "clips", fmt.Sprintf("%s_take%02d_%s.avi", p.PanelID, take, sanitizeFilenameToken(nextModel)))
			srcPath := state.CleanPlatePaths[p.PanelID]
			if srcPath == "" || !fileExists(srcPath) {
				srcPath = filepath.Join(runDir, "panels", p.PanelID+".png")
			}
			if !fileExists(srcPath) {
				return fmt.Errorf("qc fallback %s: source image %q not found", p.PanelID, srcPath)
			}

			opt := client.RenderOptions{
				PanelID:      p.PanelID,
				ModelID:      nextModel,
				Seed:         seed,
				ActionPrompt: p.Action,
				MotionType:   p.MotionPrompt,
				ComfyURL:     comfyURL,
			}

			t0 := time.Now()
			rendered, err := wc.RenderClip(ctx, srcPath, fbClipPath, nextRoute, p.Shot, p.DurationS, fps, resW, resH, opt)
			if err != nil {
				if nextRoute == model.RouteI2V {
					elapsedS := time.Since(t0).Seconds()
					if elapsedS > 0 {
						state.RecordModelCost(nextModel, elapsedS)
					}
					lastRenderErr = err.Error()
					state.ClipCandidates[p.PanelID] = append(state.ClipCandidates[p.PanelID], model.ClipCandidate{
						PanelID:        p.PanelID,
						Path:           filepath.ToSlash(fbClipPath),
						TakeNumber:     take,
						Seed:           seed,
						Route:          nextRoute,
						ModelID:        nextModel,
						GenSeconds:     elapsedS,
						FallbackReason: fbReason,
						RenderError:    err.Error(),
						Accepted:       false,
					})
					continue
				}
				return fmt.Errorf("qc fallback %s (%s): %w", p.PanelID, nextRoute, err)
			}

			lastRenderErr = ""
			routeUsed := nextRoute
			if rendered.RouteUsed != "" {
				routeUsed = model.AnimationRoute(rendered.RouteUsed)
				p.Route = routeUsed
			}
			passed, reasons := qc.EvaluateClip(rendered.QCScores, qcCfg, routeUsed)
			rendered.QCScores.OverallPass = passed
			rendered.QCScores.Reasons = reasons

			if rendered.GenSeconds > 0 && routeUsed == model.RouteI2V {
				state.RecordModelCost(nextModel, rendered.GenSeconds)
			}

			state.ClipCandidates[p.PanelID] = append(state.ClipCandidates[p.PanelID], model.ClipCandidate{
				PanelID:        p.PanelID,
				Path:           filepath.ToSlash(rendered.ClipPath),
				RawClipPath:    filepath.ToSlash(rendered.RawClipPath),
				TakeNumber:     take,
				Seed:           seed,
				Route:          routeUsed,
				ModelID:        nextModel,
				PositivePrompt: rendered.PositivePrompt,
				NegativePrompt: rendered.NegativePrompt,
				NativeWidth:    rendered.NativeWidth,
				NativeHeight:   rendered.NativeHeight,
				NativeFrames:   rendered.NativeFrames,
				NativeFPS:      rendered.NativeFPS,
				GenSeconds:     rendered.GenSeconds,
				PeakVRAMGB:     rendered.PeakVRAMGB,
				Cached:         rendered.Cached,
				FallbackReason: fbReason,
				QC:             rendered.QCScores,
				Accepted:       passed,
			})
			hasAccepted = passed
		}
	}

	if reportBytes, err := json.MarshalIndent(state.ClipCandidates, "", "  "); err == nil {
		_ = os.WriteFile(filepath.Join(qcDir, "qc_report.json"), reportBytes, 0644)
	}

	return nil
}

// TTSNode synthesizes dialogue speech audio files and synchronizes panel duration with spoken dialogue.
func TTSNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Generating TTS audio...")

	runDir := resolveRunDir(state, cfg)
	audioDir := filepath.Join(runDir, "audio")
	if err := os.MkdirAll(audioDir, 0755); err != nil {
		return fmt.Errorf("create audio dir %q: %w", audioDir, err)
	}

	state.AudioTracks = make([]model.AudioTrack, 0)
	wc := getWorkerClient(cfg)

	for i := range state.Panels {
		p := &state.Panels[i]
		panelSpeechDur := 0.0
		for dIdx, d := range p.Dialogue {
			safeSpk := sanitizeFilenameToken(d.SpeakerID)
			voiceID := d.SpeakerID
			if state.CharacterBible != nil {
				if entry, ok := state.CharacterBible.Characters[d.SpeakerID]; ok && entry.Voice.VoiceID != "" {
					voiceID = entry.Voice.VoiceID
				}
			}
			audioPath := filepath.Join(audioDir, fmt.Sprintf("%s_%02d_%s.wav", p.PanelID, dIdx+1, safeSpk))
			res, err := wc.SynthesizeTTS(ctx, d.Text, audioPath, voiceID, d.Emotion, d.Lang)
			if err != nil {
				return fmt.Errorf("tts %s line %d: %w", p.PanelID, dIdx+1, err)
			}
			dur := res.DurationS
			if res.AudioPath != "" {
				audioPath = res.AudioPath
			}
			panelSpeechDur += dur + 0.25
			state.AudioTracks = append(state.AudioTracks, model.AudioTrack{
				PanelID:   p.PanelID,
				SpeakerID: d.SpeakerID,
				AudioPath: filepath.ToSlash(audioPath),
				DurationS: dur,
				Emotion:   d.Emotion,
			})
		}
		if panelSpeechDur > 0 {
			p.DurationS = model.ClampDuration(math.Round(math.Max(p.DurationS, panelSpeechDur+0.3)*100) / 100)
		}
	}

	routesDir := filepath.Join(runDir, "routes")
	if err := os.MkdirAll(routesDir, 0755); err == nil {
		planFile := filepath.Join(routesDir, "routing_plan.json")
		updatedStructured := false
		if raw, err := os.ReadFile(planFile); err == nil {
			var plan RoutingPlan
			if json.Unmarshal(raw, &plan) == nil && len(plan.Panels) == len(state.Panels) {
				for i := range state.Panels {
					plan.Panels[i].DurationS = state.Panels[i].DurationS
					plan.Panels[i].Route = state.Panels[i].Route
				}
				if planBytes, err := json.MarshalIndent(plan, "", "  "); err == nil {
					_ = os.WriteFile(planFile, planBytes, 0644)
					updatedStructured = true
				}
			}
		}
		if !updatedStructured {
			if planBytes, err := json.MarshalIndent(state.Panels, "", "  "); err == nil {
				_ = os.WriteFile(planFile, planBytes, 0644)
			}
		}
	}

	return nil
}

// AssembleNode stitches video clips, transitions, subtitles, and audio tracks into the final deliverable.
func AssembleNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Assembling final video...")

	runDir := resolveRunDir(state, cfg)
	outDir := filepath.Join(runDir, "output")
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return fmt.Errorf("create output dir %q: %w", outDir, err)
	}

	// Persist current state so Python assembler reads latest routes, durations, accepted clips, and audio tracks
	if err := state.Save(runDir); err != nil {
		return fmt.Errorf("save state before assemble: %w", err)
	}

	totalDur := 0.0
	for _, p := range state.Panels {
		totalDur += p.DurationS
	}

	fps, resW, resH := renderParams(cfg)
	wc := getWorkerClient(cfg)
	res, err := wc.Assemble(ctx, runDir, fps, resW, resH, totalDur)
	if err != nil {
		return fmt.Errorf("assemble final video: %w", err)
	}
	slog.Info("Final anime video assembled successfully",
		"mp4", res.MP4Path,
		"avi", res.AVIPath,
		"webp", res.WebPPath,
		"duration_s", res.DurationS)

	return nil
}

// RegisterDefaultNodes attaches all stage handlers to the pipeline engine.
func RegisterDefaultNodes(e *Engine) {
	e.RegisterNode(model.StageIngest, IngestNode)
	e.RegisterNode(model.StageSegment, SegmentNode)
	e.RegisterNode(model.StageUnderstand, UnderstandNode)
	e.RegisterNode(model.StageCleanPlates, CleanPlatesNode)
	e.RegisterNode(model.StageRoute, RouteNode)
	e.RegisterNode(model.StageTTS, TTSNode)
	e.RegisterNode(model.StageGenerate, GenerateNode)
	e.RegisterNode(model.StageQC, QCNode)
	e.RegisterNode(model.StageAssemble, AssembleNode)
}

func resolveRunDir(state *model.RunState, cfg *config.PipelineConfig) string {
	if cfg != nil && cfg.General.RunDir != "" {
		return cfg.General.RunDir
	}
	if state != nil && state.Config.RunID != "" {
		return filepath.Join("runs", state.Config.RunID)
	}
	return "runs"
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func containsStr(slice []string, target string) bool {
	for _, s := range slice {
		if strings.EqualFold(s, target) {
			return true
		}
	}
	return false
}

func sanitizeFilenameToken(raw string) string {
	var b []byte
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			b = append(b, c)
		} else {
			b = append(b, '_')
		}
	}
	if len(b) == 0 {
		return "speaker"
	}
	return string(b)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		return err
	}
	return out.Close()
}
