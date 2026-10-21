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
	"sync"

	"github.com/m2a/manhwa2anime/internal/client"
	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
	"github.com/m2a/manhwa2anime/internal/qc"
)

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
	if cfg != nil && cfg.QC.IdentityThreshold > 0 {
		qcCfg = cfg.QC
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
	wc := client.NewWorkerClient(cfg)
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

	wc := client.NewWorkerClient(cfg)
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

	wc := client.NewWorkerClient(cfg)
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

	wc := client.NewWorkerClient(cfg)
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

// RouteNode assigns animation routes and camera motion prompts to each panel.
func RouteNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Routing panels...")

	runDir := resolveRunDir(state, cfg)
	routesDir := filepath.Join(runDir, "routes")
	if err := os.MkdirAll(routesDir, 0755); err != nil {
		return fmt.Errorf("create routes dir %q: %w", routesDir, err)
	}

	allowI2V := cfg != nil && cfg.Generation.PrimaryModel != "kenburns" && cfg.Generation.PrimaryModel != "parallax" && state.Config.ComputeTier != "t0"
	gpuBudget := 30
	if !allowI2V {
		gpuBudget = 0
	}

	for i := range state.Panels {
		p := &state.Panels[i]
		isHero := (p.Shot == model.ShotCloseUp || p.Shot == model.ShotSplash) && p.MotionIntensity >= 0.55

		if isHero && allowI2V && gpuBudget >= 10 {
			p.Route = model.RouteI2V
			gpuBudget -= 10
		} else if (p.Shot == model.ShotMedium || p.Shot == model.ShotCloseUp || p.Shot == model.ShotSplash) && (p.MotionIntensity >= 0.35 || i%2 == 1) {
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

	if planBytes, err := json.MarshalIndent(state.Panels, "", "  "); err == nil {
		_ = os.WriteFile(filepath.Join(routesDir, "routing_plan.json"), planBytes, 0644)
	}

	return nil
}

// GenerateNode handles animation generation for all panels based on their assigned routes.
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

	wc := client.NewWorkerClient(cfg)
	fps, resW, resH := renderParams(cfg)
	qcCfg := effectiveQCConfig(cfg)

	type genResult struct {
		cand      model.ClipCandidate
		routeUsed model.AnimationRoute
		reasons   []string
		err       error
	}
	results := make([]genResult, len(state.Panels))

	sem := make(chan struct{}, maxPanelWorkers())
	var wg sync.WaitGroup
	for i := range state.Panels {
		i := i
		p := state.Panels[i]
		srcPath := state.CleanPlatePaths[p.PanelID]
		if srcPath == "" || !fileExists(srcPath) {
			srcPath = filepath.Join(runDir, "panels", p.PanelID+".png")
		}
		if !fileExists(srcPath) {
			results[i] = genResult{err: fmt.Errorf("generate %s: source image %q not found", p.PanelID, srcPath)}
			continue
		}

		clipPath := filepath.Join(clipsDir, p.PanelID+"_clip_01.avi")
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			rendered, err := wc.RenderClip(ctx, srcPath, clipPath, p.Route, p.Shot, p.DurationS, fps, resW, resH)
			if err != nil {
				results[i] = genResult{err: fmt.Errorf("generate %s (%s): %w", p.PanelID, p.Route, err)}
				return
			}
			passed, reasons := qc.EvaluateClip(rendered.QCScores, qcCfg)
			rendered.QCScores.OverallPass = passed
			routeUsed := p.Route
			if rendered.RouteUsed != "" {
				routeUsed = model.AnimationRoute(rendered.RouteUsed)
			}
			results[i] = genResult{
				cand: model.ClipCandidate{
					PanelID:    p.PanelID,
					Path:       filepath.ToSlash(rendered.ClipPath),
					TakeNumber: 1,
					QC:         rendered.QCScores,
					Accepted:   passed,
				},
				routeUsed: routeUsed,
				reasons:   reasons,
			}
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

// QCNode verifies quality and consistency metrics on candidate video clips and degrades failed clips to fallback lanes.
func QCNode(ctx context.Context, state *model.RunState, cfg *config.PipelineConfig) error {
	slog.Info("Running quality checks...")

	runDir := resolveRunDir(state, cfg)
	qcDir := filepath.Join(runDir, "qc")
	if err := os.MkdirAll(qcDir, 0755); err != nil {
		return fmt.Errorf("create qc dir %q: %w", qcDir, err)
	}

	qcCfg := effectiveQCConfig(cfg)
	wc := client.NewWorkerClient(cfg)
	fps, resW, resH := renderParams(cfg)
	maxRetries := 2
	if cfg != nil && cfg.Generation.MaxRetries > 0 {
		maxRetries = cfg.Generation.MaxRetries
	}

	for i := range state.Panels {
		p := &state.Panels[i]
		candidates := state.ClipCandidates[p.PanelID]
		hasAccepted := false

		for j := range candidates {
			passed, _ := qc.EvaluateClip(candidates[j].QC, qcCfg)
			candidates[j].QC.OverallPass = passed
			candidates[j].Accepted = passed
			if passed {
				hasAccepted = true
			}
		}
		state.ClipCandidates[p.PanelID] = candidates

		for take := len(candidates) + 1; !hasAccepted; take++ {
			nextRoute := qc.NextFallbackRoute(p.Route)
			if nextRoute == p.Route || take > maxRetries+1 {
				// Reached terminal fallback lane; do not re-render identical clip.
				// Select best candidate by identity similarity and record a review warning.
				allCands := state.ClipCandidates[p.PanelID]
				if len(allCands) > 0 {
					bestIdx := bestCandidateIndex(allCands)
					state.ClipCandidates[p.PanelID][bestIdx].Accepted = true
					slog.Warn("Panel QC below threshold on final fallback lane; accepting best take for assembly",
						"panel", p.PanelID, "lane", p.Route, "take", bestIdx+1,
						"identity", allCands[bestIdx].QC.IdentitySimilarity)
				}
				break
			}

			p.Route = nextRoute
			fbClipPath := filepath.Join(runDir, "clips", fmt.Sprintf("%s_take%02d_%s.avi", p.PanelID, take, nextRoute))
			srcPath := state.CleanPlatePaths[p.PanelID]
			if srcPath == "" || !fileExists(srcPath) {
				srcPath = filepath.Join(runDir, "panels", p.PanelID+".png")
			}
			if !fileExists(srcPath) {
				return fmt.Errorf("qc fallback %s: source image %q not found", p.PanelID, srcPath)
			}

			rendered, err := wc.RenderClip(ctx, srcPath, fbClipPath, nextRoute, p.Shot, p.DurationS, fps, resW, resH)
			if err != nil {
				return fmt.Errorf("qc fallback %s (%s): %w", p.PanelID, nextRoute, err)
			}
			passed, _ := qc.EvaluateClip(rendered.QCScores, qcCfg)
			rendered.QCScores.OverallPass = passed
			if rendered.RouteUsed != "" {
				p.Route = model.AnimationRoute(rendered.RouteUsed)
			}

			state.ClipCandidates[p.PanelID] = append(state.ClipCandidates[p.PanelID], model.ClipCandidate{
				PanelID:    p.PanelID,
				Path:       filepath.ToSlash(rendered.ClipPath),
				TakeNumber: take,
				QC:         rendered.QCScores,
				Accepted:   passed,
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
	wc := client.NewWorkerClient(cfg)

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
		if planBytes, err := json.MarshalIndent(state.Panels, "", "  "); err == nil {
			_ = os.WriteFile(filepath.Join(routesDir, "routing_plan.json"), planBytes, 0644)
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

	fps, resW, resH := renderParams(cfg)
	wc := client.NewWorkerClient(cfg)
	res, err := wc.Assemble(ctx, runDir, fps, resW, resH)
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
		if s == target {
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
