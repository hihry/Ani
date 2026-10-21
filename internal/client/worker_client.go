package client

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
)

// httpStatusError indicates the HTTP worker server was reached and returned an application error (4xx/5xx).
// Work must NOT be re-executed via CLI when the server already processed and rejected the request.
type httpStatusError struct {
	statusCode int
	endpoint   string
	detail     string
}

func (e *httpStatusError) Error() string {
	if e.detail != "" {
		return fmt.Sprintf("worker HTTP %d on %s: %s", e.statusCode, e.endpoint, e.detail)
	}
	return fmt.Sprintf("worker HTTP %d on %s", e.statusCode, e.endpoint)
}

// WorkerClient communicates with the Python AI Worker service over HTTP (:8100)
// and transparently falls back to invoking workers/pipeline_cli.py when the HTTP
// daemon is not running.
type WorkerClient struct {
	baseURL         string
	httpClient      *http.Client
	cliScript       string
	pythonBin       string
	httpUnavailable atomic.Bool
}

// NewWorkerClient constructs a WorkerClient from config without a global HTTP timeout;
// per-call context deadlines govern each stage.
func NewWorkerClient(cfg *config.PipelineConfig) *WorkerClient {
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

	return &WorkerClient{
		baseURL:    fmt.Sprintf("http://%s:%d", host, port),
		httpClient: &http.Client{},
		cliScript:  resolveCLIScript(),
		pythonBin:  resolvePythonBin(),
	}
}

func withDefaultTimeout(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	if _, hasDeadline := ctx.Deadline(); hasDeadline {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, d)
}

func resolveCLIScript() string {
	if envPath := os.Getenv("M2A_WORKER_CLI"); envPath != "" {
		if abs, err := filepath.Abs(envPath); err == nil {
			return abs
		}
		return envPath
	}
	candidates := []string{
		"workers/pipeline_cli.py",
		"../workers/pipeline_cli.py",
		"../../workers/pipeline_cli.py",
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates = append(candidates, filepath.Join(exeDir, "workers", "pipeline_cli.py"))
	}
	for _, c := range candidates {
		if info, err := os.Stat(c); err == nil && !info.IsDir() {
			if abs, err := filepath.Abs(c); err == nil {
				return abs
			}
			return c
		}
	}
	return "workers/pipeline_cli.py"
}

func (c *WorkerClient) doJSON(ctx context.Context, method string, endpoint string, payload any, out any) error {
	var bodyReader io.Reader
	if payload != nil {
		body, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		bodyReader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+endpoint, bodyReader)
	if err != nil {
		return err
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return &httpStatusError{
			statusCode: resp.StatusCode,
			endpoint:   endpoint,
			detail:     strings.TrimSpace(string(errBody)),
		}
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

func (c *WorkerClient) postJSON(ctx context.Context, endpoint string, payload any, out any) error {
	return c.doJSON(ctx, http.MethodPost, endpoint, payload, out)
}

func resolvePythonBin() string {
	if envPy := os.Getenv("M2A_PYTHON"); envPy != "" {
		return envPy
	}
	for _, name := range []string{"python3", "python"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		misePy := filepath.Join(home, ".local", "share", "mise", "shims", "python")
		if _, err := os.Stat(misePy); err == nil {
			return misePy
		}
	}
	return "python3"
}

func (c *WorkerClient) runCLI(ctx context.Context, args []string, out any) error {
	cmdArgs := append([]string{c.cliScript}, args...)
	cmd := exec.CommandContext(ctx, c.pythonBin, cmdArgs...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		errText := strings.TrimSpace(stderr.String())
		if len(errText) > 2000 {
			errText = errText[len(errText)-2000:]
		}
		if errText != "" {
			return fmt.Errorf("python %v failed: %w\nstderr: %s", cmdArgs, err, errText)
		}
		return fmt.Errorf("python %v failed: %w", cmdArgs, err)
	}
	if out != nil {
		lines := strings.Split(strings.TrimSpace(string(output)), "\n")
		lastLine := lines[len(lines)-1]
		if err := json.Unmarshal([]byte(lastLine), out); err != nil {
			return fmt.Errorf("decode worker CLI json %q: %w", lastLine, err)
		}
	}
	return nil
}

// call executes the request via HTTP when the server is reachable, and only falls back to CLI
// when the HTTP daemon is unreachable (dial/connection error). HTTP 4xx/5xx or context cancellation
// errors are returned directly without re-executing via CLI.
func (c *WorkerClient) call(ctx context.Context, endpoint string, payload any, cliArgs []string, out any) error {
	if !c.httpUnavailable.Load() {
		err := c.postJSON(ctx, endpoint, payload, out)
		if err == nil {
			return nil
		}
		var statusErr *httpStatusError
		if errors.As(err, &statusErr) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		// Connection failed (HTTP server not running); remember for subsequent calls on this client
		c.httpUnavailable.Store(true)
	}
	return c.runCLI(ctx, cliArgs, out)
}

// HealthResponse describes Python worker daemon status and video model readiness.
type HealthResponse struct {
	Status         string            `json:"status"`
	Device         string            `json:"device"`
	Workers        map[string]string `json:"workers"`
	VideoProviders map[string]string `json:"video_providers"`
}

// Health queries GET /health on the Python worker server.
func (c *WorkerClient) Health(ctx context.Context) (*HealthResponse, error) {
	ctx, cancel := withDefaultTimeout(ctx, 5*time.Second)
	defer cancel()
	var res HealthResponse
	if err := c.doJSON(ctx, http.MethodGet, "/health", nil, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// FreeGPU requests the worker server to unload GPU models and release VRAM (`POST /gpu/free`).
func (c *WorkerClient) FreeGPU(ctx context.Context) error {
	if c.httpUnavailable.Load() {
		return nil
	}
	ctx, cancel := withDefaultTimeout(ctx, 15*time.Second)
	defer cancel()
	_ = c.postJSON(ctx, "/gpu/free", map[string]any{}, nil)
	return nil
}

// IngestResult holds stitched strip dimensions.
type IngestResult struct {
	StitchedPath string `json:"stitched_path"`
	Width        int    `json:"width"`
	Height       int    `json:"height"`
}

// Ingest stitches vertical manhwa pages into a single strip.
func (c *WorkerClient) Ingest(ctx context.Context, pages []string, outputPath string) (*IngestResult, error) {
	ctx, cancel := withDefaultTimeout(ctx, 180*time.Second)
	defer cancel()
	var res IngestResult
	payload := map[string]any{
		"page_paths":   pages,
		"output_path":  outputPath,
		"target_width": 800,
	}
	args := []string{
		"ingest",
		"--pages", strings.Join(pages, ","),
		"--output", outputPath,
	}
	if err := c.call(ctx, "/ingest", payload, args, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// DetectedPanel represents a segmented panel returned by the worker.
type DetectedPanel struct {
	PanelID      string     `json:"panel_id"`
	BBox         model.BBox `json:"bbox"`
	ReadingOrder int        `json:"reading_order"`
	ShotType     string     `json:"shot_type"`
	CropPath     string     `json:"crop_path"`
}

// SegmentResult holds all detected panels.
type SegmentResult struct {
	Panels []DetectedPanel `json:"panels"`
}

// Segment runs panel detection and saves individual panel crops.
func (c *WorkerClient) Segment(ctx context.Context, imagePath string, outputDir string, method string) (*SegmentResult, error) {
	ctx, cancel := withDefaultTimeout(ctx, 180*time.Second)
	defer cancel()
	var res SegmentResult
	payload := map[string]any{
		"image_path": imagePath,
		"output_dir": outputDir,
		"method":     method,
	}
	args := []string{
		"segment",
		"--image", imagePath,
		"--output-dir", outputDir,
	}
	if err := c.call(ctx, "/segment", payload, args, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// UnderstandResult holds VLM/semantic extraction for a single panel.
type UnderstandResult struct {
	PanelID         string               `json:"panel_id"`
	Shot            model.ShotType       `json:"shot"`
	Action          string               `json:"action"`
	MotionPrompt    string               `json:"motion_prompt"`
	MotionIntensity float64              `json:"motion_intensity"`
	Dialogue        []model.DialogueLine `json:"dialogue"`
	SFX             []model.SFX          `json:"sfx"`
	Characters      []model.CharacterRef `json:"characters"`
}

// Understand analyzes a panel image for scene semantics, dialogue, and characters.
func (c *WorkerClient) Understand(ctx context.Context, panelImagePath string, panelID string, lang string, readingOrder int, opts ...string) (*UnderstandResult, error) {
	ctx, cancel := withDefaultTimeout(ctx, 180*time.Second)
	defer cancel()
	var res UnderstandResult
	payload := map[string]any{
		"panel_image_path": panelImagePath,
		"panel_id":         panelID,
		"language":         lang,
		"reading_order":    readingOrder,
	}
	args := []string{
		"understand",
		"--image", panelImagePath,
		"--panel-id", panelID,
		"--lang", lang,
		"--order", strconv.Itoa(readingOrder),
	}
	if len(opts) > 0 && opts[0] != "" {
		payload["crops_dir"] = opts[0]
		args = append(args, "--crops-dir", opts[0])
	}
	if len(opts) > 1 && opts[1] != "" {
		payload["annotated_path"] = opts[1]
		args = append(args, "--annotated", opts[1])
	}
	if err := c.call(ctx, "/understand", payload, args, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// InpaintResult holds the path to the generated clean plate.
type InpaintResult struct {
	OutputPath string `json:"output_path"`
	Success    bool   `json:"success"`
}

// Inpaint removes speech bubbles and text regions from a panel image.
func (c *WorkerClient) Inpaint(ctx context.Context, imagePath string, outputPath string, bboxes []model.BBox) (*InpaintResult, error) {
	ctx, cancel := withDefaultTimeout(ctx, 180*time.Second)
	defer cancel()
	var res InpaintResult
	payload := map[string]any{
		"image_path":    imagePath,
		"output_path":   outputPath,
		"bubble_bboxes": bboxes,
	}
	bboxesJSON, _ := json.Marshal(bboxes)
	args := []string{
		"inpaint",
		"--image", imagePath,
		"--output", outputPath,
		"--bboxes", string(bboxesJSON),
	}
	if err := c.call(ctx, "/inpaint", payload, args, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// RenderOptions holds optional Phase 3 I2V metadata for RenderClip.
type RenderOptions struct {
	PanelID      string
	ModelID      string
	Seed         int
	ActionPrompt string
	MotionType   string
	Emotion      string
	ComfyURL     string
}

// RenderResult holds the rendered clip path, provenance metadata, and measured QC scores.
type RenderResult struct {
	ClipPath       string         `json:"clip_path"`
	RawClipPath    string         `json:"raw_clip_path,omitempty"`
	FramesCount    int            `json:"frames_count"`
	DurationS      float64        `json:"duration_s"`
	RouteUsed      string         `json:"route_used"`
	ModelID        string         `json:"model_id,omitempty"`
	Seed           int            `json:"seed,omitempty"`
	PositivePrompt string         `json:"positive_prompt,omitempty"`
	NegativePrompt string         `json:"negative_prompt,omitempty"`
	NativeWidth    int            `json:"native_width,omitempty"`
	NativeHeight   int            `json:"native_height,omitempty"`
	NativeFrames   int            `json:"native_frames,omitempty"`
	NativeFPS      int            `json:"native_fps,omitempty"`
	GenSeconds     float64        `json:"gen_seconds,omitempty"`
	PeakVRAMGB     float64        `json:"peak_vram_gb,omitempty"`
	Cached         bool           `json:"cached,omitempty"`
	Error          string         `json:"error,omitempty"`
	QCScores       model.QCScores `json:"qc_scores"`
}

type renderJobCreateResp struct {
	JobID string `json:"job_id"`
}

type renderJobStatusResp struct {
	JobID  string        `json:"job_id"`
	Status string        `json:"status"`
	Result *RenderResult `json:"result,omitempty"`
	Error  string        `json:"error,omitempty"`
}

// RenderClip renders a panel animation clip and returns its QC scores.
// For RouteI2V when the HTTP server is running, it submits an async job to POST /render/jobs,
// polls GET /render/jobs/{id}, and sends DELETE /render/jobs/{id} if ctx is cancelled.
func (c *WorkerClient) RenderClip(
	ctx context.Context,
	imagePath string,
	outputPath string,
	route model.AnimationRoute,
	shot model.ShotType,
	durationS float64,
	fps int,
	width int,
	height int,
	opts ...RenderOptions,
) (*RenderResult, error) {
	timeout := 10 * time.Minute
	if route == model.RouteI2V {
		timeout = 30 * time.Minute
	}
	ctx, cancel := withDefaultTimeout(ctx, timeout)
	defer cancel()

	var opt RenderOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	payload := map[string]any{
		"image_path":    imagePath,
		"output_path":   outputPath,
		"route":         string(route),
		"shot_type":     string(shot),
		"duration_s":    durationS,
		"fps":           fps,
		"width":         width,
		"height":        height,
		"panel_id":      opt.PanelID,
		"model_id":      opt.ModelID,
		"seed":          opt.Seed,
		"action_prompt": opt.ActionPrompt,
		"motion_type":   opt.MotionType,
		"emotion":       opt.Emotion,
		"comfy_url":     opt.ComfyURL,
	}

	args := []string{
		"render",
		"--image", imagePath,
		"--output", outputPath,
		"--route", string(route),
		"--shot", string(shot),
		"--duration", fmt.Sprintf("%.2f", durationS),
		"--fps", strconv.Itoa(fps),
		"--width", strconv.Itoa(width),
		"--height", strconv.Itoa(height),
	}
	if opt.ModelID != "" {
		args = append(args, "--model-id", opt.ModelID)
	}
	if opt.Seed != 0 {
		args = append(args, "--seed", strconv.Itoa(opt.Seed))
	}
	if opt.ActionPrompt != "" {
		args = append(args, "--prompt", opt.ActionPrompt)
	}
	if opt.ComfyURL != "" {
		args = append(args, "--comfy-url", opt.ComfyURL)
	}

	// For I2V over HTTP, prefer async job submission + polling (`POST /render/jobs`)
	if route == model.RouteI2V && !c.httpUnavailable.Load() {
		var created renderJobCreateResp
		err := c.postJSON(ctx, "/render/jobs", payload, &created)
		if err == nil && created.JobID != "" {
			return c.pollRenderJob(ctx, created.JobID)
		}
		var statusErr *httpStatusError
		if errors.As(err, &statusErr) && statusErr.statusCode != http.StatusNotFound {
			return nil, err
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
	}

	var res RenderResult
	if err := c.call(ctx, "/render", payload, args, &res); err != nil {
		return nil, err
	}
	if res.Error != "" {
		return &res, fmt.Errorf("render error: %s", res.Error)
	}
	return &res, nil
}

func (c *WorkerClient) pollRenderJob(ctx context.Context, jobID string) (*RenderResult, error) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			delCtx, delCancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = c.doJSON(delCtx, http.MethodDelete, "/render/jobs/"+jobID, nil, nil)
			delCancel()
			return nil, ctx.Err()
		case <-ticker.C:
			var st renderJobStatusResp
			if err := c.doJSON(ctx, http.MethodGet, "/render/jobs/"+jobID, nil, &st); err != nil {
				if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
					delCtx, delCancel := context.WithTimeout(context.Background(), 5*time.Second)
					_ = c.doJSON(delCtx, http.MethodDelete, "/render/jobs/"+jobID, nil, nil)
					delCancel()
				}
				return nil, err
			}
			switch st.Status {
			case "done":
				if st.Result == nil {
					return nil, fmt.Errorf("render job %s completed with nil result", jobID)
				}
				if st.Result.Error != "" {
					return st.Result, fmt.Errorf("render job %s failed: %s", jobID, st.Result.Error)
				}
				return st.Result, nil
			case "failed", "cancelled":
				errMsg := st.Error
				if errMsg == "" && st.Result != nil {
					errMsg = st.Result.Error
				}
				if errMsg == "" {
					errMsg = "job " + st.Status
				}
				return st.Result, fmt.Errorf("render job %s %s: %s", jobID, st.Status, errMsg)
			}
		}
	}
}

// TTSResult holds synthesized speech audio metadata.
type TTSResult struct {
	AudioPath string  `json:"audio_path"`
	DurationS float64 `json:"duration_s"`
}

// SynthesizeTTS generates a playable WAV audio file for a dialogue line.
func (c *WorkerClient) SynthesizeTTS(
	ctx context.Context,
	text string,
	outputPath string,
	voiceID string,
	emotion string,
	lang string,
) (*TTSResult, error) {
	ctx, cancel := withDefaultTimeout(ctx, 180*time.Second)
	defer cancel()
	var res TTSResult
	payload := map[string]any{
		"text":        text,
		"output_path": outputPath,
		"voice_id":    voiceID,
		"emotion":     emotion,
		"language":    lang,
	}
	args := []string{
		"tts",
		"--text", text,
		"--output", outputPath,
		"--voice-id", voiceID,
		"--emotion", emotion,
		"--lang", lang,
	}
	if err := c.call(ctx, "/tts", payload, args, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// AssembleResult holds final compiled video deliverables.
type AssembleResult struct {
	TotalFrames   int     `json:"total_frames"`
	DurationS     float64 `json:"duration_s"`
	AVIPath       string  `json:"avi_path"`
	WebPPath      string  `json:"webp_path"`
	MP4Path       string  `json:"mp4_path"`
	TimelinePath  string  `json:"timeline_path"`
	SubtitlesPath string  `json:"subtitles_path"`
}

// Assemble compiles the final video timeline from state.json in runDir.
func (c *WorkerClient) Assemble(ctx context.Context, runDir string, fps int, width int, height int, totalDurationS ...float64) (*AssembleResult, error) {
	assembleTimeout := 5 * time.Minute
	if len(totalDurationS) > 0 && totalDurationS[0] > 0 {
		assembleTimeout = 30*time.Second + time.Duration(3.0*totalDurationS[0]*float64(time.Second))
	}
	ctx, cancel := withDefaultTimeout(ctx, assembleTimeout)
	defer cancel()
	var res AssembleResult
	payload := map[string]any{
		"run_dir": filepath.ToSlash(runDir),
		"fps":     fps,
		"width":   width,
		"height":  height,
	}
	args := []string{
		"assemble",
		"--run-dir", runDir,
		"--fps", strconv.Itoa(fps),
		"--width", strconv.Itoa(width),
		"--height", strconv.Itoa(height),
	}
	if err := c.call(ctx, "/assemble", payload, args, &res); err != nil {
		return nil, err
	}
	return &res, nil
}
