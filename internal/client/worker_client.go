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

// NewWorkerClient constructs a WorkerClient from config.
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
		baseURL: fmt.Sprintf("http://%s:%d", host, port),
		httpClient: &http.Client{
			Timeout: 180 * time.Second,
		},
		cliScript: resolveCLIScript(),
		pythonBin: resolvePythonBin(),
	}
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

func (c *WorkerClient) postJSON(ctx context.Context, endpoint string, payload any, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

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

// IngestResult holds stitched strip dimensions.
type IngestResult struct {
	StitchedPath string `json:"stitched_path"`
	Width        int    `json:"width"`
	Height       int    `json:"height"`
}

// Ingest stitches vertical manhwa pages into a single strip.
func (c *WorkerClient) Ingest(ctx context.Context, pages []string, outputPath string) (*IngestResult, error) {
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
func (c *WorkerClient) Understand(ctx context.Context, panelImagePath string, panelID string, lang string, readingOrder int) (*UnderstandResult, error) {
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

// RenderResult holds the rendered clip path and measured QC scores.
type RenderResult struct {
	ClipPath    string         `json:"clip_path"`
	FramesCount int            `json:"frames_count"`
	DurationS   float64        `json:"duration_s"`
	RouteUsed   string         `json:"route_used"`
	QCScores    model.QCScores `json:"qc_scores"`
}

// RenderClip renders a panel animation clip and returns its QC scores.
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
) (*RenderResult, error) {
	var res RenderResult
	payload := map[string]any{
		"image_path":  imagePath,
		"output_path": outputPath,
		"route":       string(route),
		"shot_type":   string(shot),
		"duration_s":  durationS,
		"fps":         fps,
		"width":       width,
		"height":      height,
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
	if err := c.call(ctx, "/render", payload, args, &res); err != nil {
		return nil, err
	}
	return &res, nil
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
func (c *WorkerClient) Assemble(ctx context.Context, runDir string, fps int, width int, height int) (*AssembleResult, error) {
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
