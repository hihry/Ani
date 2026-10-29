package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/m2a/manhwa2anime/internal/config"
	"github.com/m2a/manhwa2anime/internal/model"
	"github.com/m2a/manhwa2anime/internal/pipeline"
)

const version = "0.2.0"

func main() {
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})
	logger := slog.New(handler)
	slog.SetDefault(logger)

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	var err error
	switch os.Args[1] {
	case "run":
		err = runPipeline(os.Args[2:], logger)
	case "resume":
		err = resumePipeline(os.Args[2:], logger)
	case "inspect":
		err = inspectPipeline(os.Args[2:])
	case "version", "-v", "--version":
		printVersion()
		os.Exit(0)
	case "help", "-h", "--help":
		printUsage()
		os.Exit(0)
	default:
		fmt.Fprintf(os.Stderr, "Unknown subcommand: %q\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}

	if err != nil {
		logger.Error("Command failed", "error", err)
		os.Exit(1)
	}

	os.Exit(0)
}

func printUsage() {
	fmt.Println("Manhwa-to-Anime AI Pipeline CLI (m2a)")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  m2a <command> [flags]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  run       Run the full pipeline on input manhwa pages")
	fmt.Println("  resume    Resume a paused or failed pipeline run")
	fmt.Println("  inspect   Display structured inspection details for a run directory")
	fmt.Println("  version   Display CLI version information")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  m2a run --pages page1.png,page2.png --tier t1 --lang ko --auto-approve")
	fmt.Println("  m2a resume --run-dir runs/abc-123")
	fmt.Println("  m2a inspect --run-dir runs/abc-123")
}

func printVersion() {
	fmt.Printf("m2a v%s (Manhwa-to-Anime hybrid pipeline engine)\n", version)
}

func runPipeline(args []string, logger *slog.Logger) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	pagesFlag := fs.String("pages", "", "Comma-separated paths to source page images (required)")
	tierFlag := fs.String("tier", "t1", "Compute tier override (t0, t1s, t1, t2)")
	langFlag := fs.String("lang", "ko", "Source language (ko, ja, en)")
	configFlag := fs.String("config", "./configs", "Directory containing pipeline configuration files")
	outputFlag := fs.String("output", "./runs", "Directory where run artifacts are stored")
	runDirFlag := fs.String("run-dir", "", "Optional explicit run directory path")
	runIDFlag := fs.String("run-id", "", "Optional explicit run ID (UUID)")
	idAliasFlag := fs.String("id", "", "Alias for --run-id")
	autoApproveFlag := fs.Bool("auto-approve", false, "Automatically approve human review checkpoints CP1 and CP2")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runIDFlag == "" && *idAliasFlag != "" {
		*runIDFlag = *idAliasFlag
	}

	switch *langFlag {
	case "ko", "ja", "en":
	default:
		return fmt.Errorf("unsupported --lang %q (must be ko, ja, or en)", *langFlag)
	}
	switch *tierFlag {
	case "t0", "t1s", "t1", "t2":
	default:
		return fmt.Errorf("unsupported --tier %q (must be t0, t1s, t1, or t2)", *tierFlag)
	}

	if *pagesFlag == "" {
		return errors.New("missing required flag: --pages")
	}

	rawPages := strings.Split(*pagesFlag, ",")
	pages := make([]string, 0, len(rawPages))
	for _, p := range rawPages {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			pages = append(pages, trimmed)
		}
	}
	if len(pages) == 0 {
		return errors.New("--pages flag must contain at least one page file path")
	}

	state := model.NewRunState(pages, *langFlag, *tierFlag)
	if *runIDFlag != "" {
		state.Config.RunID = *runIDFlag
	}

	runDir := *runDirFlag
	if runDir == "" {
		runDir = filepath.Join(*outputFlag, state.Config.RunID)
	}
	if absRunDir, err := filepath.Abs(runDir); err == nil {
		runDir = absRunDir
	}

	var cfg *config.PipelineConfig
	var err error
	snapPath := filepath.Join(runDir, "config.snapshot.yaml")
	if _, statErr := os.Stat(snapPath); statErr == nil {
		cfg, err = config.Load(snapPath, "")
	} else {
		defaultConfigPath := filepath.Join(*configFlag, "default.yaml")
		tierConfigPath := filepath.Join(*configFlag, "tiers", *tierFlag+".yaml")
		cfg, err = config.Load(defaultConfigPath, tierConfigPath)
	}
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	state.Config.ConfigHash = model.ComputeConfigHash(pages, *langFlag, *tierFlag, cfg)
	if cfg.Composition.Resolution[0] > 0 && cfg.Composition.Resolution[1] > 0 {
		state.Config.OutputResolution = cfg.Composition.Resolution
	}
	cfg.General.RunDir = runDir

	if err := os.MkdirAll(runDir, 0755); err != nil {
		return fmt.Errorf("create run directory %q: %w", runDir, err)
	}

	if snapBytes, err := yaml.Marshal(cfg); err == nil {
		_ = os.WriteFile(filepath.Join(runDir, "config.snapshot.yaml"), snapBytes, 0644)
	}

	if err := state.Save(runDir); err != nil {
		return fmt.Errorf("save initial run state: %w", err)
	}

	logger.Info("Starting pipeline run",
		"run_id", state.Config.RunID,
		"run_dir", runDir,
		"tier", *tierFlag,
		"lang", *langFlag,
		"auto_approve", *autoApproveFlag,
		"pages", len(pages))

	engine := pipeline.NewEngine(runDir, logger)
	engine.SetAutoApproveCheckpoints(*autoApproveFlag)
	pipeline.RegisterDefaultNodes(engine)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	return engine.Run(ctx, state, cfg)
}

func resumePipeline(args []string, logger *slog.Logger) error {
	fs := flag.NewFlagSet("resume", flag.ContinueOnError)
	runDirFlag := fs.String("run-dir", "", "Path to existing run directory (required)")
	idFlag := fs.String("id", "", "Run ID under --output directory (alternative to --run-dir)")
	outputFlag := fs.String("output", "./runs", "Base output directory when using --id")
	approveFlag := fs.String("approve", "", "Optional checkpoint name to approve (cp1, cp2, cp3)")
	configFlag := fs.String("config", "./configs", "Directory containing pipeline configuration files")
	autoApproveFlag := fs.Bool("auto-approve", false, "Automatically approve remaining checkpoints")

	if err := fs.Parse(args); err != nil {
		return err
	}
	_ = approveFlag
	if *runDirFlag == "" && *idFlag != "" {
		*runDirFlag = filepath.Join(*outputFlag, *idFlag)
	}

	if *runDirFlag == "" {
		return errors.New("missing required flag: --run-dir")
	}

	runDir := *runDirFlag
	if absRunDir, err := filepath.Abs(runDir); err == nil {
		runDir = absRunDir
	}

	state, err := model.LoadRunState(runDir)
	if err != nil {
		return fmt.Errorf("load state from %q: %w", runDir, err)
	}

	tier := state.Config.ComputeTier
	if tier == "" {
		tier = "t1"
	}

	var cfg *config.PipelineConfig
	snapPath := filepath.Join(runDir, "config.snapshot.yaml")
	if _, statErr := os.Stat(snapPath); statErr == nil {
		cfg, err = config.Load(snapPath, "")
	} else {
		defaultConfigPath := filepath.Join(*configFlag, "default.yaml")
		tierConfigPath := filepath.Join(*configFlag, "tiers", tier+".yaml")
		cfg, err = config.Load(defaultConfigPath, tierConfigPath)
	}
	if err != nil {
		return fmt.Errorf("load config for resume (use --config): %w", err)
	}
	cfg.General.RunDir = runDir

	logger.Info("Resuming pipeline run",
		"run_id", state.Config.RunID,
		"current_stage", state.CurrentStage,
		"run_dir", runDir)

	engine := pipeline.NewEngine(runDir, logger)
	engine.SetAutoApproveCheckpoints(*autoApproveFlag)
	pipeline.RegisterDefaultNodes(engine)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	return engine.Resume(ctx, state, cfg)
}

func inspectPipeline(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	runDirFlag := fs.String("run-dir", "", "Path to existing run directory (required)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if *runDirFlag == "" {
		return errors.New("missing required flag: --run-dir")
	}

	state, err := model.LoadRunState(*runDirFlag)
	if err != nil {
		return fmt.Errorf("load state from %q: %w", *runDirFlag, err)
	}

	fmt.Println("==================================================")
	fmt.Println("             PIPELINE RUN INSPECTION              ")
	fmt.Println("==================================================")
	fmt.Printf("Run ID:          %s\n", state.Config.RunID)
	fmt.Printf("Current Stage:   %s\n", state.CurrentStage)
	fmt.Printf("Compute Tier:    %s\n", state.Config.ComputeTier)
	fmt.Printf("Source Language: %s\n", state.Config.SourceLanguage)
	fmt.Printf("Created At:      %s\n", state.Config.CreatedAt.Format(time.RFC3339))
	fmt.Printf("Source Pages:    %d pages\n", len(state.Config.SourcePages))
	for i, page := range state.Config.SourcePages {
		fmt.Printf("  [%d] %s\n", i+1, page)
	}

	fmt.Println("--------------------------------------------------")
	fmt.Printf("Number of Panels: %d\n", len(state.Panels))
	for _, p := range state.Panels {
		fmt.Printf("  - Panel %-10s | Shot: %-8s | Route: %-8s | Duration: %.1fs\n",
			p.PanelID, p.Shot, p.Route, p.DurationS)
		if p.Action != "" {
			fmt.Printf("      Action: %s\n", p.Action)
		}
		if len(p.Dialogue) > 0 {
			for _, d := range p.Dialogue {
				fmt.Printf("      Dialogue (%s, %s): %q\n", d.SpeakerID, d.Emotion, d.Text)
			}
		}
	}

	fmt.Println("--------------------------------------------------")
	fmt.Printf("Clean Plates:    %d files\n", len(state.CleanPlatePaths))
	fmt.Printf("Clip Candidates: %d panels\n", len(state.ClipCandidates))
	fmt.Printf("Audio Tracks:    %d tracks\n", len(state.AudioTracks))

	fmt.Println("--------------------------------------------------")
	fmt.Println("Errors:")
	if len(state.Errors) == 0 {
		fmt.Println("  (none)")
	} else {
		for i, e := range state.Errors {
			fmt.Printf("  %d. %s\n", i+1, e)
		}
	}

	fmt.Println("--------------------------------------------------")
	fmt.Println("Cost Ledger:")
	fmt.Printf("  Total Wall Time: %.2f seconds\n", state.CostLedger.TotalWallSeconds)
	fmt.Printf("  Total GPU Time:  %.2f seconds\n", state.CostLedger.TotalGPUSeconds)
	if len(state.CostLedger.PerStage) > 0 {
		fmt.Println("  Per-Stage Breakdown:")
		for st, sec := range state.CostLedger.PerStage {
			fmt.Printf("    %-16s: %6.2f s\n", st, sec)
		}
	}
	if len(state.CostLedger.PerModel) > 0 {
		fmt.Println("  Per-Model Breakdown:")
		for md, sec := range state.CostLedger.PerModel {
			fmt.Printf("    %-16s: %6.2f s\n", md, sec)
		}
	}

	fmt.Println("--------------------------------------------------")
	fmt.Println("Stage History:")
	for i, tr := range state.StageHistory {
		endStr := "in progress"
		if tr.EndedAt != nil {
			dur := tr.EndedAt.Sub(tr.StartedAt)
			endStr = fmt.Sprintf("completed in %v", dur.Round(time.Millisecond))
		}
		errStr := ""
		if tr.Error != "" {
			errStr = fmt.Sprintf(" [ERROR: %s]", tr.Error)
		}
		fmt.Printf("  %2d. %-14s : %s%s\n", i+1, tr.Stage, endStr, errStr)
	}
	fmt.Println("==================================================")

	return nil
}
