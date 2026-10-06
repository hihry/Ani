# Manhwa-to-Anime AI Pipeline

Transforms static Manhwa pages into animated anime clips.

## Architecture (4-Layer)
1. **Perception**: Panel and bubble detection, OCR, character face extraction.
2. **Understanding**: Visual Language Models (VLMs) structure narratives and action sequences.
3. **Generation**: Text-to-Video / Image-to-Video models generate animated shots, TTS generates audio.
4. **Composition**: Stitching, audio mixing, visual effects.

## Quick Start
```bash
# Install
make dev

# Configure
# Edit configs/default.yaml as needed

# Run
make run
```

## Structure
- `m2a/`: Core application package
- `configs/`: Configuration files
- `data/`: Inputs and outputs
- `models/`: Downloaded model weights

## Compute Tiers
| Tier | VRAM      | Primary Model    | VLM             |
|------|-----------|------------------|-----------------|
| T0   | CPU Only  | KenBurns (rules) | Qwen3-VL-2B     |
| T1   | 12-16 GB  | Wan2.2-Ti2V-5B   | Qwen3-VL-8B-4bit|
| T2   | 24-48 GB  | Wan2.2-I2V-A14B  | Qwen3-VL-8B     |

## License
Apache 2.0
