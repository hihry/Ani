"""Generate synthetic evaluation manhwa pages and ground truth annotations.

Creates 3 vertical-scroll pages with realistic panels, gutters, dialogue bubbles,
and characters, and writes data/eval/ground_truth.json following ground_truth_schema.json.
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

OUTPUT_DIR = Path("data/eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors (Anime / Manhwa aesthetic)
BG_WHITE = (255, 255, 255)
PANEL_BORDER = (20, 20, 24)
BUBBLE_BG = (255, 255, 255)
BUBBLE_BORDER = (40, 40, 45)
TEXT_COLOR = (20, 20, 20)

PALETTES = [
    {"sky": (135, 206, 235), "ground": (76, 153, 0), "char": (220, 80, 60), "accent": (255, 200, 100)},
    {"sky": (40, 30, 70), "ground": (25, 20, 45), "char": (90, 160, 240), "accent": (230, 80, 180)},
    {"sky": (240, 180, 120), "ground": (160, 90, 50), "char": (60, 180, 120), "accent": (240, 220, 80)},
]

PAGE_SPECS = [
    {
        "page_file": "sample_page_1.png",
        "width": 800,
        "height": 2200,
        "panels": [
            {
                "id": "p1_panel_1",
                "bbox": {"x": 40, "y": 60, "w": 720, "h": 500},
                "reading_order": 1,
                "shot_type": "wide",
                "action": "Jin stands before the towering Gates of Awakening at dawn.",
                "dialogue": [
                    {"text": "The gate opened... after three years.", "speaker": "Jin", "lang": "en", "emotion": "serious", "bbox": {"x": 70, "y": 90, "w": 300, "h": 70}}
                ],
                "characters": [{"id": "char_jin", "name": "Jin", "emotion": "determined", "pose": "standing full body"}],
                "palette_idx": 0,
            },
            {
                "id": "p1_panel_2",
                "bbox": {"x": 60, "y": 640, "w": 680, "h": 460},
                "reading_order": 2,
                "shot_type": "medium",
                "action": "A mysterious shadow creature emerges from the swirling rift.",
                "dialogue": [
                    {"text": "GROOOOOWL!", "speaker": "Shadow Beast", "lang": "en", "emotion": "angry", "bbox": {"x": 450, "y": 670, "w": 250, "h": 60}}
                ],
                "characters": [{"id": "char_beast", "name": "Shadow Beast", "emotion": "hostile", "pose": "emerging"}],
                "palette_idx": 1,
            },
            {
                "id": "p1_panel_3",
                "bbox": {"x": 50, "y": 1180, "w": 700, "h": 920},
                "reading_order": 3,
                "shot_type": "close_up",
                "action": "Close-up of Jin's glowing blue eyes as his dagger ignites with mana.",
                "dialogue": [
                    {"text": "I won't run away this time.", "speaker": "Jin", "lang": "en", "emotion": "confident", "bbox": {"x": 80, "y": 1220, "w": 280, "h": 75}}
                ],
                "characters": [{"id": "char_jin", "name": "Jin", "emotion": "intense", "pose": "eye close-up"}],
                "palette_idx": 2,
            },
        ],
    },
    {
        "page_file": "sample_page_2.png",
        "width": 800,
        "height": 2100,
        "panels": [
            {
                "id": "p2_panel_1",
                "bbox": {"x": 50, "y": 70, "w": 700, "h": 580},
                "reading_order": 4,
                "shot_type": "medium",
                "action": "Jin dashes forward leaving a trail of blue light sparks.",
                "dialogue": [
                    {"text": "Shadow Step: First Form!", "speaker": "Jin", "lang": "en", "emotion": "focused", "bbox": {"x": 420, "y": 100, "w": 300, "h": 65}}
                ],
                "characters": [{"id": "char_jin", "name": "Jin", "emotion": "action", "pose": "dashing"}],
                "palette_idx": 1,
            },
            {
                "id": "p2_panel_2",
                "bbox": {"x": 40, "y": 730, "w": 720, "h": 620},
                "reading_order": 5,
                "shot_type": "splash",
                "action": "Massive clash as the dagger slices through the beast's armored claw.",
                "dialogue": [
                    {"text": "CRASH!!", "speaker": "SFX", "lang": "en", "emotion": "impact", "bbox": {"x": 100, "y": 760, "w": 200, "h": 80}}
                ],
                "characters": [
                    {"id": "char_jin", "name": "Jin", "emotion": "fierce", "pose": "slashing"},
                    {"id": "char_beast", "name": "Shadow Beast", "emotion": "pain", "pose": "recoiling"},
                ],
                "palette_idx": 0,
            },
            {
                "id": "p2_panel_3",
                "bbox": {"x": 60, "y": 1430, "w": 680, "h": 580},
                "reading_order": 6,
                "shot_type": "medium",
                "action": "The beast shatters into dark energy particles.",
                "dialogue": [
                    {"text": "Target eliminated. Quest complete.", "speaker": "System", "lang": "en", "emotion": "neutral", "bbox": {"x": 200, "y": 1470, "w": 380, "h": 60}}
                ],
                "characters": [{"id": "char_jin", "name": "Jin", "emotion": "calm", "pose": "sheathing dagger"}],
                "palette_idx": 2,
            },
        ],
    },
    {
        "page_file": "sample_page_3.png",
        "width": 800,
        "height": 2000,
        "panels": [
            {
                "id": "p3_panel_1",
                "bbox": {"x": 50, "y": 60, "w": 700, "h": 520},
                "reading_order": 7,
                "shot_type": "close_up",
                "action": "Jin catches his breath under the gentle morning light.",
                "dialogue": [
                    {"text": "Level up... plus three stats.", "speaker": "Jin", "lang": "en", "emotion": "relieved", "bbox": {"x": 80, "y": 90, "w": 280, "h": 65}}
                ],
                "characters": [{"id": "char_jin", "name": "Jin", "emotion": "tired", "pose": "half face"}],
                "palette_idx": 0,
            },
            {
                "id": "p3_panel_2",
                "bbox": {"x": 40, "y": 660, "w": 720, "h": 600},
                "reading_order": 8,
                "shot_type": "medium",
                "action": "A mysterious girl in white armor watches him from the clifftop.",
                "dialogue": [
                    {"text": "Who is that hunter? I've never seen him in the guild.", "speaker": "Elora", "lang": "en", "emotion": "curious", "bbox": {"x": 380, "y": 700, "w": 340, "h": 75}}
                ],
                "characters": [{"id": "char_elora", "name": "Elora", "emotion": "curious", "pose": "observing from cliff"}],
                "palette_idx": 1,
            },
            {
                "id": "p3_panel_3",
                "bbox": {"x": 50, "y": 1340, "w": 700, "h": 580},
                "reading_order": 9,
                "shot_type": "wide",
                "action": "Establishing shot of the horizon showing the capital city in the distance.",
                "dialogue": [
                    {"text": "The next chapter begins now.", "speaker": "Narrator", "lang": "en", "emotion": "epic", "bbox": {"x": 100, "y": 1380, "w": 320, "h": 60}}
                ],
                "characters": [],
                "palette_idx": 2,
            },
        ],
    },
]

def draw_panel(draw: ImageDraw.ImageDraw, panel: dict):
    bx = panel["bbox"]["x"]
    by = panel["bbox"]["y"]
    bw = panel["bbox"]["w"]
    bh = panel["bbox"]["h"]
    pal = PALETTES[panel["palette_idx"]]
    
    # Gradient / layered background inside panel
    draw.rectangle([bx, by, bx + bw, by + bh], fill=pal["sky"])
    ground_y = by + int(bh * 0.6)
    draw.rectangle([bx, ground_y, bx + bw, by + bh], fill=pal["ground"])
    
    # Sun / Accent circle
    accent_r = min(bw, bh) // 8
    draw.ellipse([bx + bw - accent_r * 2 - 40, by + 30, bx + bw - 40, by + 30 + accent_r * 2], fill=pal["accent"])
    
    # Stylized Character shape
    cx = bx + bw // 2
    cy = ground_y - 40
    cw = min(bw, bh) // 4
    ch = min(bw, bh) // 3
    draw.polygon([
        (cx, cy - ch),
        (cx - cw // 2, cy),
        (cx + cw // 2, cy),
    ], fill=pal["char"])
    # Character head
    head_r = cw // 4
    draw.ellipse([cx - head_r, cy - ch - head_r * 2, cx + head_r, cy - ch], fill=(255, 220, 190))
    
    # Dark panel border (distinct outline)
    draw.rectangle([bx, by, bx + bw, by + bh], outline=PANEL_BORDER, width=4)
    
    # Speech bubbles
    for d in panel["dialogue"]:
        db = d["bbox"]
        dx, dy, dw, dh = db["x"], db["y"], db["w"], db["h"]
        # Rounded speech bubble
        draw.rounded_rectangle([dx, dy, dx + dw, dy + dh], radius=12, fill=BUBBLE_BG, outline=BUBBLE_BORDER, width=3)
        # Tail pointer
        draw.polygon([(dx + 20, dy + dh), (dx + 35, dy + dh), (dx + 15, dy + dh + 15)], fill=BUBBLE_BG, outline=BUBBLE_BORDER)
        # Text label
        draw.text((dx + 15, dy + 15), d["text"][:35], fill=TEXT_COLOR)

def generate_all():
    all_panels_for_gt = []
    source_page_names = []
    
    for spec in PAGE_SPECS:
        page_file = spec["page_file"]
        source_page_names.append(page_file)
        w, h = spec["width"], spec["height"]
        
        # White background (manhwa vertical gutter canvas)
        img = Image.new("RGB", (w, h), BG_WHITE)
        draw = ImageDraw.Draw(img)
        
        for p in spec["panels"]:
            draw_panel(draw, p)
            # Record for ground truth
            gt_panel = {
                "panel_id": p["id"],
                "page": page_file,
                "bbox": p["bbox"],
                "reading_order": p["reading_order"],
                "shot_type": p["shot_type"],
                "action_description": p["action"],
                "dialogue": [
                    {
                        "text": d["text"],
                        "speaker": d["speaker"],
                        "language": d["lang"],
                        "emotion": d["emotion"],
                        "bbox": d["bbox"],
                    }
                    for d in p["dialogue"]
                ],
                "characters": p["characters"],
            }
            all_panels_for_gt.append(gt_panel)
            
        img_path = OUTPUT_DIR / page_file
        img.save(img_path, quality=95)
        print(f"Generated {img_path} ({w}x{h}, {len(spec['panels'])} panels)")
        
    ground_truth = {
        "source_pages": source_page_names,
        "panels": all_panels_for_gt,
        "characters": {
            "char_jin": {
                "name": "Jin",
                "description": "Black-haired shadow hunter with glowing blue eyes.",
                "appearances": ["p1_panel_1", "p1_panel_3", "p2_panel_1", "p2_panel_2", "p2_panel_3", "p3_panel_1"],
            },
            "char_beast": {
                "name": "Shadow Beast",
                "description": "Armored demonic beast with dark aura.",
                "appearances": ["p1_panel_2", "p2_panel_2"],
            },
            "char_elora": {
                "name": "Elora",
                "description": "Guild scout in white armor.",
                "appearances": ["p3_panel_2"],
            },
        },
    }
    
    gt_path = OUTPUT_DIR / "ground_truth.json"
    if not gt_path.exists():
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)
        print(f"Generated {gt_path} with {len(all_panels_for_gt)} total panels.")

if __name__ == "__main__":
    generate_all()
