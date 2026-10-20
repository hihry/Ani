package model

// BBox represents a bounding box in pixel coordinates.
type BBox struct {
	X int `json:"x"`
	Y int `json:"y"`
	W int `json:"w"`
	H int `json:"h"`
}

// DialogueLine represents a single speech bubble or text block.
type DialogueLine struct {
	SpeakerID  string  `json:"speaker_id"`
	Text       string  `json:"text"`
	Lang       string  `json:"lang"`    // "ko", "ja", "en"
	Emotion    string  `json:"emotion"` // "neutral", "angry", "sad", etc.
	BBox       *BBox   `json:"bbox,omitempty"`
	Confidence float64 `json:"confidence"`
}

// SFX represents a sound effect text region.
type SFX struct {
	Text string `json:"text"`
	Kind string `json:"kind"` // "impact", "ambient", "motion"
	BBox *BBox  `json:"bbox,omitempty"`
}

// CharacterRef is a reference to a character detected in a panel.
type CharacterRef struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Emotion  string `json:"emotion"`
	Pose     string `json:"pose"`
	CropPath string `json:"crop_path,omitempty"`
}

// ShotType categorizes the framing of a panel.
type ShotType string

const (
	ShotCloseUp ShotType = "close_up"
	ShotMedium  ShotType = "medium"
	ShotWide    ShotType = "wide"
	ShotSplash  ShotType = "splash"
)

// AnimationRoute determines how a panel will be animated.
type AnimationRoute string

const (
	RouteI2V      AnimationRoute = "i2v"      // AI image-to-video
	RouteParallax AnimationRoute = "parallax" // 2.5D depth parallax
	RouteKenBurns AnimationRoute = "kenburns" // Pan & zoom
)

// PanelJSON is THE core data contract — the structured representation of a single manhwa panel.
type PanelJSON struct {
	PanelID         string         `json:"panel_id"`
	BBox            BBox           `json:"bbox"`
	Shot            ShotType       `json:"shot"`
	Characters      []CharacterRef `json:"characters"`
	Action          string         `json:"action"`           // One-sentence description
	Dialogue        []DialogueLine `json:"dialogue"`
	SFX             []SFX          `json:"sfx"`
	MotionPrompt    string         `json:"motion_prompt"`
	StyleAnchor     string         `json:"style_anchor"`     // Default: "2D anime, flat colors, clean line art"
	MotionIntensity float64        `json:"motion_intensity"` // 0.0 to 1.0
	Route           AnimationRoute `json:"route"`            // Determined by shot router
	DurationS       float64        `json:"duration_s"`       // 0.5 to 10.0
}

// NewPanelJSON constructs a new PanelJSON with sensible pipeline defaults:
// StyleAnchor = "2D anime, flat colors, clean line art", MotionIntensity = 0.3,
// Route = RouteKenBurns, DurationS = 4.0.
func NewPanelJSON(id string, bbox BBox) PanelJSON {
	return PanelJSON{
		PanelID:         id,
		BBox:            bbox,
		Shot:            ShotMedium,
		Characters:      make([]CharacterRef, 0),
		Dialogue:        make([]DialogueLine, 0),
		SFX:             make([]SFX, 0),
		StyleAnchor:     "2D anime, flat colors, clean line art",
		MotionIntensity: 0.3,
		Route:           RouteKenBurns,
		DurationS:       4.0,
	}
}
