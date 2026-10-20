package model

// VoiceProfile defines TTS settings for a character.
type VoiceProfile struct {
	VoiceID            string `json:"voice_id"`
	ReferenceAudioPath string `json:"reference_audio_path,omitempty"`
	Style              string `json:"style"` // "calm", "energetic", etc.
}

// CharacterEntry is one entry in the character bible.
type CharacterEntry struct {
	Name           string       `json:"name"`
	Description    string       `json:"description"`
	ReferenceCrops []string     `json:"reference_crops"` // Paths to reference images
	Voice          VoiceProfile `json:"voice"`
	Appearances    []string     `json:"appearances"`     // Panel IDs
}

// CharacterBible maps character IDs to their full profiles.
type CharacterBible struct {
	Characters map[string]CharacterEntry `json:"characters"`
}

// NewCharacterBible initializes an empty CharacterBible with its map ready for use.
func NewCharacterBible() *CharacterBible {
	return &CharacterBible{
		Characters: make(map[string]CharacterEntry),
	}
}
