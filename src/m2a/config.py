import yaml
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings

class ModelConfig(BaseModel):
    """Model specific configuration."""
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024

class PipelineConfig(BaseSettings):
    """Main pipeline configuration mapping to YAML settings."""
    model_config = ConfigDict(env_prefix="M2A_", frozen=False)
    
    tier: str = "default"
    resolution: str = "1920x1080"
    target_fps: int = 24
    
    llm: ModelConfig = ModelConfig(model_name="gpt-4")
    tts: ModelConfig = ModelConfig(model_name="elevenlabs")
    video_gen: ModelConfig = ModelConfig(model_name="svd")
    
    # Enable merging with extra kwargs
    extra_configs: dict = {}

def load_config(tier: str | None = None, overrides: dict | None = None) -> PipelineConfig:
    """Loads and merges pipeline configurations.
    
    TODO: Implement actual YAML reading from default.yaml and tier.yaml.
    For now, returns a merged PipelineConfig from kwargs.
    """
    config_kwargs = {}
    
    if tier:
        config_kwargs["tier"] = tier
        
    if overrides:
        config_kwargs.update(overrides)
        
    return PipelineConfig(**config_kwargs)
