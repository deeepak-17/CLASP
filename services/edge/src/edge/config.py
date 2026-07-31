from dataclasses import dataclass

@dataclass
class ModelProfile:
    name: str
    model_id: str
    max_new_tokens: int = 256

PROFILES = {
    "dev": ModelProfile(
        name='dev',
        model_id='deepseek-ai/deepseek-coder-1.3b-base'
    ),
    "target": ModelProfile(
        name='target',
        model_id='deepseek-ai/deepseek-coder-6.7b-base'
    )
}
