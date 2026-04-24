import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ModelConfig:
    """模型配置"""

    model_name_or_path: str


@dataclass
class SamplingConfig:
    """采样配置"""

    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 1024


@dataclass
class ServerConfig:
    """服务配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    ttl: float = 1800
    max_image_buffer_size: int = 2


@dataclass
class Config:
    """总配置"""

    model: ModelConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """从 YAML 文件加载配置"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            model=ModelConfig(data["model_name_or_path"]),
            sampling=SamplingConfig(**data.get("sampling", {})),
            server=ServerConfig(**data.get("server", {})),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """从字典加载配置"""
        return cls(
            model=ModelConfig(**data["model"]),
            sampling=SamplingConfig(**data.get("sampling", {})),
            server=ServerConfig(**data.get("server", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model.__dict__,
            "sampling": self.sampling.__dict__,
            "server": self.server.__dict__,
        }

    def validate(self):
        """验证配置"""

        # 检查 tokenizer 目录
        if not Path(self.model.model_name_or_path).exists():
            raise FileNotFoundError(
                f"tokenizer_dir not found: {self.model.model_name_or_path}"
            )

        logger.info("Config validation passed")


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置"""
    global _config
    if _config is None:
        raise RuntimeError("Config not initialized. Call load_config() first.")
    return _config


def load_config(config_path: str = "config.yaml") -> Config:
    """加载配置"""
    global _config
    _config = Config.from_yaml(config_path)
    _config.validate()
    logger.info(f"Config loaded from {config_path}")
    return _config
