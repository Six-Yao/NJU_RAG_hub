"""根据名称实例化具体的 RAG 架构。"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from config import Settings, load_settings
from pipeline.base_rag import BaseRAG

Builder = Callable[[Optional[Settings]], BaseRAG]
_ARCHITECTURE_BUILDERS: Dict[str, Builder] = {}


def register_architecture(name: str, builder: Builder) -> None:
	normalized = name.lower()
	_ARCHITECTURE_BUILDERS[normalized] = builder


def available_architectures() -> List[str]:
	return sorted(_ARCHITECTURE_BUILDERS.keys())


def build_rag(name: str = "standard", settings: Optional[Settings] = None) -> BaseRAG:
	normalized = name.lower()
	if normalized not in _ARCHITECTURE_BUILDERS:
		options = ", ".join(available_architectures()) or "<empty>"
		raise ValueError(f"未注册架构: {name}. 可选: {options}")
	builder = _ARCHITECTURE_BUILDERS[normalized]
	return builder(settings or load_settings())


# 延迟注册，避免循环依赖
from rag_architectures.standard_rag import StandardRAG  # noqa: E402

register_architecture(StandardRAG.name, lambda settings: StandardRAG(settings=settings))


__all__ = ["register_architecture", "available_architectures", "build_rag"]
