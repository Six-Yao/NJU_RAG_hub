"""RAG 架构接口定义。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from config import Settings, load_settings


class BaseRAG(ABC):
	"""所有 RAG 架构的统一接口。"""

	name = "base"

	def __init__(self, settings: Optional[Settings] = None) -> None:
		self.settings = settings or load_settings()

	@abstractmethod
	def query(self, question: str, *, enforce_time: bool = False) -> Dict[str, Any]:
		raise NotImplementedError

	@abstractmethod
	async def async_query(self, question: str, *, enforce_time: bool = False) -> Dict[str, Any]:
		raise NotImplementedError


__all__ = ["BaseRAG"]
