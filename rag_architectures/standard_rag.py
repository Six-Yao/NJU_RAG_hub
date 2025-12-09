"""默认 RAG 实现：llama-index + Chroma。"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Sequence

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import (
	FilterOperator,
	MetadataFilter,
	MetadataFilters,
)
from openai import OpenAI

from config import Settings
from pipeline.base_rag import BaseRAG
from pipeline.ingest import build_embedding, build_storage_context
from tools.keywords import extract_keywords
from tools.temporal import TemporalQueryRewriter

logger = logging.getLogger(__name__)

_TAIL_KEYWORDS = ["最多", "最少", "至少", "人数", "人数上限", "人数下限", "多少", "几人", "规模"]


def _extract_tail_clause(question: str) -> str:
	for delimiter in ["，", ",", "。", ".", "？", "?"]:
		if delimiter in question:
			question = question.split(delimiter)[-1]
	return question.strip()


def _merge_terms(*term_groups: Sequence[str]) -> List[str]:
	seen = set()
	ordered: List[str] = []
	for group in term_groups:
		for term in group:
			cleaned = term.strip()
			if cleaned and cleaned not in seen:
				ordered.append(cleaned)
				seen.add(cleaned)
	return ordered


def _derive_focus_terms(question: str) -> List[str]:
	terms: List[str] = []
	for keyword in _TAIL_KEYWORDS:
		if keyword in question:
			terms.append(keyword)
	tail_clause = _extract_tail_clause(question)
	if tail_clause and tail_clause not in terms:
		terms.append(tail_clause)
	return _merge_terms(terms)


def _build_time_filters(question: str, enforce_time: bool = False) -> MetadataFilters | None:
	years = [int(token) for token in question.split() if token.isdigit() and len(token) == 4]
	filters: List[MetadataFilter] = []
	if years:
		filters.append(
			MetadataFilter(
				key="latest_year",
				value=min(years),
				operator=FilterOperator.GTE,
			)
		)
		filters.append(
			MetadataFilter(
				key="earliest_year",
				value=max(years),
				operator=FilterOperator.LTE,
			)
		)
	elif enforce_time:
		filters.append(
			MetadataFilter(
				key="earliest_year",
				value=0,
				operator=FilterOperator.GT,
			)
		)
	return MetadataFilters(filters=filters) if filters else None


def _merge_nodes(primary: List[NodeWithScore], secondary: Sequence[NodeWithScore]) -> List[NodeWithScore]:
	seen = {node.node.node_id for node in primary}
	merged = list(primary)
	for node in secondary:
		node_id = node.node.node_id
		if node_id not in seen:
			merged.append(node)
			seen.add(node_id)
	return merged


def _format_context(nodes: Sequence[NodeWithScore]) -> str:
	lines: List[str] = []
	for idx, node in enumerate(nodes, start=1):
		content = node.node.get_content(metadata_mode="all")
		lines.append(f"文档 {idx}: {content}")
	return "\n".join(lines)


def _build_prompt(question: str, context: str) -> str:
	return (
		"基于以下文档内容回答用户问题,回答不要无中生有，不要解释原因，尽可能简短并包含必要信息：\n\n"
		f"相关文档:\n{context}\n\n"
		f"用户问题: {question}\n\n"
		"请基于上述文档内容提供准确回答:"
	)


def _looks_like_task_question(question: str) -> bool:
	lowered = question.lower()
	keywords = ["任务", "安排", "学习组", "pbl", "值班", "计划", "周"]
	return any(token in question for token in keywords) or "task" in lowered


def _fallback_task_queries(question: str) -> List[str]:
	seeds: List[str] = []
	lowered = question.lower()
	core = question.replace("本周", "").replace("这周", "").strip() or question
	if "pbl" in lowered:
		seeds.append("PBL学习组任务安排")
		seeds.append("PBL学习组任务清单")
	if "学习组" in question:
		seeds.append("学习组任务安排")
	if "值班" in question:
		seeds.append("值班计划")
	if "任务" in question:
		seeds.append("本周任务安排")
	seeds.append(f"{core} 任务安排")
	deduped: List[str] = []
	seen = set()
	for phrase in seeds:
		cleaned = phrase.strip()
		if cleaned and cleaned not in seen:
			seen.add(cleaned)
			deduped.append(cleaned)
	return deduped[:3]


class StandardRAG(BaseRAG):
	name = "standard"

	def __init__(self, settings: Settings):
		super().__init__(settings=settings)
		self.similarity_top_k = self.settings.runtime.similarity_top_k
		self.answer_context_k = self.settings.runtime.answer_context_k
		self.enable_focus = self.settings.runtime.enable_focus_hints
		self.enable_time_filters = self.settings.runtime.enable_time_filters
		self.temperature = self.settings.runtime.llm_temperature

		self._index: VectorStoreIndex | None = None
		self._retriever: VectorIndexRetriever | None = None
		self._llm_client: OpenAI | None = None
		self._temporal_rewriter = TemporalQueryRewriter()

	def _ensure_index(self) -> VectorStoreIndex:
		if self._index is None:
			storage_context = build_storage_context(self.settings)
			embedding = build_embedding(self.settings)
			self._index = VectorStoreIndex.from_vector_store(
				vector_store=storage_context.vector_store,
				embed_model=embedding,
				show_progress=False,
			)
		return self._index

	def _ensure_retriever(self) -> VectorIndexRetriever:
		if self._retriever is None:
			index = self._ensure_index()
			self._retriever = VectorIndexRetriever(
				index=index,
				similarity_top_k=self.similarity_top_k,
			)
		return self._retriever

	def _ensure_llm_client(self) -> OpenAI:
		if not self.settings.llm.api_key:
			raise ValueError("LLM_API_KEY 缺失，无法执行 StandardRAG")
		if self._llm_client is None:
			self._llm_client = OpenAI(
				api_key=self.settings.llm.api_key,
				base_url=self.settings.llm.base_url,
			)
		return self._llm_client

	def _rerank_nodes(self, focus_terms: Sequence[str], nodes: List[NodeWithScore]) -> List[NodeWithScore]:
		if not focus_terms or not nodes:
			return list(nodes)

		def score(node: NodeWithScore) -> float:
			base = float(node.score or 0.0)
			content = node.node.get_content(metadata_mode="all")
			bonus = sum(1.5 for term in focus_terms if term and term in content)
			return base + bonus

		return sorted(nodes, key=score, reverse=True)

	def _build_sources(self, nodes: Sequence[NodeWithScore]) -> List[Dict[str, Any]]:
		sources: List[Dict[str, Any]] = []
		for node in nodes:
			content = node.node.get_content(metadata_mode="all")
			metadata = dict(node.node.metadata or {})
			sources.append(
				{
					"content": content,
					"metadata": metadata,
					"similarity": float(node.score or 0.0),
				}
			)
		return sources

	def _expand_task_queries(self, question: str) -> List[str]:
		return _fallback_task_queries(question) if _looks_like_task_question(question) else []

	def _run_pipeline(self, question: str, enforce_time: bool) -> Dict[str, Any]:
		retriever = self._ensure_retriever()
		llm_client = self._ensure_llm_client()

		heuristic_terms = _derive_focus_terms(question)
		keyword_terms: List[str] = []
		combined_terms: List[str] = []
		if self.enable_focus:
			try:
				keyword_terms = extract_keywords(
					question,
					llm_client=llm_client,
					model=self.settings.llm.model,
				)
			except Exception as exc:  # pragma: no cover
				logger.warning("关键词提取失败: %s", exc)
			combined_terms = _merge_terms(heuristic_terms, keyword_terms)

		focus_terms = combined_terms if self.enable_focus else []

		temporal_queries: List[str] = []
		extracted_date = self._temporal_rewriter.extract_and_normalize_date(question)
		if extracted_date:
			temporal_queries = self._temporal_rewriter.rewrite_for_interval_retrieval(question, extracted_date)

		task_queries = self._expand_task_queries(question)

		filters: MetadataFilters | None = None
		if self.enable_time_filters or enforce_time:
			filters = _build_time_filters(question, enforce_time=enforce_time)

		retrieve_kwargs = {}
		if filters is not None:
			retrieve_kwargs["metadata_filters"] = filters
		retrieval_candidates: List[str] = [question]
		retrieval_candidates.extend(temporal_queries)
		retrieval_candidates.extend(task_queries)
		if focus_terms:
			retrieval_candidates.append("; ".join(focus_terms[:4]))

		retrieval_queries: List[str] = []
		seen_queries: set[str] = set()
		for candidate in retrieval_candidates:
			normalized = (candidate or "").strip()
			if not normalized or normalized in seen_queries:
				continue
			seen_queries.add(normalized)
			retrieval_queries.append(normalized)

		ranked_nodes: List[NodeWithScore] = []
		for query_text in retrieval_queries:
			query_bundle = QueryBundle(query_text)
			nodes = retriever.retrieve(query_bundle, **retrieve_kwargs)
			ranked_nodes = _merge_nodes(ranked_nodes, nodes)

		if focus_terms:
			ranked_nodes = self._rerank_nodes(focus_terms, ranked_nodes)
		ranked_nodes = ranked_nodes[: self.answer_context_k]

		if not ranked_nodes:
			return {
				"question": question,
				"answer": "未检索到相关内容，请尝试重新描述问题。",
				"sources": [],
			}

		context = _format_context(ranked_nodes)
		prompt = _build_prompt(question, context)
		if focus_terms:
			prompt += "\n请重点回应以下关键词: %s" % ", ".join(focus_terms[:3])
		response = llm_client.chat.completions.create(
			model=self.settings.llm.model,
			messages=[
				{"role": "system", "content": "你是一个基于文档内容回答问题的助手。"},
				{"role": "user", "content": prompt},
			],
			temperature=self.temperature,
			max_tokens=800,
		)
		answer = (response.choices[0].message.content or "").strip()

		return {
			"question": question,
			"answer": answer,
			"sources": self._build_sources(ranked_nodes),
			"keywords": combined_terms or heuristic_terms,
			"llm_keywords": keyword_terms,
			"heuristic_keywords": heuristic_terms,
			"temporal_queries": temporal_queries,
			"task_queries": task_queries,
			"retrieval_queries": retrieval_queries,
		}

	def query(self, question: str, *, enforce_time: bool = False) -> Dict[str, Any]:
		return self._run_pipeline(question, enforce_time=enforce_time)

	async def async_query(self, question: str, *, enforce_time: bool = False) -> Dict[str, Any]:
		return await asyncio.to_thread(self.query, question, enforce_time=enforce_time)


__all__ = ["StandardRAG"]
