"""统一的文档入库与索引构建流程。"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import Settings, load_settings
from tools.temporal import extract_temporal_metadata
from pipeline.chunking import semantic_split
from pipeline.embeddings import OpenRouterEmbedding

logger = logging.getLogger(__name__)
_METADATA_BATCH_SIZE = 512


def build_embedding(settings: Settings):
	try:
		return OpenAIEmbedding(
			model=settings.embedding.model,
			api_key=settings.embedding.api_key,
			base_url=settings.embedding.base_url,
		)
	except ValueError as exc:
		if "OpenAIEmbeddingModelType" not in str(exc):
			raise
		return OpenRouterEmbedding(
			api_key=settings.embedding.api_key,
			base_url=settings.embedding.base_url,
			model=settings.embedding.model,
		)


def _fetch_documents(settings: Settings, doc_ids: Sequence[int] | None = None) -> Iterable[Dict[str, str]]:
	conn = sqlite3.connect(str(settings.storage.sqlite_path))
	try:
		cursor = conn.cursor()
		query = "SELECT id, title, content FROM documents"
		params: Tuple[Any, ...] = ()
		if doc_ids:
			placeholders = ",".join("?" for _ in doc_ids)
			query += f" WHERE id IN ({placeholders})"
			params = tuple(doc_ids)
		cursor.execute(query, params)
		rows = cursor.fetchall()
	finally:
		conn.close()

	for doc_id, title, content in rows:
		yield {
			"doc_id": str(doc_id),
			"title": title or f"Document-{doc_id}",
			"content": content or "",
		}


def _normalize_doc_id(value: Any) -> str | None:
	if value is None:
		return None
	return str(value)


def _iter_collection_metadatas(collection, batch_size: int = _METADATA_BATCH_SIZE) -> Iterator[Dict[str, Any]]:
	offset = 0
	while True:
		response = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
		metadatas = response.get("metadatas", []) or []
		if not metadatas:
			break
		for metadata in metadatas:
			if metadata:
				yield metadata
		if len(metadatas) < batch_size:
			break
		offset += batch_size


def _gather_existing_doc_ids(vector_store: ChromaVectorStore) -> set[str]:
	collection = getattr(vector_store, "_collection", None)
	doc_ids: set[str] = set()
	if collection is None:
		return doc_ids
	for metadata in _iter_collection_metadatas(collection):
		doc_id = _normalize_doc_id(metadata.get("doc_id"))
		if doc_id:
			doc_ids.add(doc_id)
	return doc_ids


def _delete_existing_entries(vector_store: ChromaVectorStore, doc_ids: Sequence[str]) -> None:
	if not doc_ids:
		return
	collection = getattr(vector_store, "_collection", None)
	if collection is None:
		return
	where = {"doc_id": {"$in": list(doc_ids)}}
	collection.delete(where=where)


def _apply_temporal_metadata(node: BaseNode) -> None:
	temporal_meta = extract_temporal_metadata(node.get_content(metadata_mode="all"))
	if temporal_meta:
		node.metadata.update(temporal_meta)


def build_storage_context(settings: Settings, *, reset_collection: bool = False) -> StorageContext:
	chroma_client = chromadb.PersistentClient(path=str(settings.storage.chroma_path))
	if reset_collection:
		try:
			chroma_client.delete_collection(name="knowledge_base")
		except Exception:  # pragma: no cover
			pass
	vector_store = ChromaVectorStore(
		chroma_collection=chroma_client.get_or_create_collection(
			name="knowledge_base",
			metadata={"description": settings.description},
		)
	)
	return StorageContext.from_defaults(vector_store=vector_store)


def _ingest_documents(
	settings: Settings,
	*,
	force: bool = False,
	target_doc_ids: Sequence[str] | None = None,
	show_progress: bool = False,
) -> Tuple[VectorStoreIndex, Dict[str, int]]:
	requested_doc_ids = tuple(int(doc_id) for doc_id in target_doc_ids or [])
	embedding = build_embedding(settings)

	reset_collection = bool(force and not requested_doc_ids)
	storage_context = build_storage_context(settings, reset_collection=reset_collection)
	vector_store = storage_context.vector_store

	documents = list(_fetch_documents(settings, requested_doc_ids or None))
	nodes: List[BaseNode] = []
	total_nodes_generated = 0

	for doc in documents:
		chunks = semantic_split(doc["content"], embedding)
		if not chunks:
			continue
		total_nodes_generated += len(chunks)
		for idx, chunk in enumerate(chunks):
			metadata = {
				"doc_id": doc["doc_id"],
				"title": doc["title"],
				"chunk_index": idx,
			}
			node = TextNode(
				text=chunk,
				metadata=metadata,
				ref_doc_id=doc["doc_id"],
			)
			_apply_temporal_metadata(node)
			nodes.append(node)

	skipped_docs: set[str] = set()
	if requested_doc_ids:
		doc_id_strings = [str(doc_id) for doc_id in requested_doc_ids]
		_delete_existing_entries(vector_store, doc_id_strings)
	elif not reset_collection:
		existing_doc_ids = _gather_existing_doc_ids(vector_store)
		filtered: List[BaseNode] = []
		for node in nodes:
			node.metadata = dict(node.metadata or {})
			doc_id = _normalize_doc_id(node.metadata.get("doc_id") or node.ref_doc_id)
			if doc_id and doc_id in existing_doc_ids:
				skipped_docs.add(doc_id)
				continue
			if doc_id:
				node.metadata["doc_id"] = doc_id
			filtered.append(node)
		nodes = filtered

	if not nodes:
		index = VectorStoreIndex.from_vector_store(
			vector_store=vector_store,
			embed_model=embedding,
			show_progress=show_progress,
		)
		stats = {
			"requested_documents": len(documents),
			"skipped_documents": len(skipped_docs),
			"ingested_nodes": 0,
			"generated_nodes": total_nodes_generated,
		}
		return index, stats

	index = VectorStoreIndex(
		nodes,
		storage_context=storage_context,
		embed_model=embedding,
		show_progress=show_progress,
	)
	stats = {
		"requested_documents": len(documents),
		"skipped_documents": len(skipped_docs),
		"ingested_nodes": len(nodes),
		"generated_nodes": total_nodes_generated,
	}
	return index, stats


def build_index(
	settings: Settings | None = None,
	*,
	force: bool = False,
	target_doc_ids: Sequence[str] | None = None,
	show_progress: bool = False,
	return_details: bool = False,
):
	settings = settings or load_settings()
	index, stats = _ingest_documents(
		settings,
		force=force,
		target_doc_ids=target_doc_ids,
		show_progress=show_progress,
	)
	if return_details:
		return index, stats
	return index


def run_ingest(argv: Sequence[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Build or refresh the shared vector index")
	parser.add_argument("--force", action="store_true", help="重置知识库并全量重建")
	parser.add_argument("--doc-ids", nargs="+", help="仅刷新指定文档 ID")
	parser.add_argument("--no-progress", action="store_true", help="关闭进度条输出")
	args = parser.parse_args(argv)

	settings = load_settings()
	target_ids = tuple(args.doc_ids) if args.doc_ids else None
	_, stats = build_index(
		settings,
		force=args.force,
		target_doc_ids=target_ids,
		show_progress=not args.no_progress,
		return_details=True,
	)

	print("读取 %d 篇文档，解析出 %d 个节点。" % (stats["requested_documents"], stats["generated_nodes"]))
	if stats["ingested_nodes"] == 0:
		print("没有新的节点需要写入，已接入现有向量集合。")
	else:
		print(
			"索引构建完成：新增 %d 个节点，涉及 %d 篇文档。"
			% (stats["ingested_nodes"], stats["requested_documents"])
		)
	if stats["skipped_documents"]:
		print("跳过 %d 篇已存在的文档。" % stats["skipped_documents"])


__all__ = ["build_index", "run_ingest", "build_storage_context", "build_embedding"]
