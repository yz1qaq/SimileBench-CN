"""Main SymbolicSimCoT classifier for Task 4.

Pair extraction and cultural interpretation are upstream inputs. This module
loads them from a separate directory; it never uses gold pairs as predictions.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel


LABELS = ("好", "乐", "怒", "哀", "惧", "恶", "惊", "无情绪")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def _read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required data file is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return records


def load_split(
    dataset_path: str | Path,
    extraction_path: str | Path,
) -> list[dict[str, Any]]:
    """Join gold emotion labels with separate, predicted pair analyses.

    Every extraction row must contain sentence_id, sentence, and pair_analyses.
    Each pair analysis must provide tenor, vehicle, cultural_implication, and
    combined_meaning. Strict joining prevents silent sentence-only training.
    """
    samples = _read_json_records(dataset_path)
    extracted = _read_json_records(extraction_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in extracted:
        sentence_id = str(row.get("sentence_id", ""))
        if not sentence_id or sentence_id in by_id:
            raise ValueError(f"Missing or duplicate extraction sentence_id: {sentence_id!r}")
        by_id[sentence_id] = row

    joined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        sentence_id = str(sample.get("sentence_id", ""))
        if not sentence_id or sentence_id in seen:
            raise ValueError(f"Missing or duplicate dataset sentence_id: {sentence_id!r}")
        seen.add(sentence_id)
        if sentence_id not in by_id:
            raise ValueError(f"No extraction result for sentence_id={sentence_id}")
        result = by_id[sentence_id]
        if result.get("sentence") != sample.get("sentence"):
            raise ValueError(f"Sentence mismatch for sentence_id={sentence_id}")
        analyses = result.get("pair_analyses")
        if not isinstance(analyses, list):
            raise ValueError(f"Missing pair_analyses for sentence_id={sentence_id}")
        for analysis in analyses:
            if not isinstance(analysis, dict) or any(
                not str(analysis.get(key, "")).strip()
                for key in ("tenor", "vehicle", "cultural_implication", "combined_meaning")
            ):
                raise ValueError(f"Incomplete pair analysis for sentence_id={sentence_id}")
        emotion = sample.get("emotion")
        if emotion not in LABEL_TO_ID:
            raise ValueError(f"Unknown emotion for sentence_id={sentence_id}: {emotion!r}")
        joined.append(
            {
                "sentence_id": sentence_id,
                "sentence": sample["sentence"],
                "emotion": emotion,
                "pair_analyses": analyses,
            }
        )

    extra_ids = set(by_id) - seen
    if extra_ids:
        raise ValueError(f"Extraction file contains {len(extra_ids)} unknown sentence IDs")
    if not any(row["pair_analyses"] for row in joined):
        raise ValueError("All extraction results are empty; graph inputs are unavailable")
    return joined


def class_weights_from_training(records: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    """Paper weight: number of samples / (number of classes * class count)."""
    counts = Counter(row["emotion"] for row in records)
    if set(counts) != set(LABELS):
        raise ValueError("Training records must contain all eight emotion classes")
    total = len(records)
    return torch.tensor(
        [total / (len(LABELS) * counts[label]) for label in LABELS],
        dtype=torch.float32,
    )


def pair_node_text(pair: Mapping[str, Any]) -> str:
    """Equation (6): cultural implication concatenated with combined meaning."""
    return f"{pair['cultural_implication'].strip()} {pair['combined_meaning'].strip()}"


def adjacency_from_shared_tenor(pairs: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    """Equation (8): undirected same-tenor edges and self-loops."""
    tenors = [str(pair["tenor"]).strip() for pair in pairs]
    matrix = torch.eye(len(tenors), dtype=torch.bool)
    for i, tenor in enumerate(tenors):
        for j in range(i + 1, len(tenors)):
            if tenor == tenors[j]:
                matrix[i, j] = matrix[j, i] = True
    return matrix


class SimileGraphDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        pairs = record["pair_analyses"]
        return {
            "sentence_id": record["sentence_id"],
            "sentence": record["sentence"],
            "pair_texts": [pair_node_text(pair) for pair in pairs],
            "adj": adjacency_from_shared_tenor(pairs),
            "label": LABEL_TO_ID[record["emotion"]],
        }


class GraphCollator:
    """Tokenize and pad to the largest pair count in each batch."""

    def __init__(self, tokenizer: Any, sent_max_len: int = 128, pair_max_len: int = 128):
        self.tokenizer = tokenizer
        self.sent_max_len = sent_max_len
        self.pair_max_len = pair_max_len

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        batch_size = len(batch)
        max_pairs = max(1, *(len(item["pair_texts"]) for item in batch))
        sentence_tokens = self.tokenizer(
            [item["sentence"] for item in batch],
            truncation=True,
            padding="max_length",
            max_length=self.sent_max_len,
            return_tensors="pt",
        )
        texts = [
            text
            for item in batch
            for text in list(item["pair_texts"]) + [""] * (max_pairs - len(item["pair_texts"]))
        ]
        pair_tokens = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.pair_max_len,
            return_tensors="pt",
        )
        pair_mask = torch.zeros(batch_size, max_pairs, dtype=torch.bool)
        adj = torch.zeros(batch_size, max_pairs, max_pairs, dtype=torch.bool)
        for index, item in enumerate(batch):
            count = len(item["pair_texts"])
            pair_mask[index, :count] = True
            adj[index, :count, :count] = item["adj"]
        return {
            "sentence_id": [item["sentence_id"] for item in batch],
            "sent_input_ids": sentence_tokens["input_ids"],
            "sent_attention_mask": sentence_tokens["attention_mask"],
            "pair_input_ids": pair_tokens["input_ids"].reshape(batch_size, max_pairs, -1),
            "pair_attention_mask": pair_tokens["attention_mask"].reshape(batch_size, max_pairs, -1),
            "pair_mask": pair_mask,
            "adj": adj,
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        }


class DenseGATLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_source = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_target = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_target)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_size = nodes.shape
        h = self.projection(nodes).reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        source = (h * self.attn_source).sum(dim=-1).permute(0, 2, 1)
        target = (h * self.attn_target).sum(dim=-1).permute(0, 2, 1)
        scores = self.activation(source.unsqueeze(-1) + target.unsqueeze(-2))
        allowed = adj.unsqueeze(1) & mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, -1e4)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        values = h.permute(0, 2, 1, 3)
        aggregated = torch.matmul(attention, values)
        aggregated = aggregated.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, hidden_size)
        output = self.norm(nodes + self.dropout(self.output(aggregated)))
        return output * mask.unsqueeze(-1)


class SymbolicSimCoT(nn.Module):
    """RoBERTa text encoder, two-layer pair GAT, graph fusion, classifier."""

    def __init__(
        self,
        encoder_name_or_path: str,
        class_weights: torch.Tensor | None = None,
        gat_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if gat_layers < 1:
            raise ValueError("gat_layers must be positive")
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.gat = nn.ModuleList(
            [DenseGATLayer(hidden_size, gat_heads, dropout) for _ in range(gat_layers)]
        )
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(LABELS))
        if class_weights is not None and class_weights.shape != (len(LABELS),):
            raise ValueError("class_weights must have one weight per emotion class")
        self.register_buffer("class_weights", class_weights)

    def encode_cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return encoded.last_hidden_state[:, 0]

    def forward(
        self,
        sent_input_ids: torch.Tensor,
        sent_attention_mask: torch.Tensor,
        pair_input_ids: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size, num_pairs, pair_length = pair_input_ids.shape
        sentence = self.encode_cls(sent_input_ids, sent_attention_mask)
        pair_cls = self.encode_cls(
            pair_input_ids.reshape(batch_size * num_pairs, pair_length),
            pair_attention_mask.reshape(batch_size * num_pairs, pair_length),
        ).reshape(batch_size, num_pairs, -1)
        nodes = pair_cls * pair_mask.unsqueeze(-1)
        for layer in self.gat:
            nodes = layer(nodes, adj, pair_mask)
        graph = nodes.sum(dim=1) / pair_mask.sum(dim=1, keepdim=True).clamp_min(1)

        # A softmax over one graph key is always 1. A null alternative makes
        # the attention weight depend on the sentence and graph representations.
        score = (self.query(sentence) * self.key(graph)).sum(dim=-1, keepdim=True)
        score = score / math.sqrt(sentence.shape[-1])
        graph_weight = torch.softmax(torch.cat((torch.zeros_like(score), score), dim=-1), dim=-1)[:, 1:]
        injected = graph_weight * self.value(graph) * pair_mask.any(dim=1, keepdim=True)
        fused = self.fusion(torch.cat((sentence, injected), dim=-1))
        logits = self.classifier(self.dropout(fused))
        loss = F.cross_entropy(logits, labels, weight=self.class_weights) if labels is not None else None
        return {"logits": logits, "loss": loss}
