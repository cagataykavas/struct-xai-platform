from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import torch

from structxai.core import Candidate

MAX_CANDIDATES = 64
MAX_CANDIDATE_TOKENS = 128
MAX_LABEL_BYTES = 512


def _validate_inputs(
    prompt_input_ids: torch.Tensor,
    candidates: Sequence[Candidate],
    prompt_attention_mask: torch.Tensor | None,
    *,
    max_candidates: int,
    max_candidate_tokens: int,
) -> None:
    if not 2 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError(f"max_candidates must be between 2 and {MAX_CANDIDATES}")
    if not 1 <= max_candidate_tokens <= MAX_CANDIDATE_TOKENS:
        raise ValueError(f"max_candidate_tokens must be between 1 and {MAX_CANDIDATE_TOKENS}")
    if prompt_input_ids.ndim != 2 or prompt_input_ids.shape[0] != 1:
        raise ValueError("prompt_input_ids must have shape [1, sequence_length]")
    if prompt_input_ids.shape[1] == 0:
        raise ValueError("prompt_input_ids cannot be empty")
    if prompt_input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("prompt_input_ids must contain integer token ids")
    if not 2 <= len(candidates) <= max_candidates:
        raise ValueError(f"candidate count must be between 2 and {max_candidates}")

    labels = [candidate.label for candidate in candidates]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("candidate labels must be non-empty strings")
    if any(len(label.encode("utf-8")) > MAX_LABEL_BYTES for label in labels):
        raise ValueError(f"candidate labels cannot exceed {MAX_LABEL_BYTES} UTF-8 bytes")
    if len(labels) != len(set(labels)):
        raise ValueError("candidate labels must be unique")

    for candidate in candidates:
        if not candidate.token_ids:
            raise ValueError(f"candidate {candidate.label!r} has no tokens")
        if len(candidate.token_ids) > max_candidate_tokens:
            raise ValueError(f"candidate {candidate.label!r} exceeds the {max_candidate_tokens}-token budget")
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in candidate.token_ids
        ):
            raise ValueError(f"candidate {candidate.label!r} contains an invalid token id")

    if prompt_attention_mask is not None:
        if prompt_attention_mask.shape != prompt_input_ids.shape:
            raise ValueError("prompt_attention_mask must match prompt_input_ids")
        if not torch.all(prompt_attention_mask == 1):
            raise ValueError("single-prompt sequence scoring does not accept padded prompt tokens")


def score_candidate_sequences(
    model,
    prompt_input_ids: torch.Tensor,
    candidates: Sequence[Candidate],
    *,
    prompt_attention_mask: torch.Tensor | None = None,
    max_candidates: int = MAX_CANDIDATES,
    max_candidate_tokens: int = MAX_CANDIDATE_TOKENS,
) -> dict[str, object]:
    """Teacher-force complete candidate suffixes against one causal-LM prompt.

    The caller supplies candidate token ids from the same tokenizer contract used
    by the layer-wise first-token metric. Each candidate is appended to the
    prompt and scored token by token. Both total and mean log probability are
    reported because total probability is length-sensitive while mean log
    probability changes the underlying decision rule.

    This is an output-level check. It complements, but does not replace, the
    layer-wise first-token projection used elsewhere in Struct-XAI.
    """
    _validate_inputs(
        prompt_input_ids,
        candidates,
        prompt_attention_mask,
        max_candidates=max_candidates,
        max_candidate_tokens=max_candidate_tokens,
    )

    configured_vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(configured_vocab_size, int):
        for candidate in candidates:
            if max(candidate.token_ids) >= configured_vocab_size:
                raise ValueError(
                    f"candidate {candidate.label!r} contains a token outside the model vocabulary"
                )

    base_mask = (
        prompt_attention_mask if prompt_attention_mask is not None else torch.ones_like(prompt_input_ids)
    )
    prompt_length = prompt_input_ids.shape[1]
    batch_size = len(candidates)
    maximum_suffix_length = max(len(candidate.token_ids) for candidate in candidates)
    suffix_batch = torch.zeros(
        (batch_size, maximum_suffix_length),
        dtype=prompt_input_ids.dtype,
        device=prompt_input_ids.device,
    )
    suffix_mask = torch.zeros_like(suffix_batch)
    for index, candidate in enumerate(candidates):
        length = len(candidate.token_ids)
        suffix_batch[index, :length] = torch.tensor(
            candidate.token_ids,
            dtype=prompt_input_ids.dtype,
            device=prompt_input_ids.device,
        )
        suffix_mask[index, :length] = 1

    input_ids = torch.cat((prompt_input_ids.expand(batch_size, -1), suffix_batch), dim=1)
    attention_mask = torch.cat((base_mask.expand(batch_size, -1), suffix_mask), dim=1)
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        expected_shape = (batch_size, input_ids.shape[1])
        if logits.ndim != 3 or tuple(logits.shape[:2]) != expected_shape:
            raise ValueError("model logits do not align with the scored input batch")

        for index, candidate in enumerate(candidates):
            if logits.shape[2] <= max(candidate.token_ids):
                raise ValueError(
                    f"candidate {candidate.label!r} contains a token outside the model vocabulary"
                )

            prediction_logits = logits[
                index,
                prompt_length - 1 : prompt_length + len(candidate.token_ids) - 1,
                :,
            ].float()
            target_ids = suffix_batch[index, : len(candidate.token_ids)].to(
                device=prediction_logits.device,
                dtype=torch.long,
            )
            token_log_probabilities = (
                torch.log_softmax(prediction_logits, dim=-1)
                .gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1),
                )
                .squeeze(-1)
            )
            values = [float(value) for value in token_log_probabilities.cpu().tolist()]
            if not values or not all(isfinite(value) for value in values):
                raise ValueError(f"model produced non-finite log probabilities for {candidate.label!r}")

            total = sum(values)
            rows.append(
                {
                    "label": candidate.label,
                    "token_count": len(candidate.token_ids),
                    "total_log_probability": total,
                    "mean_log_probability": total / len(values),
                    "first_token_log_probability": values[0],
                    "minimum_token_log_probability": min(values),
                }
            )

    total_ranking = sorted(rows, key=lambda row: (-float(row["total_log_probability"]), str(row["label"])))
    mean_ranking = sorted(rows, key=lambda row: (-float(row["mean_log_probability"]), str(row["label"])))
    first_token_ranking = sorted(
        rows,
        key=lambda row: (-float(row["first_token_log_probability"]), str(row["label"])),
    )
    return {
        "metric": "teacher_forced_candidate_sequence_log_probability",
        "tokenization_contract": "candidate_suffix_tokenized_separately_without_special_tokens",
        "candidates": rows,
        "rankings": {
            "total_log_probability": [str(row["label"]) for row in total_ranking],
            "mean_log_probability": [str(row["label"]) for row in mean_ranking],
            "first_token_log_probability": [str(row["label"]) for row in first_token_ranking],
        },
        "first_token_winner_matches_total_winner": (
            first_token_ranking[0]["label"] == total_ranking[0]["label"]
        ),
        "total_winner_matches_mean_winner": total_ranking[0]["label"] == mean_ranking[0]["label"],
        "limitations": [
            "total log probability favors shorter candidates",
            "mean log probability is a length-normalized diagnostic, not sequence probability",
            "candidate suffix tokenization must match the generation-time contract",
        ],
    }
