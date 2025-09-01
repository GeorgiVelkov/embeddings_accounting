# api/ml.py
import os, re, threading
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
from django.conf import settings
from sentence_transformers import SentenceTransformer

# Optional (lazy) import for reranker
_xnli_pipeline = None

# Global state (per process/worker)
_lock = threading.Lock()
_loaded = False
_names: List[str] = []
_descs: List[str] = []
_label_embs: np.ndarray = None
_label_hyps: List[str] = []
_e5 = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
E5_MODEL = os.getenv("E5_MODEL", "intfloat/multilingual-e5-base")
XNLI_MODEL = os.getenv("XNLI_MODEL", "joeddav/xlm-roberta-large-xnli")
HYPOTHESIS_TEMPLATE = os.getenv("HYP_TEMPLATE", "Този текст е за {}.")

def _first_sentences(text: str, max_sents=2, max_chars=220) -> str:
    parts = re.split(r"(?<=[\.\!\?])\s+", (text or "").strip())
    out = " ".join([p.strip() for p in parts[:max_sents] if p.strip()])
    return (out[:max_chars].rstrip()+"…") if len(out) > max_chars else out

def _pick_col(df: pd.DataFrame, candidates):
    for c in df.columns:
        if c.strip().lower() in candidates:
            return c
    return None

def _load_labels_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    name_col = _pick_col(df, {"name","label","category","име"}) or df.columns[0]
    desc_col = _pick_col(df, {"description","desc","описание"}) or df.columns[1]
    names = df[name_col].astype(str).tolist()
    descs = df[desc_col].fillna("").astype(str).tolist()
    return names, descs

def ensure_loaded(force: bool=False):
    global _loaded, _e5, _names, _descs, _label_embs, _label_hyps
    if _loaded and not force:
        return
    with _lock:
        if _loaded and not force:
            return
        csv_path = os.getenv("LABELS_CSV", getattr(settings, "LABELS_CSV", "labels_name_description_bg.csv"))
        _names, _descs = _load_labels_csv(csv_path)
        _e5 = SentenceTransformer(E5_MODEL, device=DEVICE)
        passages = [f"passage: {n}. {d}".strip() for n, d in zip(_names, _descs)]
        _label_embs = _e5.encode(passages, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        _label_embs = np.asarray(_label_embs, dtype=np.float32)
        _label_hyps = [f"{n}: {_first_sentences(d)}" if d else n for n, d in zip(_names, _descs)]
        _loaded = True

def ensure_xnli():
    global _xnli_pipeline
    if _xnli_pipeline is not None:
        return _xnli_pipeline
    from transformers import pipeline
    _xnli_pipeline = pipeline(
        "zero-shot-classification",
        model=XNLI_MODEL,
        device=0 if DEVICE == "cuda" else -1,
    )
    return _xnli_pipeline

def _cosine_topk(q_embs: np.ndarray, k: int):
    S = np.matmul(q_embs, _label_embs.T)  # embeddings are L2-normalized
    order = np.argsort(-S, axis=1)[:, :k]
    return S, order

def predict(texts: List[str], top_k=5, rerank=False, rerank_top=15) -> List[Dict[str, Any]]:
    ensure_loaded()
    q = [f"query: {t}" for t in texts]
    q_embs = _e5.encode(q, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    S, idx = _cosine_topk(q_embs, max(top_k, rerank_top if rerank else top_k))

    results = []
    if not rerank:
        for i, text in enumerate(texts):
            order = idx[i][:top_k]
            items = [{
                "label": _names[j],
                "score": float(S[i,j]),        # cosine
                "cosine": float(S[i,j]),
                "description": _descs[j],
            } for j in order]
            results.append({"query": text, "results": items})
        return results

    # Rerank with XNLI on top candidates
    xnli = ensure_xnli()
    for i, text in enumerate(texts):
        cand_ids = idx[i].tolist()
        cand_names = [_names[j] for j in cand_ids]
        cand_hyps  = [_label_hyps[j] for j in cand_ids]
        z = xnli(sequences=text, candidate_labels=cand_hyps, multi_label=True, hypothesis_template=HYPOTHESIS_TEMPLATE)
        lab2p = {lab: sc for lab, sc in zip(z["labels"], z["scores"])}  # P(entailment)
        ent = np.array([lab2p[h] for h in cand_hyps], dtype=np.float32)
        cos = S[i, cand_ids].astype(np.float32)
        comb = np.sqrt(np.clip(cos,1e-6,1.0) * np.clip(ent,1e-6,1.0))  # geometric mean
        order = np.argsort(-comb)[:top_k]
        items = [{
            "label": cand_names[j],
            "score": float(comb[j]),
            "cosine": float(cos[j]),
            "xnli": float(ent[j]),
            "description": _descs[cand_ids[j]],
        } for j in order]
        results.append({"query": text, "results": items})
    return results

def reload_labels():
    ensure_loaded(force=True)
