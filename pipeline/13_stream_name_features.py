#!/usr/bin/env python3
"""Stream the full name bank through a model for the cross-cue transfer test.

`08_extract_activations.py` crosses the fixed cue list in `cues.py`, which carries
three names per race x gender group -- twelve in all. The transfer result (RQ3) is
therefore estimated on twelve strings, against the 562 the generation arm rotates,
and cannot separate what the model encodes about a demographic category from what it
encodes about those particular names. This script runs the whole bank.

Storing raw activations for 562 names is not an option: 562 x 19 issues x 5 templates
is 53,390 prompts, which at full residual width is ~116 GB across the three models and
would not fit in RAM before the save. Instead we exploit the fact that the transfer
probe never sees a raw activation. Its pipeline is StandardScaler -> PCA(256) ->
logistic regression, and the scaler and PCA are fit on the *explicit-label* rows
alone. So we:

  pass A   forward the 380 explicit-label prompts, keep their activations, and fit
           one scaler + PCA per layer on them (exactly the transform the probe uses)
  pass B   stream the name prompts, projecting each batch through that fixed
           transform and keeping only the 256 reduced dimensions

which is ~8 GB for all three models instead of 116, fits in memory, and is lossless
with respect to anything the probe can do -- including the shuffled-label control,
which permutes training labels and so leaves the scaler and PCA untouched.

Everything downstream (probe fit, control, bootstraps) then runs on CPU from the
saved features via analysis/06_probe/transfer_namebank.py.

    python pipeline/13_stream_name_features.py --tag llama --device cuda:2 \
        --model /data/resource/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct \
        --out-dir /data/<user>/probe_activations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (DEFAULT_ISSUES_CSV, DEFAULT_NAME_BANK_CSV,
                    DEFAULT_TEMPLATES_ALL_CSV, DEFAULT_WORDING_CSV)
from cues import all_cues, name_cues_from_csv
from hf_utils import apply_chat_template
from io_utils import read_csv, write_csv
from prompting import (apply_issue_wording, build_system_text, fill_template,
                       main_issues, slugify, stable_seed, stratified_templates)

import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location(
    "_extract", _pl.Path(__file__).resolve().parent / "08_extract_activations.py")
_extract = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_extract)
load_model = _extract.load_model

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x=None, **kwargs):
        return x if x is not None else iter(())

PCA_COMPONENTS = 256  # matches analysis/06_probe/train_identity_probe.py
META_COLUMNS = ["row_id", "cue_condition", "cue_family", "cue_group", "cue_value",
                "issue_id", "template_id", "template_rank", "seed"]


def build_rows(n_templates: int) -> tuple[list[dict], list[dict]]:
    """(explicit-label rows, name rows) over the same issue x template grid."""
    issues = apply_issue_wording(
        main_issues(read_csv(DEFAULT_ISSUES_CSV)), read_csv(DEFAULT_WORDING_CSV))
    templates = stratified_templates(read_csv(DEFAULT_TEMPLATES_ALL_CSV), n_templates)
    explicit = [c for c in all_cues() if c.cue_family == "explicit_demographic"]
    names = name_cues_from_csv(DEFAULT_NAME_BANK_CSV)

    def rows_for(cues):
        out = []
        for issue in issues:
            iid = issue.get("ces_variable", "").strip() or slugify(issue["topic_neutral"])
            topic = issue.get("prompt_topic", issue.get("topic_neutral", "")).strip()
            for template in templates:
                tid = template.get("id", "") or f"rank_{template.get('rank')}"
                user_text = fill_template(template["selected_template"].strip(), topic)
                for cue in cues:
                    rid = f"{cue.cue_condition}__{iid}__{tid}"
                    out.append({
                        "row_id": rid, "cue_condition": cue.cue_condition,
                        "cue_family": cue.cue_family, "cue_group": cue.cue_group,
                        "cue_value": cue.cue_value, "issue_id": iid, "template_id": tid,
                        "template_rank": template.get("rank", ""),
                        "seed": str(stable_seed(rid)),
                        "system_text": build_system_text(cue.cue_memory),
                        "user_text": user_text,
                    })
        return out

    return rows_for(explicit), rows_for(names)


def forward_batches(rows, tok, model, torch, dev, batch_size, max_tokens, desc):
    """Yield (batch_index, list of [B, H] arrays, one per layer)."""
    bar = tqdm(total=len(rows), desc=desc, unit="row")
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        formatted = [apply_chat_template(tok, r["user_text"], r["system_text"]) for r in batch]
        inputs = tok(formatted, return_tensors="pt", padding=True, truncation=True,
                     max_length=max_tokens).to(dev)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        # float32 not float16: Gemma-3 residual-stream outliers exceed fp16 max.
        yield [h[:, -1, :].float().cpu().numpy() for h in out.hidden_states]
        if hasattr(bar, "update"):
            bar.update(len(batch))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--templates", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--limit-names", type=int, help="Debug: cap the number of name rows.")
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_rows, name_rows = build_rows(a.templates)
    if a.limit_names:
        name_rows = name_rows[:a.limit_names]
    n_names = len({r["cue_condition"] for r in name_rows})
    print(f"[{a.tag}] explicit rows {len(exp_rows)} | name rows {len(name_rows)} "
          f"({n_names} distinct names)", flush=True)

    tok, model, torch, dev = load_model(a.model, a.device)

    # --- pass A: explicit-label activations, then the per-layer transform ----------
    chunks: dict[int, list] = {}
    for layer_acts in forward_batches(exp_rows, tok, model, torch, dev,
                                      a.batch_size, a.max_input_tokens, f"explicit:{a.tag}"):
        for i, arr in enumerate(layer_acts):
            chunks.setdefault(i, []).append(arr)
    n_layers = len(chunks)
    exp_feats, transforms = {}, {}
    for i in range(n_layers):
        X = np.concatenate(chunks[i], axis=0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        sc = StandardScaler().fit(X)
        pca = PCA(n_components=min(PCA_COMPONENTS, X.shape[0] - 1),
                  random_state=0).fit(sc.transform(X))
        transforms[i] = (sc, pca)
        exp_feats[f"layer_{i}"] = pca.transform(sc.transform(X)).astype(np.float32)
    del chunks
    print(f"[{a.tag}] fit {n_layers} layer transforms on {len(exp_rows)} explicit rows",
          flush=True)

    # --- pass B: stream the name bank through the fixed transform ------------------
    name_chunks: dict[int, list] = {i: [] for i in range(n_layers)}
    for layer_acts in forward_batches(name_rows, tok, model, torch, dev,
                                      a.batch_size, a.max_input_tokens, f"names:{a.tag}"):
        for i, arr in enumerate(layer_acts):
            sc, pca = transforms[i]
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            name_chunks[i].append(pca.transform(sc.transform(arr)).astype(np.float32))
    name_feats = {f"layer_{i}": np.concatenate(name_chunks[i], axis=0)
                  for i in range(n_layers)}

    np.savez(out_dir / f"{a.tag}_explicit_feats.npz", **exp_feats)
    np.savez(out_dir / f"{a.tag}_namebank_feats.npz", **name_feats)
    write_csv(out_dir / f"{a.tag}_explicit_meta.csv", exp_rows, META_COLUMNS)
    write_csv(out_dir / f"{a.tag}_namebank_meta.csv", name_rows, META_COLUMNS)
    shape = name_feats["layer_0"].shape
    print(f"[{a.tag}] saved {n_layers} layers x {shape[0]} name rows x {shape[1]} dims",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
