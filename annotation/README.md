# Stance-annotation validation set

Human validation of the DeBERTa stance classifier (thesis Appendix A). Two
annotators independently rate the same items; agreement establishes the human
ceiling, and DeBERTa / an LLM-judge are scored against the adjudicated gold.

## What the annotator does

Rate **the stance of the response text** on the stated proposition, on a 0–100
slider: **0 = fully against**, **50 = neutral / balanced**, **100 = fully for**.
Judge only the text — the model, the user cue, and the classifier's own score are
hidden. Flag genuinely unratable items (refusal / off-topic) with the checkbox.

## Sample

- `analysis/07_validation/sample_annotation_set.py` → `out/sample_keys.csv`
  (blinding key: model, cue, `bert_pred_stance`, `pop_weight`). 50/model, split
  across 3 score-bins, prompt_ids unique across the whole sample.
- 150 items now (3 open-weight models). When GPT-5.4 / Claude Sonnet 5 responses
  exist, draw the remaining 100 with:
  `python analysis/07_validation/sample_annotation_set.py --models gpt,claude --start-index 151 --out out/sample_keys_frontier.csv`
- `build_items.py` joins keys → response text (on Brains) → blinded `items.csv`.

## Running the UI (on Brains)

```bash
cd /home/kell8360/stereoissuebench
nohup python3 annotation/app.py --items annotation/items.csv --host 0.0.0.0 --port 8730 \
  >/data/kell8360/annot_app.log 2>&1 &
```

Zero dependencies (stdlib only). Ratings append live to
`annotation/ratings_<name>.csv`; a reopened session resumes where it left off.

## Accessing it (Brains firewalls all ports except SSH)

Each annotator opens an SSH tunnel from their own machine (needs any Brains SSH
login + Oxford VPN), then browses `http://localhost:8731`:

```bash
ssh -N -L 8731:localhost:8730 <your-user>@brains.oii.ox.ac.uk
```

Enter your **initials** at the start screen (use distinct initials per annotator
so the two rating files stay separate). Leave the tunnel open while annotating.

## Collecting results

`~/.claude/skills/brains/scripts/brains.sh get annotation/ratings_<name>.csv`
(or `sync-down`). Gold = mean of the two annotators' 0–100; collapse at 40/60 for
the 3-class agreement; reweight by `pop_weight` for the deployment number.
