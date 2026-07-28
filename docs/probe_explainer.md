# What the probes did, in plain English

This is the intuition behind the probe study — no jargon. For the numbers and
methods see [`probe_findings.md`](probe_findings.md) and [`methodology.md`](methodology.md).

## The puzzle we were trying to explain

In the main experiment we plant a fact about the "user" in the model's memory and
then ask it for help writing about a political issue. We already knew:

- If we tell it **outright** — "the user is a Republican" — the writing it produces
  clearly shifts to the conservative side.
- If we only **hint** — give the user a name like "Jamal", or say "I live in
  Texas" — the writing barely changes at all.

The obvious question: **why doesn't the hint do anything?** There are three very
different explanations, and they matter for completely different reasons:

1. **It doesn't notice.** The model can't tell anything about you from a name or a
   state, so there's nothing to act on.
2. **It notices but doesn't connect the dots.** It figures out who you probably
   are, but doesn't link that to a political opinion.
3. **It notices, connects the dots, but holds back.** It forms a view of what you
   likely think — and then declines to write to it.

You can't tell these apart just by reading the model's essays. So we asked the
model directly, and we looked inside it.

## What we did

**Two kinds of probe.**

- **Behavioral (we just ask it).** Two questions, over the same 19 issues and the
  same identity hints as the main run:
  - *"How likely is this user to support policy X?"* — i.e. what does the model
    *believe* you think, before it writes anything?
  - *"How useful is knowing someone's party / race / gender / state / first name
    for guessing their opinion?"* — i.e. which clues does the model itself think
    are worth paying attention to?
- **Internal (we look inside).** A language model turns text into a long list of
  numbers ("activations") as it reads. We can train a tiny, simple detector on
  those numbers to check whether a particular fact — "this user is a Black woman",
  "this user is conservative" — is *written down inside the model*, even if it
  never says so out loud. (This only works on the open models we can open up:
  Llama, Gemma, Qwen. We can't see inside GPT.)

The clever bit of the internal test: we teach the detector what "Black woman"
looks like inside the model using the **explicit** cases ("I am a Black woman"),
then check whether it also lights up for the **name** cases ("My name is
Aaliyah"). If it does, the model is quietly reading the name as a demographic
signal on its own.

## What we found (the same story in all three models)

1. **The model does read you — even from a name.** Inside the model, "My name is
   Aaliyah" lights up the same "Black woman" pattern that the explicit label does
   (well above chance). It also clearly registers that a Texan leans conservative.
   So explanation #1 — "it doesn't notice" — is **wrong**. The information is in
   there.

2. **When it forms an opinion about you, it writes to match it.** Across all the
   identity hints, how strongly the model *believes* you lean one way lines up
   tightly (correlation ~0.8) with how its *writing* leans. And we can see the
   same thing inside: a single internal "left↔right" dial moves, and the writing
   moves with it. So the machinery is: *form an impression of the user → write
   toward it.*

3. **But it under-reacts, and the hints barely move the dial.**
   - From a **name**, the model forms only a *faint* impression — and, tellingly,
     when we ask it outright it rates a first name as **almost useless** (about
     2–4 out of 100) for guessing someone's politics, far below race or party.
   - From a **state**, it forms a *strong* impression ("a Texan is probably
     conservative") — and then **doesn't act on it**: the writing stays put.

So the answer to the puzzle is mostly **explanation #3**: the model isn't blind to
subtle identity cues — it picks them up internally — it just **doesn't carry them
through to what it writes.** Names fail because the model treats them as
uninformative; states fail because a strong hunch simply isn't used.

## Why this matters

- **It's not an accuracy problem; it's a choice (or a default).** "The model can't
  tell race from a name" would be a simple capability gap. Instead the model *can*
  tell, and largely declines to use it in its output. That's a more interesting —
  and more controllable — finding.
- **Two faces for fairness.** On the surface this looks reassuring: the model
  doesn't visibly slant its writing based on your name. But the demographic read
  is **present inside the model**, so the slant is latent — it could surface in
  other behaviours, or be amplified by a small nudge. (Our optional next step,
  the "causal test", checks exactly that: if we gently push that internal
  left↔right dial ourselves, does the writing swing? If yes, the dial is a real
  lever, not just a coincidence.)
- **It sharpens the headline of the thesis.** The main result ("implicit cues
  don't move stance") could have been dismissed as the model being oblivious.
  These probes show it's *aware but restrained* — a much stronger and more
  defensible claim, and one that replicates across three different model families.

## One-line version

> The model quietly figures out who you are — even from a first name — and when it
> forms an opinion about you it writes to match. It just forms only a faint
> impression from a name, and doesn't act on the strong impression it forms from
> your location. The silence in its writing is restraint, not blindness.
