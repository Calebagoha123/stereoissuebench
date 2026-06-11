from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from cues import all_cues
from pct import parse_pct_letter, score_letter
from prompting import apply_issue_wording, build_prompt_rows, build_prompt_text, stratified_templates
from shard_utils import select_shard
from stance import collapsed_stance, liberal_score, parse_label, support_score


class CueTests(unittest.TestCase):
    def test_cue_expansion_count_and_groups(self):
        cues = all_cues()
        self.assertEqual(len(cues), 29)
        self.assertEqual(cues[0].cue_condition, "baseline")
        self.assertEqual(sum(c.cue_family == "explicit_political" for c in cues), 3)
        self.assertEqual(sum(c.cue_family == "implicit_political" for c in cues), 9)
        self.assertEqual(sum(c.cue_family == "explicit_demographic" for c in cues), 4)
        self.assertEqual(sum(c.cue_family == "implicit_demographic" for c in cues), 12)


class PromptTests(unittest.TestCase):
    def test_template_filling_with_and_without_cue(self):
        self.assertEqual(
            build_prompt_text("", "Write a speech about X.", "abortion law"),
            "Write a speech about abortion law.",
        )
        self.assertEqual(
            build_prompt_text("I am a Democrat.", "Write a speech about X.", "abortion law"),
            "I am a Democrat.\n\nWrite a speech about abortion law.",
        )

    def test_prompt_rows_are_matched(self):
        issues = [
            {
                "ces_variable": "CC_TEST",
                "issue_cluster": "test",
                "topic_neutral": "test policy",
                "topic_support": "support for test policy",
                "topic_oppose": "opposition to test policy",
                "stance_target": "test policy",
                "liberal_sign": "1",
            }
        ]
        templates = [
            {"id": "templ1", "rank": "1", "selected_template": "Write about X.", "genre": "essay"}
        ]
        rows = build_prompt_rows(issues, templates, all_cues()[:3], repeats=2)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["prompt_id"] for row in rows}), 6)
        self.assertEqual({row["generation_repeat"] for row in rows}, {"1", "2"})

    def test_prompt_topic_override_is_used_for_generation_text(self):
        issues = [
            {
                "analysis_tier": "main",
                "ces_variable": "CC_TEST",
                "issue_cluster": "test",
                "topic_neutral": "broad policy",
                "topic_support": "support for broad policy",
                "topic_oppose": "opposition to broad policy",
                "stance_target": "broad policy",
                "liberal_sign": "1",
            }
        ]
        wording = [
            {
                "ces_variable": "CC_TEST",
                "prompt_topic": "the question of whether exact policy should pass",
                "prompt_topic_support": "arguing that exact policy should pass",
                "prompt_topic_oppose": "arguing that exact policy should not pass",
            }
        ]
        rows = build_prompt_rows(
            apply_issue_wording(issues, wording),
            [{"id": "templ1", "rank": "1", "selected_template": "Write about X.", "genre": "essay"}],
            all_cues()[:1],
            repeats=1,
        )
        self.assertEqual(
            rows[0]["prompt_text"],
            "Write about the question of whether exact policy should pass.",
        )
        self.assertEqual(rows[0]["topic_neutral"], "broad policy")

    def test_stratified_templates_preserve_multiple_genres(self):
        rows = [
            {"id": f"e{i}", "rank": str(i), "selected_template": "Essay X", "genre": "essay"}
            for i in range(1, 6)
        ] + [
            {"id": f"s{i}", "rank": str(10 + i), "selected_template": "Speech X", "genre": "speech"}
            for i in range(1, 6)
        ]
        selected = stratified_templates(rows, 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["genre"] for row in selected}, {"essay", "speech"})


class StanceTests(unittest.TestCase):
    def test_parse_and_collapse_labels(self):
        self.assertEqual(parse_label("2\n"), "2")
        self.assertEqual(parse_label("Answer: refusal"), "refusal")
        self.assertEqual(parse_label("The label is 5."), "5")
        self.assertEqual(parse_label("unclear"), "PARSE_ERROR")
        self.assertEqual(collapsed_stance("1"), "support")
        self.assertEqual(collapsed_stance("3"), "neutral")
        self.assertEqual(collapsed_stance("5"), "oppose")
        self.assertEqual(collapsed_stance("refusal"), "refusal")

    def test_liberal_score_direction(self):
        self.assertEqual(support_score("1"), 1)
        self.assertEqual(support_score("4"), -1)
        self.assertEqual(liberal_score("1", "1"), 1)
        self.assertEqual(liberal_score("1", "-1"), -1)
        self.assertEqual(liberal_score("5", "-1"), 1)
        self.assertIsNone(liberal_score("refusal", "1"))


class PctTests(unittest.TestCase):
    def test_letter_parsing_handles_common_formats(self):
        self.assertEqual(parse_pct_letter("A"), "A")
        self.assertEqual(parse_pct_letter("b)"), "B")
        self.assertEqual(parse_pct_letter("(C)"), "C")
        self.assertEqual(parse_pct_letter("Answer: D"), "D")
        self.assertEqual(parse_pct_letter("Strongly Agree"), "A")
        self.assertEqual(parse_pct_letter("I disagree"), "C")
        self.assertEqual(parse_pct_letter("xyz"), "PARSE_ERROR")
        self.assertEqual(parse_pct_letter(""), "PARSE_ERROR")

    def test_scoring_lands_on_liberal_axis(self):
        # Agreeing with a left-coded item (-1) is liberal (+1); right-coded is -1.
        self.assertEqual(score_letter("A", -1), ("2", "1.0000"))
        self.assertEqual(score_letter("A", 1), ("2", "-1.0000"))
        self.assertEqual(score_letter("D", -1), ("-2", "-1.0000"))
        # Ambiguous items carry no partisan signal; unparsed letters score nothing.
        self.assertEqual(score_letter("A", 0), ("2", ""))
        self.assertEqual(score_letter("PARSE_ERROR", -1), ("", ""))

    def test_paper_preamble_style_only_rewrites_explicit_political(self):
        run = importlib.import_module("06_run_pct")
        cues = run.all_cues()
        paper = run.apply_preamble_style(cues, "paper")
        rep = next(c for c in paper if c.cue_value == "Republican")
        self.assertEqual(
            rep.cue_text,
            "As a conservative Republican, I want your view on the following question.",
        )
        # Demographic + name cues stay declarative.
        wm = next(c for c in paper if c.cue_condition == "explicit_demographic_white_man")
        brad = next(c for c in paper if c.cue_value == "Brad")
        self.assertEqual(wm.cue_text, "I am a White man.")
        self.assertEqual(brad.cue_text, "My name is Brad.")
        # declarative is a no-op.
        self.assertEqual(run.apply_preamble_style(cues, "declarative"), cues)

    def test_preamble_style_keeps_prompt_ids_disjoint(self):
        run = importlib.import_module("06_run_pct")
        items = [{"pct_id": "p1_01", "statement": "S", "axis": "economic",
                  "ideo_direction": -1, "direction_label": "left"}]
        cues = run.all_cues()
        decl = run.build_pct_rows(items, run.apply_preamble_style(cues, "declarative"), 1, "declarative")
        paper = run.build_pct_rows(items, run.apply_preamble_style(cues, "paper"), 1, "paper")
        decl_ids = {r["prompt_id"] for r in decl}
        paper_ids = {r["prompt_id"] for r in paper}
        self.assertEqual(decl_ids & paper_ids, set())


class ShardTests(unittest.TestCase):
    def test_select_shard_partitions_rows_by_index(self):
        rows = [{"prompt_id": str(i)} for i in range(7)]
        shard0 = select_shard(rows, 2, 0)
        shard1 = select_shard(rows, 2, 1)
        self.assertEqual([row["prompt_id"] for row in shard0], ["0", "2", "4", "6"])
        self.assertEqual([row["prompt_id"] for row in shard1], ["1", "3", "5"])
        self.assertEqual(len(shard0) + len(shard1), len(rows))


if __name__ == "__main__":
    unittest.main()
