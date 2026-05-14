from __future__ import annotations

import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from cues import all_cues
from prompting import apply_issue_wording, build_prompt_rows, build_prompt_text, stratified_templates
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


if __name__ == "__main__":
    unittest.main()
