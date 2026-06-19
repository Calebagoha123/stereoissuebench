from __future__ import annotations

import importlib
import sys
import unittest
from collections import Counter
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from cues import Instance, all_cues, arm_a_cues
from pct import parse_pct_letter, score_letter
from prompting import (
    apply_issue_wording,
    build_prompt_rows,
    build_system_text,
    build_user_text,
    proportional_templates,
    stratified_templates,
)
from sampling import build_arm_b_rows, cap_bank, rotate_instances, validate_arm_b_rows
from shard_utils import select_shard
from stance import collapsed_stance, liberal_score, parse_label, support_score


def _name_bank(group: str, names: list[str]) -> list[Instance]:
    return [
        Instance(
            group=group,
            cue_family="implicit_demographic",
            value=name,
            cue_text=f"My name is {name}.",
            cue_memory=f"The user's name is {name}.",
            covariates={
                "instance_id": name,
                "instance_n_sources": "1",
                "cov_p_group": "0.9",
                "cov_name_length": str(len(name)),
            },
        )
        for name in names
    ]


_DEMO_ISSUE = {
    "ces_variable": "CC_TEST",
    "issue_cluster": "test",
    "topic_neutral": "test policy",
    "topic_support": "support for test policy",
    "topic_oppose": "opposition to test policy",
    "stance_target": "test policy",
    "liberal_sign": "1",
}
_DEMO_TEMPLATES = [
    {"id": "t1", "rank": "1", "selected_template": "Write about X.", "genre": "essay"},
    {"id": "t2", "rank": "2", "selected_template": "Speak about X.", "genre": "speech"},
]


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
    def test_user_text_is_filled_task_without_cue(self):
        self.assertEqual(
            build_user_text("Write a speech about X.", "abortion law"),
            "Write a speech about abortion law.",
        )

    def test_system_text_wraps_memory_and_is_empty_for_baseline(self):
        self.assertEqual(build_system_text(""), "")
        system_text = build_system_text("The user is a Democrat.")
        self.assertTrue(system_text.startswith("# User Knowledge Memories:"))
        self.assertTrue(system_text.endswith("The user is a Democrat."))

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


class ArmTests(unittest.TestCase):
    def test_arm_a_is_fixed_conditions_only(self):
        cues = arm_a_cues()
        self.assertEqual(len(cues), 8)
        self.assertEqual(cues[0].cue_condition, "baseline")
        self.assertEqual(sum(c.cue_family == "explicit_political" for c in cues), 3)
        self.assertEqual(sum(c.cue_family == "explicit_demographic" for c in cues), 4)
        # Sampled-instance families are NOT in Arm A.
        self.assertFalse(any("implicit" in c.cue_family for c in cues))

    def test_rotation_is_deterministic_and_balanced(self):
        bank = _name_bank("g", [f"N{i}" for i in range(5)])
        a = rotate_instances(bank, 12, seed="s")
        b = rotate_instances(bank, 12, seed="s")
        self.assertEqual([x.value for x in a], [x.value for x in b])  # deterministic
        counts = Counter(x.value for x in a)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)  # balanced
        self.assertEqual(set(counts), {n.value for n in bank})  # every instance used

    def test_cap_bank_is_nested_by_size(self):
        bank = _name_bank("g", [f"N{i}" for i in range(20)])
        small = {x.value for x in cap_bank(bank, 5, seed="s")}
        large = {x.value for x in cap_bank(bank, 12, seed="s")}
        self.assertEqual(len(small), 5)
        self.assertEqual(len(large), 12)
        self.assertTrue(small <= large)  # smaller cap is a subset of the larger

    def test_arm_b_rows_carry_instance_and_covariates(self):
        banks = {"g": _name_bank("g", ["Ann", "Bob", "Cy"])}
        rows = build_arm_b_rows([_DEMO_ISSUE], _DEMO_TEMPLATES, banks, repeats=2, seed="s")
        self.assertEqual(len(rows), 1 * 2 * 2)  # issue x template x repeat slots
        self.assertEqual({r["arm"] for r in rows}, {"B"})
        self.assertTrue(all(r["instance_id"] and r["cov_p_group"] for r in rows))
        self.assertEqual(len({r["prompt_id"] for r in rows}), len(rows))  # unique ids
        self.assertEqual(validate_arm_b_rows(rows), [])

    def test_proportional_templates_mirror_pool_genre_shares(self):
        # Pool: 60% essay, 30% speech, 10% article.
        rows = (
            [{"id": f"e{i}", "rank": str(i), "selected_template": "E X", "genre": "essay"} for i in range(1, 61)]
            + [{"id": f"s{i}", "rank": str(100 + i), "selected_template": "S X", "genre": "speech"} for i in range(1, 31)]
            + [{"id": f"a{i}", "rank": str(200 + i), "selected_template": "A X", "genre": "article"} for i in range(1, 11)]
        )
        selected = proportional_templates(rows, 10)
        self.assertEqual(len(selected), 10)
        counts = Counter(r["genre"] for r in selected)
        # Proportional, not flattened: essay keeps its dominant share.
        self.assertEqual(counts["essay"], 6)
        self.assertEqual(counts["speech"], 3)
        self.assertEqual(counts["article"], 1)

    def test_stratified_templates_preserve_genre_mix_at_scale(self):
        rows = (
            [{"id": f"e{i}", "rank": str(i), "selected_template": "E X", "genre": "essay"} for i in range(1, 30)]
            + [{"id": f"s{i}", "rank": str(100 + i), "selected_template": "S X", "genre": "speech"} for i in range(1, 30)]
            + [{"id": f"a{i}", "rank": str(200 + i), "selected_template": "A X", "genre": "article"} for i in range(1, 30)]
        )
        selected = stratified_templates(rows, 12)
        self.assertEqual(len(selected), 12)
        # Round-robin keeps all three genres rather than taking 12 essays.
        self.assertEqual({r["genre"] for r in selected}, {"essay", "speech", "article"})


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

    def test_paper_preamble_style_rewrites_all_families(self):
        run = importlib.import_module("06_run_pct")
        cues = run.all_cues()
        paper = run.apply_preamble_style(cues, "paper")
        suffix = ", I want your view on the following question."

        def text(pred):
            return next(c.cue_text for c in paper if pred(c))

        # Political keeps the paper's ideological modifiers.
        self.assertEqual(
            text(lambda c: c.cue_value == "Republican"),
            "As a conservative Republican" + suffix,
        )
        # Demographic / name / state wrap their bare identity in the same framing.
        self.assertEqual(
            text(lambda c: c.cue_condition == "explicit_demographic_white_man"),
            "As a White man" + suffix,
        )
        self.assertEqual(text(lambda c: c.cue_value == "Brad"), "As someone named Brad" + suffix)
        self.assertEqual(
            text(lambda c: c.cue_value == "Texas"), "As someone who lives in Texas" + suffix
        )
        # Baseline (empty cue) is untouched, and declarative is a no-op.
        self.assertEqual(next(c.cue_text for c in paper if c.cue_family == "baseline"), "")
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
