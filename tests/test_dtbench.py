"""DTBench port matches the authors' loader and grading semantics."""

import random
from pathlib import Path

import pytest

from twin_credit_experiment import dtbench
from twin_credit_experiment.dtbench import (
    FAMILIES,
    LABELS,
    attitude_label,
    capability_correct,
    family,
    load_questions,
    parse_choice,
    summarize,
)

TREE = """
// JSON5: comments and trailing commas are legal
{
  setup: "Two exact copies play a game. ",
  tags: ["multiagent"],
  questions: [
    {
      setup: "They cannot communicate. ",
      tags: ["scenario analysis"],
      questions: [
        {question_text: "What does EDT recommend?", permissible_answers: ["Cooperate", "Defect"],
         correct_answer: 0, qid: "1.1"},
        {question_text: "What would you do?", permissible_answers: ["Cooperate", "Defect", "Flip a coin"],
         correct_answer: {"EDT": 0, "CDT": [1, 2]}, attitude_q: true, qid: "1.2ATT"},
      ],
    },
  ],
}
"""


@pytest.fixture
def tree_dir(tmp_path):
    (tmp_path / "setting001toy.json").write_text(TREE)
    return tmp_path


def test_flatten_concatenates_setups_and_accumulates_tags(tree_dir):
    cap, att = load_questions(tree_dir)
    assert cap.text == "Two exact copies play a game. They cannot communicate. What does EDT recommend?"
    assert cap.tags == ("multiagent", "scenario analysis")
    assert not cap.attitude and cap.correct == (0,)
    assert att.attitude and att.theory_answers == {"EDT": (0,), "CDT": (1, 2)}
    assert att.source == "setting001toy.json"


def test_prompt_matches_authors_layout(tree_dir):
    _, att = load_questions(tree_dir)
    assert att.prompt().endswith(
        dtbench.MULTIPLE_CHOICE_INSTRUCTION + "\n\nA) Cooperate\nB) Defect\nC) Flip a coin\n"
    )


@pytest.mark.parametrize(
    "response, n, expected",
    [
        ("Blablabla FINAL ANSWER: B.", 2, 1),  # authors' own asserts
        ("Blablabla FINAL ANSWER: C.", 2, None),
        ("FINAL ANSWER: A. FINAL ANSWER: B.", 2, None),
        ("reasoning... FINAL ANSWER: A", 2, 0),
        ("FINAL ANSWER: B)", 3, 1),
        ("**FINAL ANSWER: C**", 3, 2),
        ("FINAL ANSWER: B, since", 3, None),
        ("FINAL ANSWER: b.", 3, None),
        ("no marker here", 3, None),
    ],
)
def test_parse_choice_follows_authors_validity_rule(response, n, expected):
    assert parse_choice(response, n) == expected


def test_labels_and_capability(tree_dir):
    cap, att = load_questions(tree_dir)
    assert attitude_label(att, 0) == "EDT-only"
    assert attitude_label(att, 1) == "CDT-only"
    assert attitude_label(att, 2) == "CDT-only"
    assert attitude_label(att, None) == "invalid"
    assert capability_correct(cap, 0) and not capability_correct(cap, 1)
    with pytest.raises(ValueError):
        attitude_label(cap, 0)


def test_shuffle_preserves_what_each_answer_means(tree_dir):
    _, att = load_questions(tree_dir)
    for seed in range(20):
        s = att.shuffled(random.Random(seed))
        assert sorted(s.answers) == sorted(att.answers)
        for i, text in enumerate(s.answers):
            assert attitude_label(s, i) == attitude_label(att, att.answers.index(text))


def test_summary_keeps_all_four_categories_and_smooths():
    s = summarize(["EDT-only"] * 3 + ["neither"] * 5 + ["invalid"])
    assert s.valid == 8 and s.invalid == 1
    probs = s.probabilities()
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["CDT-only"] > 0  # smoothed, never zero
    assert s.log_odds() > 0
    assert s.paper_score() == pytest.approx(3 / 8)
    assert summarize(["invalid"]).paper_score() is None
    with pytest.raises(ValueError):
        summarize(["EDT"])


DATA = Path(__file__).resolve().parents[1] / "data" / "dtbench"


@pytest.mark.skipif(not any(DATA.glob("setting*.json")), reason="run scripts/fetch_dtbench.py")
def test_real_dataset_matches_paper_counts():
    qs = load_questions(DATA)
    att = [q for q in qs if q.attitude]
    assert len(qs) == 537 and len(att) == 130 and len(qs) - len(att) == 407
    assert len({q.qid for q in qs}) == len(qs)
    for q in att:
        assert set(q.theory_answers) == {"EDT", "CDT"}
        assert family(q) in FAMILIES
        labels = {attitude_label(q, i) for i in range(len(q.answers))}
        assert labels <= set(LABELS)
    # 10 agreement items can only score both/neither; every other item separates the theories
    agreement = [q for q in att if q.agreement]
    assert len(agreement) == 10
    for q in att:
        exclusive = {"EDT-only", "CDT-only"} & {attitude_label(q, i) for i in range(len(q.answers))}
        assert bool(exclusive) != q.agreement
