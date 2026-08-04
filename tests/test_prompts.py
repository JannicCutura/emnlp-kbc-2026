from __future__ import annotations

import unittest

from lm_kbc.prompts import select_examples


def rows(relation: str, empty: int, non_empty: int) -> list[dict]:
    return [
        {
            "SubjectEntity": f"empty-{index}",
            "Relation": relation,
            "ObjectEntities": [],
        }
        for index in range(empty)
    ] + [
        {
            "SubjectEntity": f"filled-{index}",
            "Relation": relation,
            "ObjectEntities": [["Object"]],
        }
        for index in range(non_empty)
    ]


def mix(selected: list[dict]) -> str:
    return "".join("E" if not row["ObjectEntities"] else "N" for row in selected)


class DemonstrationBalanceTests(unittest.TestCase):
    """Demonstration label mix must track the relation's training distribution.

    Over-representing empty answers biases the model toward abstention, which is
    the failure mode that hard null gates already produced.
    """

    def test_demonstration_mix_follows_the_training_empty_rate(self):
        for empty, non_empty, expected_empty in (
            (42, 58, 2),   # personHasCityOfDeath
            (34, 66, 2),   # companyTradesAtStockExchange
            (12, 55, 1),   # countryLandBordersCountry
            (0, 100, 0),   # hasArea / hasCapacity
        ):
            pool = rows("r", empty, non_empty)
            with self.subTest(empty=empty):
                selected = select_examples(
                    pool, "r", 5, exclude_subject="absent", offset=0
                )
                self.assertEqual(len(selected), 5)
                self.assertEqual(mix(selected).count("E"), expected_empty)

    def test_empty_demonstrations_are_spaced_not_blocked(self):
        selected = select_examples(
            rows("r", 42, 58), "r", 5, exclude_subject="absent", offset=0
        )
        self.assertEqual(mix(selected), "NENNE")

    def test_selection_is_stable_across_rotations_and_has_no_repeat_subject(self):
        pool = rows("r", 12, 55)
        for offset in (0, 5, 10, 25, 133):
            with self.subTest(offset=offset):
                selected = select_examples(
                    pool, "r", 5, exclude_subject="absent", offset=offset
                )
                self.assertEqual(mix(selected).count("E"), 1)
                subjects = [row["SubjectEntity"] for row in selected]
                self.assertEqual(len(set(subjects)), 5)

    def test_rotation_actually_varies_the_chosen_examples(self):
        pool = rows("r", 12, 55)
        first = select_examples(pool, "r", 5, exclude_subject="absent", offset=0)
        later = select_examples(pool, "r", 5, exclude_subject="absent", offset=7)
        self.assertNotEqual(
            [row["SubjectEntity"] for row in first],
            [row["SubjectEntity"] for row in later],
        )

    def test_target_subject_is_excluded_from_its_own_demonstrations(self):
        pool = rows("r", 12, 55)
        selected = select_examples(
            pool, "r", 5, exclude_subject="filled-0", offset=0
        )
        self.assertNotIn("filled-0", [row["SubjectEntity"] for row in selected])

    def test_short_class_is_backfilled_from_the_other(self):
        # One empty row available but the rate asks for two.
        selected = select_examples(
            rows("r", 1, 9), "r", 5, exclude_subject="absent", offset=0
        )
        self.assertEqual(len(selected), 5)
        self.assertLessEqual(mix(selected).count("E"), 1)

    def test_synthetic_cot_rows_are_preferred_over_plain_train_rows(self):
        pool = [
            {
                "SubjectEntity": "filled-0",
                "Relation": "r",
                "ObjectEntities": [["Object"]],
                "Rationale": "a verified reasoning path",
            }
        ] + rows("r", 0, 5)
        selected = select_examples(
            pool, "r", 2, exclude_subject="absent", offset=0
        )
        self.assertEqual(selected[0]["Rationale"], "a verified reasoning path")
        # The duplicated subject must not also appear as its plain train copy.
        subjects = [row["SubjectEntity"] for row in selected]
        self.assertEqual(len(set(subjects)), len(subjects))

    def test_degenerate_inputs_return_no_examples(self):
        self.assertEqual(
            select_examples(rows("r", 5, 5), "r", 0, exclude_subject="a", offset=0),
            [],
        )
        self.assertEqual(
            select_examples([], "r", 5, exclude_subject="a", offset=0), []
        )


if __name__ == "__main__":
    unittest.main()
