import unittest

from lm_kbc.parsing import (
    build_unambiguous_alias_maps,
    canonicalize_candidates,
    evaluator_normalize,
    majority_single_with_none,
    median_numeric_candidates,
    parse_json_array,
    support_fraction_candidates,
    threshold_multi_with_none,
)


class ParsingTests(unittest.TestCase):
    def test_official_normalization_handles_apostrophes_and_punctuation(self):
        self.assertEqual(evaluator_normalize("O’Brien"), "obrien")
        self.assertEqual(evaluator_normalize("New-York"), "new york")
        self.assertEqual(
            support_fraction_candidates(
                [["O’Brien"], ["OBrien"], ["O'BRIEN"]], 1.0
            ),
            ["O’Brien"],
        )

    def test_tolerant_parser_cleans_numbers_and_relation_specific_none(self):
        self.assertEqual(
            parse_json_array('answer: ["35,000 people"]', "hasCapacity"),
            ["35000"],
        )
        self.assertEqual(
            parse_json_array('["1.5 million km2"]', "hasArea"), ["1500000"]
        )
        self.assertEqual(
            parse_json_array('["1.23e6 km2"]', "hasArea"), ["1230000"]
        )
        self.assertEqual(
            parse_json_array('["2E+4 seats"]', "hasCapacity"), ["20000"]
        )
        self.assertEqual(
            parse_json_array('["None"]', "personHasCityOfDeath"), []
        )
        self.assertEqual(parse_json_array('["None"]', "awardWonBy"), ["None"])

    def test_numeric_median_averages_the_middle_pair(self):
        self.assertEqual(
            median_numeric_candidates([["100"], ["110"], ["120"], ["130"]]),
            ["115"],
        )

    def test_none_requires_strict_plurality_and_threshold(self):
        self.assertEqual(
            majority_single_with_none([[], ["Paris"]], 0.5), ["Paris"]
        )
        self.assertEqual(
            threshold_multi_with_none([[], [], ["NYSE"]], 0.3, 0.5), []
        )

    def test_train_alias_map_rewrites_only_unambiguous_aliases(self):
        rows = [
            {
                "Relation": "companyTradesAtStockExchange",
                "ObjectEntities": [["New York Stock Exchange", "NYSE"]],
            },
            {
                "Relation": "personHasCityOfDeath",
                "ObjectEntities": [["Springfield, Illinois", "Springfield"]],
            },
            {
                "Relation": "personHasCityOfDeath",
                "ObjectEntities": [["Springfield, Ohio", "Springfield"]],
            },
        ]
        maps = build_unambiguous_alias_maps(rows)
        values, rewrites = canonicalize_candidates(
            ["NYSE", "New York Stock Exchange"],
            maps["companyTradesAtStockExchange"],
        )
        self.assertEqual(values, ["New York Stock Exchange"])
        self.assertEqual(
            rewrites, [{"from": "NYSE", "to": "New York Stock Exchange"}]
        )
        self.assertNotIn(
            evaluator_normalize("Springfield"), maps["personHasCityOfDeath"]
        )


if __name__ == "__main__":
    unittest.main()
