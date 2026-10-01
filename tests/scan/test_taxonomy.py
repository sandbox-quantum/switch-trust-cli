"""
Tests for taxonomy.py — OWASP ASI taxonomy lookups.
"""

import unittest

from switch_trust.scan.taxonomy import (
    AGENT_TAXONOMY,
    BEYOND_ASI_SENTINEL,
    FLAT_TAXONOMY,
    get_compliance_mappings,
    get_finding_metadata,
    is_owasp_classified,
)


class TestAgentTaxonomy(unittest.TestCase):
    def test_has_10_asi_categories(self):
        for i in range(1, 11):
            key = f"asi{i:02d}"
            matching = [k for k in AGENT_TAXONOMY if k.startswith(key)]
            self.assertGreaterEqual(len(matching), 1, f"Missing ASI category {key}")

    def test_each_category_has_subcategories(self):
        for key, data in AGENT_TAXONOMY.items():
            self.assertIn("subcategories", data, f"{key} missing subcategories")
            self.assertGreater(len(data["subcategories"]), 0)


class TestFlatTaxonomy(unittest.TestCase):
    def test_not_empty(self):
        self.assertGreater(len(FLAT_TAXONOMY), 0)

    def test_entries_have_required_keys(self):
        for _subcat, meta in FLAT_TAXONOMY.items():
            self.assertIn("category", meta)
            self.assertIn("asi_code", meta)


class TestGetFindingMetadata(unittest.TestCase):
    def test_known_subcategory(self):
        meta = get_finding_metadata("direct_prompt_injection")
        self.assertIn("category", meta)
        self.assertIn("cvss_vector", meta)
        self.assertTrue(meta["category"].startswith("asi"))

    def test_unknown_subcategory_returns_defaults(self):
        meta = get_finding_metadata("totally_unknown_xyz")
        self.assertEqual(meta["category"], "beyond_asi")
        self.assertEqual(meta["asi_code"], "BEYOND-ASI")
        self.assertIn("cvss_vector", meta)


class TestGetComplianceMappings(unittest.TestCase):
    def test_known_subcategory(self):
        mappings = get_compliance_mappings("direct_prompt_injection")
        self.assertIn("owasp_agentic", mappings)

    def test_unknown_subcategory_returns_beyond(self):
        mappings = get_compliance_mappings("nonexistent_xyz")
        self.assertEqual(mappings["owasp_agentic"], [BEYOND_ASI_SENTINEL])


class TestIsOwaspClassified(unittest.TestCase):
    def test_classified(self):
        self.assertTrue(is_owasp_classified({"owasp_agentic": ["ASI01-1"]}))

    def test_not_classified(self):
        self.assertFalse(is_owasp_classified({"owasp_agentic": [BEYOND_ASI_SENTINEL]}))

    def test_empty(self):
        self.assertFalse(is_owasp_classified({}))


if __name__ == "__main__":
    unittest.main()
