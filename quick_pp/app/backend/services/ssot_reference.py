"""Petrophysical Single-Source-of-Truth (SSOT) Reference Service.

Ingests and serves grounded petrophysical equations, workflow contracts, and diagnostic
guidelines directly from the version-controlled documentation in `docs/workflows/`
(`single_well.md`, `multi_well.md`, and `index.md`).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from quick_pp import logger

_DOCS_WORKFLOWS_DIR = Path(__file__).resolve().parents[4] / "docs" / "workflows"

# Step aliases to normalize step IDs between frontend state and markdown anchors
STEP_ALIASES: Dict[str, Dict[str, str]] = {
    "single-well": {
        "overview": "sw-level-1-2-diagrams",
        "diagrams": "sw-level-1-2-diagrams",
        "qaqc": "sw-step-1-qaqc",
        "1": "sw-step-1-qaqc",
        "ancillary": "sw-step-2-ancillary",
        "trajectory": "sw-step-2-ancillary",
        "tops": "sw-step-2-ancillary",
        "2": "sw-step-2-ancillary",
        "litho-poro": "sw-step-3-litho-poro",
        "litho_poro": "sw-step-3-litho-poro",
        "lithology": "sw-step-3-litho-poro",
        "porosity": "sw-step-3-litho-poro",
        "vsh": "sw-step-3-litho-poro",
        "vshale": "sw-step-3-litho-poro",
        "3": "sw-step-3-litho-poro",
        "perm": "sw-step-4-perm",
        "permeability": "sw-step-4-perm",
        "timur": "sw-step-4-perm",
        "coates": "sw-step-4-perm",
        "4": "sw-step-4-perm",
        "saturation": "sw-step-5-saturation",
        "sat": "sw-step-5-saturation",
        "archie": "sw-step-5-saturation",
        "waxman-smits": "sw-step-5-saturation",
        "dual-water": "sw-step-5-saturation",
        "5": "sw-step-5-saturation",
        "ressum": "sw-step-6-ressum",
        "summary": "sw-step-6-ressum",
        "blocking": "sw-step-6-ressum",
        "upscaling": "sw-step-6-ressum",
        "netpay": "sw-step-6-ressum",
        "6": "sw-step-6-ressum",
        "report": "sw-step-7-report",
        "cpi": "sw-step-7-report",
        "7": "sw-step-7-report",
        # Standalone Lab Module
        "scal": "sw-standalone-core-pc",
        "core": "sw-standalone-core-pc",
        "pc": "sw-standalone-core-pc",
        "shift": "sw-standalone-core-pc",
        "depth-shift": "sw-standalone-core-pc",
        "lab": "sw-standalone-core-pc",
    },
    "multi-well": {
        "overview": "mw-level-1-2-diagrams",
        "diagrams": "mw-level-1-2-diagrams",
        "inventory": "mw-step-1-inventory",
        "1": "mw-step-1-inventory",
        "qaqc": "mw-step-2-qaqc",
        "normalization": "mw-step-2-qaqc",
        "norm": "mw-step-2-qaqc",
        "strat-norm": "mw-step-2-qaqc",
        "2": "mw-step-2-qaqc",
        "rock-typing": "mw-step-3-rock-typing",
        "rock_typing": "mw-step-3-rock-typing",
        "fzi": "mw-step-3-rock-typing",
        "winland": "mw-step-3-rock-typing",
        "lucia": "mw-step-3-rock-typing",
        "rqi": "mw-step-3-rock-typing",
        "3": "mw-step-3-rock-typing",
        "perm-transform": "mw-step-4-perm-transform",
        "perm_transform": "mw-step-4-perm-transform",
        "perm": "mw-step-4-perm-transform",
        "4": "mw-step-4-perm-transform",
        "shf": "mw-step-5-shf",
        "leverett": "mw-step-5-shf",
        "skelt": "mw-step-5-shf",
        "saturation-height": "mw-step-5-shf",
        "5": "mw-step-5-shf",
        "volumetrics": "mw-step-6-volumetrics",
        "uncertainty": "mw-step-6-volumetrics",
        "stoiip": "mw-step-6-volumetrics",
        "giip": "mw-step-6-volumetrics",
        "monte-carlo": "mw-step-6-volumetrics",
        "lhs": "mw-step-6-volumetrics",
        "6": "mw-step-6-volumetrics",
        "report": "mw-step-7-report",
        "fdp": "mw-step-7-report",
        "dossier": "mw-step-7-report",
        "7": "mw-step-7-report",
        # Standalone Modules
        "swirr-corr": "mw-standalone-scal-relperm",
        "swirr": "mw-standalone-scal-relperm",
        "scal": "mw-standalone-scal-relperm",
        "relperm": "mw-standalone-scal-relperm",
        "swof": "mw-standalone-scal-relperm",
        "let": "mw-standalone-scal-relperm",
        "brooks-corey": "mw-standalone-scal-relperm",
        "corey": "mw-standalone-scal-relperm",
        "fpt": "mw-standalone-fpt",
        "pressure": "mw-standalone-fpt",
        "contacts": "mw-standalone-fpt",
        "fwl": "mw-standalone-fpt",
        "goc": "mw-standalone-fpt",
        "owc": "mw-standalone-fpt",
        "gradient": "mw-standalone-fpt",
        "mdt": "mw-standalone-fpt",
        "rft": "mw-standalone-fpt",
        "rt-classifier": "mw-standalone-rt-classifier",
        "classifier": "mw-standalone-rt-classifier",
        "drt": "mw-standalone-rt-classifier",
        "ml-training": "mw-standalone-ml-training",
        "ml_training": "mw-standalone-ml-training",
        "ml": "mw-standalone-ml-training",
        "piml": "mw-standalone-ml-training",
        "conformal": "mw-standalone-ml-training",
    },
}


class SSOTReferenceRegistry:
    """In-memory cache and parser for quick_pp workflow technical docs."""

    def __init__(self, docs_dir: Optional[Path] = None):
        self.docs_dir = docs_dir or _DOCS_WORKFLOWS_DIR
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_loaded: float = 0.0

    def _ensure_loaded(self) -> None:
        """Reload docs if files have been modified or cache is empty."""
        now = time.time()
        if self._cache and (now - self._last_loaded < 5.0):
            return

        single_file = self.docs_dir / "single_well.md"
        multi_file = self.docs_dir / "multi_well.md"
        index_file = self.docs_dir / "index.md"

        if not single_file.is_file() and not multi_file.is_file():
            logger.warning("SSOT docs directory not found at: %s", self.docs_dir)
            return

        cache: Dict[str, Dict[str, Any]] = {
            "single-well": self._parse_markdown_file(single_file, "Single-Well Petrophysics"),
            "multi-well": self._parse_markdown_file(multi_file, "Multi-Well Field Evaluation"),
            "overview": self._parse_markdown_file(index_file, "Dual-Space Workflow Architecture"),
        }
        self._cache = cache
        self._last_loaded = now

    def _parse_markdown_file(self, path: Path, title: str) -> Dict[str, Any]:
        """Extract structured sections and anchor blocks from a workflow markdown file."""
        if not path.is_file():
            return {"title": title, "raw": "", "sections": {}, "anchors": {}}

        raw = path.read_text(encoding="utf-8")
        sections: Dict[str, str] = {}
        anchors: Dict[str, str] = {}

        # Split on headers of level 2 or 3: ## or ###
        header_pattern = re.compile(r"^(#{2,3}\s+.*)$", re.MULTILINE)
        splits = header_pattern.split(raw)

        # First chunk before any header
        if splits:
            preamble = splits[0].strip()
            if preamble:
                sections["preamble"] = preamble

        for i in range(1, len(splits), 2):
            header_line = splits[i].strip()
            body_text = splits[i + 1].strip() if i + 1 < len(splits) else ""
            combined_sec = f"{header_line}\n\n{body_text}".strip()

            # Find explicit anchor tag like {#sw-step-1-qaqc}
            anchor_match = re.search(r"\{#([\w\-]+)\}", header_line)
            if anchor_match:
                anchor_tag = anchor_match.group(1)
                anchors[anchor_tag] = combined_sec

            # Clean title
            clean_title = re.sub(r"\{#[\w\-]+\}", "", header_line).lstrip("#").strip()
            sections[clean_title] = combined_sec

        return {
            "title": title,
            "raw": raw,
            "sections": sections,
            "anchors": anchors,
        }

    def get_step_reference(self, scope: str, step: str) -> str:
        """Retrieve grounded markdown reference for a specific scope and step."""
        self._ensure_loaded()
        scope_key = "multi-well" if "multi" in scope.lower() else "single-well"
        step_norm = step.lower().strip()

        # Map step ID to anchor tag
        anchor_tag = STEP_ALIASES.get(scope_key, {}).get(step_norm)
        scope_data = self._cache.get(scope_key, {})
        anchors = scope_data.get("anchors", {})

        if anchor_tag and anchor_tag in anchors:
            return anchors[anchor_tag]

        # Try partial match on section titles
        for title, content in scope_data.get("sections", {}).items():
            if step_norm in title.lower():
                return content

        # Fallback to overview or preamble
        if "sw-level-1-2-diagrams" in anchors and scope_key == "single-well":
            return anchors["sw-level-1-2-diagrams"]
        if "mw-level-1-2-diagrams" in anchors and scope_key == "multi-well":
            return anchors["mw-level-1-2-diagrams"]

        return scope_data.get("raw", "")[:2500]

    def query_reference(
        self, query: str, scope: Optional[str] = None, max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Search SSOT documentation for keywords or formulas."""
        self._ensure_loaded()
        stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
            "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
            "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
            "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
            "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
            "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
            "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
            "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
            "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
            "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
            "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your",
            "yours", "yourself", "yourselves", "tell", "please", "average", "value"
        }
        raw_tokens = re.findall(r"\b[a-z0-9_-]{2,}\b", query.lower())
        meaningful_tokens = [t for t in raw_tokens if t not in stopwords]
        if not meaningful_tokens:
            return []

        results: List[Dict[str, Any]] = []
        scopes_to_search = [scope] if scope and scope in self._cache else ["single-well", "multi-well"]

        for sc in scopes_to_search:
            sc_data = self._cache.get(sc, {})
            anchors = sc_data.get("anchors", {})

            for anchor_id, content in anchors.items():
                content_lower = content.lower()
                first_line = content.splitlines()[0] if content else anchor_id
                clean_header = re.sub(r"\{#[\w\-]+\}", "", first_line).lstrip("#").strip()
                header_lower = clean_header.lower()

                title_matches = sum(1 for token in meaningful_tokens if token in header_lower or token in anchor_id.lower())
                body_matches = sum(content_lower.count(token) for token in meaningful_tokens)
                if title_matches > 0 or body_matches > 0:
                    score = title_matches * 5 + body_matches
                    results.append(
                        {
                            "scope": sc,
                            "anchor": anchor_id,
                            "title": clean_header,
                            "score": score,
                            "content": content,
                        }
                    )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]


# Singleton instance
ssot_registry = SSOTReferenceRegistry()


def get_step_reference(scope: str, step: str) -> str:
    """Convenience getter for active step reference."""
    return ssot_registry.get_step_reference(scope, step)


def query_workflow_reference(
    query: str, scope: Optional[str] = None, max_results: int = 3
) -> List[Dict[str, Any]]:
    """Convenience query function."""
    return ssot_registry.query_reference(query, scope, max_results)

