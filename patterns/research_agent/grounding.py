"""
Grounding layer: checks that already-schema-valid output is actually TRUE,
not just well-formed.

Schema validation (schema.py) only proves the JSON has the right shape - a
well-formed URL can still be dead, or point to a page that never mentions the
company. This layer catches that class of hallucination:

1. website_is_live(url)         - does the claimed company site actually resolve?
2. evidence_mentions_company()  - does the evidence page's content actually
                                   contain the company name (weak but cheap
                                   grounding signal)?

Network calls are isolated in small functions so they're easy to mock in tests.
"""

import re
from dataclasses import dataclass, field

import requests

from schema import Company, CompanyList

REQUEST_TIMEOUT = 8
USER_AGENT = "Mozilla/5.0 (research-agent grounding-check)"


@dataclass
class GroundingResult:
    company: Company
    passed: bool
    issues: list[str] = field(default_factory=list)


def website_is_live(url: str) -> bool:
    """A website that 404s or refuses to connect is not a real grounded result."""
    try:
        resp = requests.get(
            str(url), timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}, allow_redirects=True
        )
        return resp.status_code < 400
    except requests.RequestException:
        return False


def evidence_mentions_company(evidence_url: str, company_name: str) -> bool:
    """Cheap grounding signal: the evidence page's own text must mention the company name."""
    try:
        resp = requests.get(
            str(evidence_url), timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
        resp.raise_for_status()
    except requests.RequestException:
        return False

    text = resp.text
    pattern = re.escape(company_name)
    return re.search(pattern, text, re.IGNORECASE) is not None


def check_company(company: Company, verify_website_live: bool = True) -> GroundingResult:
    issues: list[str] = []

    if verify_website_live and not website_is_live(company.website):
        issues.append(f"website did not resolve: {company.website}")

    if not evidence_mentions_company(company.evidence_url, company.name):
        issues.append(
            f"evidence_url does not mention '{company.name}': {company.evidence_url}"
        )

    return GroundingResult(company=company, passed=not issues, issues=issues)


def ground_check_list(companies: CompanyList, verify_website_live: bool = True) -> list[GroundingResult]:
    return [check_company(c, verify_website_live=verify_website_live) for c in companies.companies]


def filter_grounded(companies: CompanyList, verify_website_live: bool = True) -> tuple[CompanyList, list[GroundingResult]]:
    """Return a new CompanyList containing only grounded companies, plus the full results for logging."""
    results = ground_check_list(companies, verify_website_live=verify_website_live)
    grounded = [r.company for r in results if r.passed]
    return CompanyList(query=companies.query, companies=grounded), results
