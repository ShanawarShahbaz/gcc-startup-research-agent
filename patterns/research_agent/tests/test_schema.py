"""
Left-shifted schema tests: no LLM, no network. These run in milliseconds and
catch structural regressions before the agent is ever invoked - the cheapest
place to catch a bug.
"""

import pytest
from pydantic import ValidationError

from schema import Company, CompanyList

VALID_COMPANY = {
    "name": "Careem",
    "website": "https://www.careem.com",
    "country": "UAE",
    "evidence_url": "https://example.com/article-about-careem",
    "summary": "Ride-hailing and delivery super-app based in Dubai.",
}


def test_valid_company_passes():
    Company.model_validate(VALID_COMPANY)


def test_missing_website_fails():
    bad = {**VALID_COMPANY}
    del bad["website"]
    with pytest.raises(ValidationError):
        Company.model_validate(bad)


def test_invalid_url_fails():
    bad = {**VALID_COMPANY, "website": "not-a-url"}
    with pytest.raises(ValidationError):
        Company.model_validate(bad)


def test_non_gcc_country_rejected():
    bad = {**VALID_COMPANY, "country": "USA"}
    with pytest.raises(ValidationError):
        Company.model_validate(bad)


@pytest.mark.parametrize(
    "country",
    ["UAE", "Saudi Arabia", "Qatar", "Bahrain", "Kuwait", "Oman", "uae", "KSA"],
)
def test_all_gcc_countries_accepted(country):
    Company.model_validate({**VALID_COMPANY, "country": country})


def test_empty_company_list_is_valid():
    CompanyList.model_validate({"query": "no results found", "companies": []})


def test_duplicate_websites_rejected():
    dup = {**VALID_COMPANY, "name": "Careem Duplicate"}
    with pytest.raises(ValidationError):
        CompanyList.model_validate(
            {"query": "test", "companies": [VALID_COMPANY, dup]}
        )


def test_summary_too_long_rejected():
    bad = {**VALID_COMPANY, "summary": "x" * 401}
    with pytest.raises(ValidationError):
        Company.model_validate(bad)


def test_empty_name_rejected():
    bad = {**VALID_COMPANY, "name": ""}
    with pytest.raises(ValidationError):
        Company.model_validate(bad)
