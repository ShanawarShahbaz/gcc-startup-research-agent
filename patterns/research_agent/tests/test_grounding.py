"""
Grounding tests: all network calls are mocked, so this stays fast and
deterministic (part of the left-shift suite, not an integration test).
"""

from unittest.mock import Mock, patch

import requests

from grounding import (
    check_company,
    evidence_mentions_company,
    filter_grounded,
    website_is_live,
)
from schema import Company, CompanyList

COMPANY = Company(
    name="Careem",
    website="https://www.careem.com",
    country="UAE",
    evidence_url="https://example.com/article-about-careem",
    summary="Ride-hailing and delivery super-app based in Dubai.",
)


def _mock_response(status_code=200, text=""):
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    resp.raise_for_status = Mock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    return resp


@patch("grounding.requests.get")
def test_website_is_live_true_on_200(mock_get):
    mock_get.return_value = _mock_response(200)
    assert website_is_live("https://www.careem.com") is True


@patch("grounding.requests.get")
def test_website_is_live_false_on_404(mock_get):
    mock_get.return_value = _mock_response(404)
    assert website_is_live("https://dead-site.example") is False


@patch("grounding.requests.get")
def test_website_is_live_false_on_connection_error(mock_get):
    mock_get.side_effect = requests.ConnectionError("could not connect")
    assert website_is_live("https://unreachable.example") is False


@patch("grounding.requests.get")
def test_evidence_mentions_company_true(mock_get):
    mock_get.return_value = _mock_response(200, text="<html>Careem raised funding...</html>")
    assert evidence_mentions_company("https://example.com/article", "Careem") is True


@patch("grounding.requests.get")
def test_evidence_mentions_company_false(mock_get):
    mock_get.return_value = _mock_response(200, text="<html>Unrelated content about cars</html>")
    assert evidence_mentions_company("https://example.com/article", "Careem") is False


@patch("grounding.requests.get")
def test_evidence_mentions_company_false_on_fetch_failure(mock_get):
    mock_get.side_effect = requests.Timeout("timed out")
    assert evidence_mentions_company("https://example.com/article", "Careem") is False


@patch("grounding.requests.get")
def test_check_company_passes_when_both_checks_pass(mock_get):
    mock_get.return_value = _mock_response(200, text="Careem is a UAE startup")
    result = check_company(COMPANY)
    assert result.passed is True
    assert result.issues == []


@patch("grounding.requests.get")
def test_check_company_fails_and_reports_issue_when_evidence_irrelevant(mock_get):
    mock_get.return_value = _mock_response(200, text="nothing relevant here")
    result = check_company(COMPANY)
    assert result.passed is False
    assert any("does not mention" in issue for issue in result.issues)


@patch("grounding.requests.get")
def test_filter_grounded_drops_failing_companies(mock_get):
    mock_get.return_value = _mock_response(200, text="irrelevant content")
    company_list = CompanyList(query="test", companies=[COMPANY])
    filtered, results = filter_grounded(company_list)
    assert filtered.companies == []
    assert len(results) == 1
    assert results[0].passed is False


@patch("grounding.requests.get")
def test_filter_grounded_keeps_passing_companies(mock_get):
    mock_get.return_value = _mock_response(200, text="Careem is featured here")
    company_list = CompanyList(query="test", companies=[COMPANY])
    filtered, results = filter_grounded(company_list)
    assert len(filtered.companies) == 1
    assert filtered.companies[0].name == "Careem"
