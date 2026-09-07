"""
Strict output contract for the research agent.

The agent's raw output (JSON produced by the LLM) is parsed into these models
before it is ever returned to a caller. If the LLM produces something that
doesn't fit this shape, validation fails loudly instead of silently shipping
garbage - this is the "left shift": catch bad output at the boundary, not in
production logs three days later.
"""

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Company(BaseModel):
    name: str = Field(min_length=1, description="Company legal / brand name")
    website: HttpUrl = Field(description="Official company website, must be a valid URL")
    country: str = Field(min_length=1, description="Country of headquarters, e.g. UAE, Saudi Arabia")
    evidence_url: HttpUrl = Field(
        description="Source URL (search result) that supports this company's existence"
    )
    summary: str = Field(min_length=1, max_length=400, description="One-line description of what the company does")

    @field_validator("country")
    @classmethod
    def country_in_gcc(cls, v: str) -> str:
        allowed = {
            "uae", "united arab emirates", "saudi arabia", "ksa", "qatar",
            "bahrain", "kuwait", "oman",
        }
        if v.strip().lower() not in allowed:
            raise ValueError(f"country '{v}' is not a recognized UAE/GCC country")
        return v


class CompanyList(BaseModel):
    query: str
    companies: list[Company] = Field(default_factory=list)

    @field_validator("companies")
    @classmethod
    def no_duplicate_websites(cls, v: list[Company]) -> list[Company]:
        seen = set()
        for c in v:
            key = str(c.website).rstrip("/").lower()
            if key in seen:
                raise ValueError(f"duplicate website in results: {c.website}")
            seen.add(key)
        return v
