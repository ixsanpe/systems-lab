import re

from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator
from nsearch.ingestion.constants import BASE_URL
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from pathlib import Path
import yaml

# Top-level entries follow "XX_Something/" (canton or "CH" federal code).
# A handful of entries (Index/, Jobs/, Sitemaps/, ...) don't and must be skipped.
AREA_PATTERN = re.compile(r"^([A-Z]{2})_")

async def generate_valid_areas() -> set[str]:
    valid_areas = set()
    # async I/O network call
    async with aiohttp.ClientSession() as session, session.get(BASE_URL) as resp:
        if resp.status != 200:
            raise ConnectionError

        html = await resp.text()

    # Move the processing outside of the async context manager
    soup = BeautifulSoup(html, "lxml")
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not isinstance(href, str):
            continue
        if not href.endswith("/") or href == "../":
            continue
        match = AREA_PATTERN.match(href)
        if match:
            valid_areas.add(match.group(1))
    return valid_areas

class EntscheidSuche(BaseModel):
    """ Parameters configuration for the Entscheidsuche model """
    areas: dict[str, list[str]]
    """Maps each area code to the list of topics valid for that area (e.g. {"ZH": ["Steuerrekurs", ...]}).

    Deliberately not two flat `areas`/`topics` lists: not every area has every
    topic (CH's federal topics aren't ZH's cantonal ones), so a cross product
    of separate lists could express area/topic pairs that don't actually exist.
    """

    @field_validator("areas", mode="after")
    @classmethod
    def check_areas_known(cls, areas: dict[str, list[str]], info: ValidationInfo) -> dict[str, list[str]]:
        """Check area codes against a known-valid set, when the caller supplies one via context."""
        valid_areas = info.context.get("valid_areas") if info.context else None
        if valid_areas is not None and not all(area in valid_areas for area in areas):
            raise ValueError(f"unknown area code(s): {sorted(set(areas) - valid_areas)}")
        return areas

    @classmethod
    def from_yaml(cls, path: str | Path, valid_areas: set[str] | None = None) -> "EntscheidSuche":
        """Load configuration from YAML file.

        `valid_areas`, if given, is checked against `areas` during validation
        (fetch it once via `generate_valid_areas()` and pass it in here —
        this method itself makes no network calls).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        try:
            return cls.model_validate(config_dict, context={"valid_areas": valid_areas})
        except ValidationError as e:
            print(f"Configuration validation failed for {path}:")
            print(e)
            raise
