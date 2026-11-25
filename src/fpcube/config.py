"""Fixed definitions for the Florida Pasture Cube. Every stage reads its constants from here."""
from pathlib import Path
import os

# Data store: data/ at the project root (not in git). Override with FPCUBE_DATA.
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FPCUBE_DATA", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RELEASE = DATA / "release"

# Field-set version. v1.0 lives in INTERIM/fields, INTERIM/optical, ...; a later version (FPCUBE_FIELDS=v1.1) keeps
# its own fields and the observations of its new or reshaped fields in INTERIM/<name>_<version> and reuses the v1.0
# observations of the fields it did not change (see fpcube.fields.lineage).
FIELDS_VERSION = os.environ.get("FPCUBE_FIELDS", "v1.0")


def vdir(name, version=None):
    v = version or FIELDS_VERSION
    return INTERIM / name if v == "v1.0" else INTERIM / f"{name}_{v}"


# Field selections: a version that keeps a subset of another version's fields (no new delineation, no new
# extraction). v1.2 keeps the v1.1 fields without land-use change and, in tier B, only those whose pixels were CDL
# pasture in at least TIER_B_MIN_PASTURE_YEARS of the ten years on average (the official FLUCCS land use showed
# tier B fields below that to be mostly crops, residential or reclaimed land).
SELECTION = {"v1.2": "v1.1"}
TIER_B_MIN_PASTURE_YEARS = 9.5


# Field recoveries: a version that keeps every field of a base version and adds new fields (only these are extracted).
# v1.3 adds tier C: official FLUCCS pasture that the CDL maps as woody or wetland, open in ESA WorldCover 2021.
RECOVERY = {"v1.3": "v1.2"}
RECOVER_FLUCCS = (2100, 2140)            # FLUCCS pasture codes [lo, hi)
RECOVER_WOODY_WET_YEARS = 5              # CDL woody, woody wetland or herbaceous wetland in >= this many PASTURE_YEARS
RECOVER_MIN_OPEN_PCT = 67                # WorldCover grassland + cropland share of the 30 m cell
CDL_WETLAND_HERB = (195,)


def obs_version(version=None):
    """Field version whose interim observation and static tables a version reads (a selection reads its base's)."""
    v = version or FIELDS_VERSION
    return SELECTION.get(v, v)


FIELDS_DIR = vdir("fields")


def extract_fields_path():
    """Fields whose observations must be extracted in this version: all of them in v1.0, only the new or reshaped
    ones in a later version."""
    p = FIELDS_DIR / "fields_extract.parquet"
    return p if p.exists() else FIELDS_DIR / "fields.parquet"

# Equal-area CRS for geometry and areas (same as the Cropland Data Layer).
CRS = "EPSG:5070"

# Study period and calendar.
YEARS = list(range(2016, 2027))
START, END = "2016-01-01", "2026-10-31"
COMPOSITE_DAYS = 5                      # regular step of the composites
COMPOSITE_ANCHOR = "2016-01-01"          # first composite starts here

# Cropland Data Layer classes.
CDL_PASTURE = (176, 37)                  # Grass/Pasture, Other Hay/Non Alfalfa
CDL_WOODY = (141, 142, 143, 152)         # forests, shrubland
CDL_WETLAND_WOODY = (190,)              # woody wetlands: added to the woody screen from v1.1
CDL_WATER = (111,)
CDL_DEVELOPED = (121, 122, 123, 124)
CDL_CITRUS = (72,)

# Field definition.
PASTURE_YEARS = list(range(2016, 2026))  # CDL years used for the persistent-pasture mask
PASTURE_MIN_FRACTION = 0.6               # pixel is pasture in at least 60% of those years
WOODY_SCREEN_YEARS = list(range(2012, 2017))
WOODY_MAX_YEARS = 1                      # drop pixels woody in 2+ screen years (young pine mapped as pasture)
MIN_FIELD_HA = 2.0
INTERIOR_BUFFER_M = -30.0                # one-pixel inward buffer for optical/radar extraction

# Florida Department of Revenue land-use codes for grazing land (soil capability classes I-VI).
DOR_GRAZING = ("060", "061", "062", "063", "064", "065")
PARCEL_SERVICE = ("https://gis.fdacs.gov/mapping/rest/services/"
                  "GIO_Parcels_for_OAWP_Optimized/FeatureServer/0")
PARCEL_FIELDS = "CO_NO,PARCEL_ID,DOR_UC,AgTaxExempt"   # owner names are never downloaded

FLORIDA_FIPS = "12"

# regions used to summarise fields (Table of fields by region, validation sample strata)
_R = {"Northwest": "Bay Calhoun Escambia Franklin Gadsden Gulf Holmes Jackson Jefferson Leon Liberty Madison Okaloosa "
                   "Santa_Rosa Taylor Wakulla Walton Washington",
      "Northeast": "Alachua Baker Bradford Clay Columbia Dixie Duval Flagler Gilchrist Hamilton Lafayette Levy Marion "
                   "Nassau Putnam St._Johns Suwannee Union",
      "Central": "Brevard Citrus Hernando Lake Orange Osceola Pasco Pinellas Polk Seminole Sumter Volusia Hillsborough",
      "South": "Broward Charlotte Collier DeSoto Glades Hardee Hendry Highlands Indian_River Lee Manatee Martin "
               "Miami-Dade Monroe Okeechobee Palm_Beach Sarasota St._Lucie"}
REGION = {c.replace("_", " "): r for r, cs in _R.items() for c in cs.split()}
