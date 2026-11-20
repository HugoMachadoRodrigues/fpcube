"""Fixed definitions for the Florida Pasture Cube. Every stage reads its constants from here."""
from pathlib import Path
import os

# Data store: data/ at the project root (not in git). Override with FPCUBE_DATA.
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FPCUBE_DATA", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RELEASE = DATA / "release"

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
CDL_WETLAND_WOODY = (190,)
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
