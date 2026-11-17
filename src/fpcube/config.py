"""Fixed definitions for the Florida Pasture Cube. Every stage reads its constants from here."""
from pathlib import Path
import os

# Data store outside OneDrive (large, regenerable). Override with FPCUBE_DATA.
DATA = Path(os.environ.get("FPCUBE_DATA", Path.home() / "fpcube_data"))
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
