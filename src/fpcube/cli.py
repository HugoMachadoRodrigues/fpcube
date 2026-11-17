"""Command line entry point: fpcube <step> [args].

Steps, in build order:
  cdl          download Cropland Data Layer 2012-2025 for Florida
  parcels      download FDOR grazing parcels;  parcels-all  other parcels touching pasture
  fields       delineate fields (needs cdl, parcels, parcels-all)
  static       soil, terrain, wetlands per field
  optical      HLS extraction per MGRS tile (resumable)  [tile ...]
  radar        Sentinel-1 extraction (resumable)
  prism        PRISM daily weather
  labels       USDM, NASS (needs FPCUBE_NASS_KEY), hurricanes
  cube         assemble the release (cube.zarr, fields, long tables, labels)
  benchmark    run the three benchmark tasks
  release      datasheet, STAC catalog, checksums
"""
import runpy
import sys

STEPS = {
    "cdl": ("fpcube.sources.cdl", []), "parcels": ("fpcube.sources.parcels", []),
    "parcels-all": ("fpcube.sources.parcels", ["all"]), "fields": ("fpcube.fields", []),
    "static": ("fpcube.sources.static", []), "optical": ("fpcube.sources.optical", []),
    "radar": ("fpcube.sources.radar", []), "prism": ("fpcube.sources.prism", []),
    "labels": ("fpcube.sources.labels", []), "cube": ("fpcube.cube", []),
    "benchmark": ("fpcube.benchmark.run", []), "release": ("fpcube.release", []),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        print(__doc__)
        sys.exit(1)
    mod, extra = STEPS[sys.argv[1]]
    sys.argv = [mod] + extra + sys.argv[2:]
    runpy.run_module(mod, run_name="__main__")


if __name__ == "__main__":
    main()
