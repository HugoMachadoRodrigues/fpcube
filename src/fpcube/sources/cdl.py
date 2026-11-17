"""Download the USDA NASS Cropland Data Layer for Florida (FIPS 12) through the CropScape web service."""
import re
import time

import requests

from fpcube import config

SERVICE = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={y}&fips=" + config.FLORIDA_FIPS
YEARS = range(2012, 2026)


def download(year, retries=6):
    out = config.RAW / "cdl" / f"CDL_{year}_{config.FLORIDA_FIPS}.tif"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            r = requests.get(SERVICE.format(y=year), timeout=600)
            r.raise_for_status()
            url = re.search(r"<returnURL>(.*?)</returnURL>", r.text).group(1)
            with requests.get(url, stream=True, timeout=1200) as g:
                g.raise_for_status()
                tmp = out.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in g.iter_content(1 << 22):
                        f.write(chunk)
            tmp.rename(out)
            return out
        except Exception as e:
            print(f"cdl {year} retry {i + 1}: {e}", flush=True)
            time.sleep(30 * (i + 1))
    raise RuntimeError(f"CDL {year} failed")


if __name__ == "__main__":
    for y in YEARS:
        print(download(y), flush=True)
