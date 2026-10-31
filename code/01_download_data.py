"""Step 1 - Download the raw data (World Bank WDI API + UNDP HDR time series).

Raw files are stored unchanged in data/raw together with a metadata file
(indicator names, API 'lastupdated' stamp, retrieval time, SHA-256 of the HDR
file) so that the exact vintage used in the paper is documented.  The
downloaded raw files are shipped with the package, so this step only needs to
be re-run to refresh the data.
"""
import hashlib, json, time, datetime as dt
import requests, pandas as pd
from config import *

API = "https://api.worldbank.org/v2/country/{c}/indicator/{k}"


def fetch_indicator(code, countries):
    out, meta = [], {}
    params = {"format": "json", "per_page": 20000, "date": "1990:2024"}
    for attempt in range(5):
        try:
            r = requests.get(API.format(c=";".join(countries), k=code), params=params, timeout=120)
            r.raise_for_status()
            js = r.json()
            break
        except Exception as e:  # network hiccup -> retry
            print("  retry", code, e)
            time.sleep(3)
    else:
        raise RuntimeError(f"could not download {code}")
    if len(js) < 2 or js[1] is None:
        print("  !! no data for", code)
        return pd.DataFrame(), meta
    meta = {"name": js[1][0]["indicator"]["value"], "lastupdated": js[0].get("lastupdated")}
    for x in js[1]:
        iso = x["countryiso3code"] or {"XD": "HIC"}.get(x["country"]["id"], x["country"]["id"])   # API id of "High income" is XD
        out.append({"iso3": iso, "year": int(x["date"]),
                    "indicator": code, "value": x["value"]})
    return pd.DataFrame(out), meta


def main():
    countries = list(LATAM) + list(AGGREGATES)
    frames, metadata = [], {}
    for code in WDI:
        print("WDI", code)
        d, m = fetch_indicator(code, countries)
        frames.append(d)
        metadata[code] = m
    wdi = pd.concat(frames, ignore_index=True)
    wdi.to_csv(RAW / "wdi_raw.csv", index=False)

    # Internet use for ALL economies + country metadata (used to build the frontier-diffusion series
    # = mean Internet use of high-income, non-Latin-American economies, for the IV exercise)
    print("WDI IT.NET.USER.ZS, all economies")
    d, m = fetch_indicator("IT.NET.USER.ZS", ["all"])
    d.to_csv(RAW / "wdi_internet_all.csv", index=False)
    cm = requests.get("https://api.worldbank.org/v2/country", params={"format": "json", "per_page": 500}, timeout=120).json()[1]
    pd.DataFrame([{"id": c["id"], "iso3": c["id"], "name": c["name"], "region": c["region"]["id"],
                   "income": c["incomeLevel"]["id"]} for c in cm]).to_csv(RAW / "wb_countries.csv", index=False)

    print("UNDP HDR time series")
    r = requests.get(HDR_URL, timeout=300)
    r.raise_for_status()
    HDR_FILE.write_bytes(r.content)
    metadata["_hdr"] = {"url": HDR_URL, "sha256": hashlib.sha256(r.content).hexdigest(),
                        "bytes": len(r.content)}
    metadata["_retrieved_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds")
    (RAW / "download_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("done:", len(wdi), "WDI rows")


if __name__ == "__main__":
    main()
