"""
Test finale su un'unica configurazione, con forecast operativa e tempo di esecuzione.

Controlla in automatico:
  1. la LOOCV non e' cambiata rispetto alla run del paper (SS_RMSE atteso, se indicato);
  2. r_pvalue e' compilato;
  3. forecast.csv ha status ok, fase coerente con le soglie di training;
  4. i modelli final_model_<fase>.npz sono nel bundle;
  5. il modello riapplicato offline (apply_final_model) da' lo stesso valore del servizio;
  6. il bundle contiene il target e non contiene piu' i file eliminati;
  7. <var>_timeseries.csv ha le colonne attese.
Misura il tempo dell'Execute e segnala se il campo predittore e' stato scaricato
(prima esecuzione) o letto dalla copia locale.

Uso: python test_wps_drought.py   oppure   pytest -s test_wps_drought.py
Esito a schermo e in OUT_DIR/test_report.txt.
Il log interno di PyWPS segue la configurazione di PyWPS, non questo file.
"""
import gc
import json
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# --- configurazione ---------------------------------------------------------------
CFG = dict(case="ZRB", target="/Users/giuliopalcic/Desktop/albatross_inc/albatross/data/ZRB_monthly_inflows.txt",
           months="3,4,5", start="1981", end="2017", field="sst")
M = 2
PHASE_INDEX = "NINO 3.4"
FORECAST_YEAR = 2018
EXPECTED_SSRMSE = None

# target: nome di un file in albatross/data su GitHub (senza .txt) OPPURE percorso locale.
# Java, per la sezione 5.1 del paper:
# CFG = dict(case="SA-OBS", target="/percorso/completo/file_java.txt", months="10,11,12",
#            start="1981", end="2010", field="sst")
# M = 2; PHASE_INDEX = "NINO 3.4"; FORECAST_YEAR = 2025; EXPECTED_SSRMSE = 34.8
# -----------------------------------------------------------------------------------

OUT_DIR = Path(f"albatross_test_{datetime.now():%Y%m%d_%H%M}").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["ALBATROSS_OUT_DIR"] = str(OUT_DIR)

import numpy as np
import pandas as pd
from pywps import Service
from pywps.tests import client_for
from albatross.processes.wps_drought import Drought
from albatross import atmos_ocean_data as aod
from albatross.climdiv_data import create_kwgroups, get_data
from albatross.utils import apply_final_model

BASE_URL = ("https://raw.githubusercontent.com/climateintelligence/albatross/"
            "refs/heads/main/albatross/data/")

REPORT = OUT_DIR / "test_report.txt"
REMOVED = ("operational_status.json", "index_training.csv", "phase_mask_")
results = []


def out(msg=""):
    """Scrive a schermo e, sempre, su test_report.txt (indipendente da pytest)."""
    print(msg, flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def check(name, ok, detail=""):
    ok = bool(ok)
    results.append(ok)
    out(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _predictor_file():
    """Percorso della copia locale del campo predittore (None se non e' un file PSL)."""
    key = {"sst": "sst", "slp": "slp", "z500": "hgt"}.get(CFG["field"])
    if key is None:
        return None
    env, name, _ = aod.PSL_SOURCES[key]
    return Path(aod._local_path(env, name))


def _clear_memory_caches():
    """Svuota le cache in memoria, cosi' il tempo misurato include la lettura del NetCDF."""
    if hasattr(get_data, "_cache"):
        get_data._cache.clear()
    for f in (aod.openDAPsst_cached_frozen, aod.openDAPslp_cached_frozen,
              aod.openDAPz500_cached_frozen):
        f.cache_clear()


def _target_url():
    """URL del target. Se CFG['target'] e' un file locale esistente, lo serve con un piccolo
    server HTTP su localhost (il servizio accetta il target come URL o come upload)."""
    p = Path(CFG["target"]).expanduser()
    if not p.is_file():
        return BASE_URL + f"{CFG['target']}.txt"
    import functools
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(p.parent.resolve()))
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}/{quote(p.name)}"


def run_service():
    client = client_for(Service(processes=[Drought()]))
    try:
        inputs = {
            "target": _target_url(),
            "start_year": CFG["start"],
            "end_year": CFG["end"],
            "month": CFG["months"],
            "phase_mode": str(M),
            "glo_var_name": CFG["field"],
            "aggregation": "mean",
            "forecast_year": str(FORECAST_YEAR),
        }
        if M > 1:
            inputs["indicator"] = PHASE_INDEX
        q = quote(";".join(f"{k}={v}" for k, v in inputs.items()), safe="")

        pf = _predictor_file()
        mtime0 = pf.stat().st_mtime if pf is not None and pf.exists() else None
        _clear_memory_caches()
        t0 = time.perf_counter()
        resp = client.get("/wps?service=WPS&request=Execute&version=1.0.0&identifier=drought"
                          f"&datainputs={q}")
        elapsed = time.perf_counter() - t0
        mtime1 = pf.stat().st_mtime if pf is not None and pf.exists() else None
        downloaded = pf is not None and mtime1 is not None and mtime1 != mtime0

        txt = resp.data.decode("utf-8", errors="replace")
        if resp.status_code != 200 or "ProcessSucceeded" not in txt:
            m = re.search(r"<(?:\w+:)?ExceptionText>(.*?)</(?:\w+:)?ExceptionText>", txt, re.DOTALL)
            return None, (m.group(1).strip() if m else txt[:2000]), elapsed, downloaded
    finally:
        del client
        gc.collect()
    runs = sorted(OUT_DIR.rglob("run_*/metrics_by_phase.csv"))
    return (runs[-1].parent if runs else None), None, elapsed, downloaded


def test_wps_drought():
    idx = f", indice {PHASE_INDEX}" if M > 1 else ""
    out(f"Test {CFG['case']} {CFG['months']} {CFG['field']} M={M}{idx}, "
        f"forecast {FORECAST_YEAR} -> {OUT_DIR}")
    run, err, elapsed, downloaded = run_service()
    out(f"Tempo Execute: {elapsed:.1f} s "
        f"({'campo predittore scaricato da NOAA' if downloaded else 'campo predittore gia in locale'})")
    check("processo WPS completato", run is not None, err or "")
    assert run is not None, err
    out(f"Run: {run}")
    pc = run / "pc_vs_hindcast"

    # 1-2) LOOCV invariata, p-value compilato
    met = pd.read_csv(run / "metrics_by_phase.csv")
    row = met[met["phase"] == ("allyears" if M == 1 else "overall")].iloc[0]
    out("\n--- metrics_by_phase.csv ---\n" + met.to_string(index=False) + "\n--------------------")
    if EXPECTED_SSRMSE is not None:
        check("SS_RMSE come nel paper", abs(row["SSRMSE_%"] - EXPECTED_SSRMSE) < 0.05,
              f"{row['SSRMSE_%']:.2f}% vs {EXPECTED_SSRMSE}%")
    check("r_pvalue compilato", not np.isnan(row["r_pvalue"]), str(row["r_pvalue"]))

    # 3) forecast
    fc = pd.read_csv(run / "forecast.csv").iloc[0]
    out("\n--- forecast.csv ---\n" + fc.to_string() + "\n--------------------")
    check("forecast status", fc["status"] == "ok", str(fc["status"]))
    assert fc["status"] == "ok", fc["status"]
    if M > 1:
        meta = json.loads((pc / "phase_thresholds.json").read_text())
        thr = meta["thresholds"][0]
        expected_phase = meta["labels"][0] if fc["index_value"] <= thr else meta["labels"][-1]
        check("fase coerente con le soglie", fc["phase"] == expected_phase,
              f"indice {fc['index_value']:.3f}, soglia {thr:.3f}, fase {fc['phase']}")
    check("componenti sommano al forecast",
          np.isclose(fc["trend_component"] + fc["teleconnection_component"], fc["forecast"]))

    # 4) modelli salvati
    npz = pc / f"final_model_{fc['phase']}.npz"
    check("final_model salvato", npz.exists(), npz.name)

    # 5) riapplicazione offline sul campo GREZZO dell'anno di forecast
    kw = create_kwgroups(climdata_months=[int(m) for m in CFG["months"].split(",")],
                         climdata_startyr=FORECAST_YEAR, n_yrs=1, n_phases=1,
                         glo_var_name=CFG["field"])
    _, _, gv, _, _ = get_data(kw, glo_var_name=CFG["field"], load_target=False)
    off = apply_final_model(npz, np.asarray(gv.data, dtype=float)[0], FORECAST_YEAR)
    check("offline == servizio", np.isclose(off["forecast"], fc["forecast"]),
          f"{off['forecast']:.3f} vs {fc['forecast']:.3f}")

    # 6) contenuto del bundle
    zips = sorted(run.glob("outputs_*.zip"))
    check("bundle presente", bool(zips), zips[0].name if zips else "")
    if zips:
        names = zipfile.ZipFile(zips[0]).namelist()
        out("\n--- contenuto del bundle ---\n" + "\n".join(sorted(names)) + "\n--------------------")
        check("target nel bundle", any(Path(n).name.startswith("target_") for n in names))
        left = [n for n in names if any(r in n for r in REMOVED)]
        check("file eliminati assenti", not left, ", ".join(left))

    # 7) colonne della serie di hindcast
    ts = pd.read_csv(run / f"{CFG['field']}_timeseries.csv")
    expected = ["year", "observed", "hindcast", "trend_component"] + (
        ["phase", "index_value"] if M > 1 else [])
    check("colonne di timeseries", list(ts.columns) == expected, ", ".join(ts.columns))

    n_ok = sum(results)
    out(f"\nTempo Execute: {elapsed:.1f} s")
    out(f"{n_ok}/{len(results)} controlli superati. Report: {REPORT}")
    assert all(results), "alcuni controlli sono falliti (vedi test_report.txt)"


if __name__ == "__main__":
    test_wps_drought()