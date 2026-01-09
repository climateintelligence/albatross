# albatross/processes/wps_drought.py

import logging
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from importlib.resources import files

import logging

logging.basicConfig(
    level=logging.DEBUG,   # or INFO
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

def dbg(msg: str):
    print(f"[DBG] {msg}", flush=True)
    LOGGER.info(f"[DBG] {msg}")


from pywps import Process, LiteralInput, ComplexInput, ComplexOutput, FORMATS, Format
from pywps.app.Common import Metadata


from albatross.climdiv_data import get_data, create_kwgroups
from albatross.new_simpleNIPA import NIPAphase
from albatross.utils import (
    glo_var_Map,
    glo_var_Map_unfiltered,
    plot_model_results,
    append_model_pcs,
    extract_target_name,
    _compute_phase_thresholds,
    _save_phase_artifacts,
)

LOGGER = logging.getLogger("PYWPS")
FORMAT_PDF = Format("image/pdf", extension=".pdf", encoding="base64")

default_target_file = "APGD_Como"
crv_flag = True
map_flag = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def resolve_target_input(inp, outdir: Path) -> Path:
    """
    Download target file if input is a URL, or accept uploaded file.
    Compatible with your original behavior.
    """
    if inp and hasattr(inp, "data") and isinstance(inp.data, str) and inp.data.startswith("http"):
        url = inp.data
        filename = Path(urlparse(url).path).name or "target.txt"
        dst = outdir / filename
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dst.write_bytes(r.content)
        return dst

    if hasattr(inp, "file") and inp.file:
        path = Path(inp.file)
        with path.open("r") as f:
            first = f.readline().strip()
            second = f.readline().strip()

        # If uploaded file contains only a URL, download it
        if first.startswith("http") and not second:
            url = first
            filename = Path(urlparse(url).path).name or "target.txt"
            dst = outdir / filename
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dst.write_bytes(r.content)
            return dst

        return path

    raise ValueError("No valid target supplied. Provide a URL or upload a .txt file.")


# -----------------------------------------------------------------------------
# Process
# -----------------------------------------------------------------------------
class Drought(Process):
    def __init__(self):
        inputs = [
            ComplexInput(
                identifier="target",
                title="Upload the variable to forecast",
                abstract="text file of target file",
                default=(
                    "https://raw.githubusercontent.com/climateintelligence/albatross/"
                    f"refs/heads/main/albatross/data/{default_target_file}.txt"
                ),
                supported_formats=[FORMATS.TEXT],
            ),
            LiteralInput(identifier="start_year", title="Start Year", default="1971", data_type="string"),
            LiteralInput(identifier="end_year", title="End Year", default="2019", data_type="string"),
            LiteralInput(identifier="month", title="Target season", default="4,5,6", data_type="string"),
            LiteralInput(
                identifier="indicator",
                title="Climate Indicator",
                abstract="Choose between NAO, NINO 3.4, SCA, EAWR, AO, ONI, DMI, SOI, WP, PNA, QBO",
                data_type="string",
                allowed_values=["SCA", "EAWR", "AO", "ONI", "DMI", "SOI", "WP", "PNA", "NAO", "QBO", "NINO 3.4"],
                default="NAO",
            ),
            LiteralInput(
                identifier="phase_mode",
                title="Phase mode: 1/2/3/4",
                abstract="1=allyears, 2=pos/neg, 3=pos/neutral/neg, 4=pos/neutpos/neutneg/neg",
                default="2",
                data_type="string",
            ),
            LiteralInput(
                identifier="glo_var_name",
                title="Global Variable Name",
                abstract="Global variable to use ('sst' or 'slp')",
                data_type="string",
                allowed_values=["sst", "slp"],
                default="sst",
            ),
            LiteralInput(
                identifier="forecast_year",
                title="Forecast year (optional)",
                abstract="If provided, attempt operational forecast for that year.",
                data_type="string",
                min_occurs=0,
            ),
        ]

        outputs = [
            ComplexOutput(identifier="forecast_file", title="Forecast File", as_reference=True, supported_formats=[FORMATS.TEXT]),
            ComplexOutput(identifier="scatter_plot", title="Scatter Plot", as_reference=True, supported_formats=[FORMAT_PDF]),
            ComplexOutput(identifier="forecast_bundle", title="All Forecast Outputs (ZIP)", as_reference=True, supported_formats=[FORMATS.ZIP]),
            ComplexOutput(identifier="glo_var_map", title="Global variable correlation map", as_reference=True, supported_formats=[FORMAT_PDF]),
            ComplexOutput("pcs_plot", "Predicted vs Observed (PCA regression)", as_reference=True, supported_formats=[FORMAT_PDF]),
        ]

        super().__init__(
            self._handler,
            identifier="drought",
            title="Albatross",
            abstract="Seasonal hydroclimate forecasting using EOF/PCR and climate indices.",
            metadata=[
                Metadata("PyWPS", "https://pywps.org/"),
                Metadata("Birdhouse", "http://bird-house.github.io/"),
            ],
            version="0.1",
            inputs=inputs,
            outputs=outputs,
            store_supported=True,
            status_supported=True,
        )

    def _handler(self, request, response):
        LOGGER.info("Processing...")
        response.update_status("Retrieving input data and initializing variables...", 10)

        glo_var_name = request.inputs["glo_var_name"][0].data.lower()

        months = list(map(int, request.inputs["month"][0].data.split(",")))
        startyr = int(request.inputs["start_year"][0].data)
        endyr = int(request.inputs["end_year"][0].data)

        M = int(request.inputs["phase_mode"][0].data)
        if M not in (1, 2, 3, 4):
            raise ValueError("phase_mode must be 1, 2, 3, or 4")

        indicator = request.inputs["indicator"][0].data.upper()
        index_file = files("albatross").joinpath("data", f"{indicator}.txt")

        workdir = Path(self.workdir)

        # Target is only for training/hindcast in this step
        pr_input = request.inputs.get("target", [None])[0]
        clim_file = resolve_target_input(pr_input, workdir)

        # Output dirs
        maps_dir = workdir / f"{glo_var_name}_corr_maps"
        scat_dir = workdir / f"scatter_plots_{glo_var_name}"
        pc_dir = workdir / "pc_vs_hindcast"
        maps_dir.mkdir(parents=True, exist_ok=True)
        scat_dir.mkdir(parents=True, exist_ok=True)
        pc_dir.mkdir(parents=True, exist_ok=True)

        response.update_status("Loading training data...", 20)

        kwgroups = create_kwgroups(
            debug=True,
            climdata_months=months,
            climdata_startyr=startyr,
            n_yrs=endyr - startyr + 1,
            n_mon_glo_var=3,
            n_mon_index=3,
            glo_var_lag=3,
            n_phases=M,
            phases_even=True,
            index_lag=3,
            index_fp=index_file,
            climdata_fp=clim_file,
            glo_var_name=glo_var_name,
        )

        years = np.arange(startyr, endyr + 1)

        climdata, _, glo_var, index_avg, phaseind = get_data(
            kwgroups, glo_var_name=glo_var_name, workdir=workdir
        )

        # ✅ PHASE-AGNOSTIC: no special-casing for M=1; just use what get_data produced
        phases = list(phaseind.keys())

        # Save thresholds + per-phase masks using your utils (correct signature)
        phase_meta = _compute_phase_thresholds(index_avg=np.asarray(index_avg), phaseind=phaseind)
        _save_phase_artifacts(
            workdir=workdir,
            years=years,
            index_avg=np.asarray(index_avg),
            phaseind=phaseind,
            thresholds=phase_meta,
        )

        training_config = {
            "indicator": indicator,
            "glo_var_name": glo_var_name,
            "season_months": list(months),
            "n_mon_glo_var": kwgroups["glo_var"]["n_mon"],
            "glo_var_lag": kwgroups["glo_var"].get("lag", 3),
            "n_mon_index": kwgroups["index"]["n_mon"],
            "index_lag": kwgroups["index"].get("lag", 3),
            "n_phases": kwgroups["index"]["n_phases"],
            "phases_even": kwgroups["index"]["phases_even"],
        }
        (pc_dir / "training_config.json").write_text(json.dumps(training_config, indent=2))

        response.update_status("Training per-phase models...", 50)

        def run_one_phase(phase: str) -> dict:
            model = NIPAphase(climdata, glo_var_name, glo_var, index_avg, phaseind[phase])
            model.phase = phase
            model.years = years[phaseind[phase]]

            model.bootcorr(corrconf=0.95)
            model.gridCheck()
            model.crossvalpcr(xval=crv_flag)
            model.save_regressor(workdir)

            # Hindcast rows
            ts_rows = []
            if model.hindcast is not None:
                for k in range(len(model.years)):
                    ts_rows.append(
                        {
                            "year": int(model.years[k]),
                            "observed": float(model.clim_data[k]),
                            "hindcast": float(model.hindcast[k]),
                            "phase": phase,
                        }
                    )

            # Maps
            filt_pdf = maps_dir / f"corr_map_{phase}.pdf"
            filt_png = maps_dir / f"corr_map_{phase}.png"
            full_pdf = maps_dir / f"corr_map_full_{phase}.pdf"

            if map_flag:
                f, a, _ = glo_var_Map(model)
                a.set_title(f"{phase}, {model.correlation:.2f}" if model.correlation is not None else phase)
                f.savefig(filt_pdf)
                f.savefig(filt_png, dpi=150)
                plt.close(f)

                f2 = glo_var_Map_unfiltered(model)
                f2.savefig(full_pdf)
                plt.close(f2)

            # Scatter
            scatter_pdf = scat_dir / f"scatter_{phase}.pdf"
            if model.lin_model is not None:
                plot_model_results(model, scatter_pdf, crv_flag=crv_flag)

            # PCs rows
            pcs_rows = []
            append_model_pcs(model, pcs_rows)

            return {
                "phase": phase,
                "timeseries_rows": ts_rows,
                "pcs_rows": pcs_rows,
                "scatter_pdf": scatter_pdf,
                "filt_pdf": filt_pdf,
                "filt_png": filt_png,
                "full_pdf": full_pdf,
            }

        results = [run_one_phase(ph) for ph in phases]

        # Publish WPS outputs (keep consistent with your original expectations)
        response.outputs["scatter_plot"].file = results[0]["scatter_pdf"]
        response.outputs["pcs_plot"].file = results[0]["scatter_pdf"]

        if len(results) == 1:
            response.outputs["glo_var_map"].file = results[0]["filt_pdf"]
        else:
            combined_map_fp = maps_dir / "combined_glo_var_map.pdf"
            n = len(results)
            fig_cm, axs_cm = plt.subplots(1, n, figsize=(6 * n, 5))
            if n == 1:
                axs_cm = [axs_cm]
            for ax, r in zip(axs_cm, results):
                img = plt.imread(str(r["filt_png"]))
                ax.imshow(img)
                ax.set_title(r["phase"])
                ax.axis("off")
            plt.tight_layout()
            fig_cm.savefig(combined_map_fp)
            plt.close(fig_cm)
            response.outputs["glo_var_map"].file = combined_map_fp

        # Build hindcast tables
        timeseries_rows, pcs_file_rows = [], []
        for r in results:
            timeseries_rows.extend(r["timeseries_rows"])
            pcs_file_rows.extend(r["pcs_rows"])

        ts_file = workdir / f"{glo_var_name}_timeseries.csv"
        pd.DataFrame(timeseries_rows).sort_values("year").to_csv(ts_file, index=False)
        response.outputs["forecast_file"].file = ts_file

        pcs_csv_fp = pc_dir / "pcs_vs_hindcast.csv"
        pd.DataFrame(pcs_file_rows).to_csv(pcs_csv_fp, index=False)

        from albatross.utils import (
            predictors_available,
            latest_valid_index_value,
            pick_phase_from_thresholds,
            apply_saved_pcr,
        )

        # -------------------------
        # OPERATIONAL FORECAST
        # -------------------------
        forecast_status_fp = pc_dir / "operational_status.json"
        forecast_out_fp = pc_dir / "operational_forecast.csv"

        fy_in = request.inputs.get("forecast_year")
        if fy_in:
            forecast_year = int(fy_in [ 0 ].data)
        else:
            forecast_year = None

        op_status = {
            "forecast_attempted": bool(forecast_year is not None),
            "forecast_generated": False,
            "forecast_year": forecast_year,
            "season_months": ",".join(map(str, months)),
            "reason": "",
            "forecast_file": "",
        }

        if forecast_year is not None:
            try:
                dbg(f"OP: starting forecast for year={forecast_year}, months={months}, glo_var={glo_var_name}, indicator={indicator}, M={M}")
                dbg(f"OP: workdir={workdir}")
                dbg(f"OP: pc_dir={pc_dir}")
                dbg(f"OP: pc_dir contents (pc_transform): {[ p.name for p in pc_dir.glob('pc_transform_*.npz') ]}")

                dbg("OP: building kw_fc")
                kw_fc = create_kwgroups(
                    debug=True,
                    climdata_months=months,
                    # CHANGE 1: Start 1 year earlier
                    climdata_startyr=forecast_year - 1,
                    # CHANGE 2: Request 2 years (Padding)
                    n_yrs=2,
                    n_mon_glo_var=kwgroups [ "glo_var" ] [ "n_mon" ],
                    n_mon_index=kwgroups [ "index" ] [ "n_mon" ],
                    glo_var_lag=kwgroups [ "glo_var" ].get("lag", 3),
                    n_phases=M,
                    phases_even=kwgroups [ "index" ] [ "phases_even" ],
                    index_lag=kwgroups [ "index" ].get("lag", 3),
                    index_fp=index_file,
                    climdata_fp=clim_file,
                    glo_var_name=glo_var_name,
                )
                dbg(f"OP: kw_fc.glo_var={kw_fc [ 'glo_var' ]}")
                dbg(f"OP: kw_fc.index={kw_fc [ 'index' ]}")
                dbg("OP: calling get_data(load_target=False)")

                clim_none, gv_name2, glo_var_fc, index_fc, phaseind_fc = get_data(
                    kw_fc,
                    glo_var_name=glo_var_name,
                    workdir=workdir,
                    use_cache=True,
                    load_target=False,
                )
                dbg(f"OP: get_data returned gv_name2={gv_name2}, clim_none={clim_none is None}")
                dbg(f"OP: glo_var_fc.data shape={np.asarray(glo_var_fc.data).shape}")
                dbg(f"OP: index_fc type={type(index_fc)}, shape={np.asarray(index_fc).shape}")
                dbg(f"OP: phaseind_fc keys={list(phaseind_fc.keys())}")

                dbg("OP: running predictors_available")
                predictors_available(glo_var_fc, index_fc)
                dbg("OP: predictors_available OK")

                idx_val = float(np.asarray(index_fc).ravel() [ -1 ])
                dbg(f"OP: idx_val(latest)={idx_val}")

                dbg("OP: reading phase_thresholds.json")
                meta = json.loads((pc_dir / "phase_thresholds.json").read_text())
                phase = "allyears" if M==1 else pick_phase_from_thresholds(idx_val, meta)
                dbg(f"OP: selected phase={phase}")

                expected_npz = pc_dir / f"pc_transform_{phase}.npz"
                dbg(f"OP: expecting NPZ={expected_npz} exists={expected_npz.exists()}")

                field = np.asarray(glo_var_fc.data)
                dbg(f"OP: raw field ndim={field.ndim}, shape={field.shape}")

                # CHANGE 3: Select the LAST year (-1), not the first (0)
                if field.ndim==3:
                    field = field [ -1 ]

                dbg(f"OP: field_2d shape={field.shape}")

                dbg("OP: calling apply_saved_pcr")
                yhat = apply_saved_pcr(workdir=workdir, phase=phase, field_2d=field)
                dbg(f"OP: yhat={yhat}")

                dbg("OP: writing operational_forecast.csv")
                pd.DataFrame([ {
                    "forecast_year": int(forecast_year),
                    "season_months": ",".join(map(str, months)),
                    "indicator": indicator,
                    "indicator_value": float(idx_val),
                    "phase": phase,
                    "forecast": float(yhat),
                } ]).to_csv(forecast_out_fp, index=False)

                dbg(f"OP: wrote {forecast_out_fp} (exists={forecast_out_fp.exists()})")

                op_status [ "forecast_generated" ] = True
                op_status [ "forecast_file" ] = forecast_out_fp.name

            except Exception as e:
                LOGGER.exception("OP: failed with exception")  # <-- prints traceback
                op_status [ "reason" ] = repr(e)
                dbg(f"OP: FAILED reason={repr(e)}")
                # For debugging, you may also want to raise to fail the whole process:
                raise

        forecast_status_fp.write_text(json.dumps(op_status, indent=2))

        response.update_status("Packaging outputs...", 90)

        zip_path = workdir / f"outputs_{indicator}.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            pack = [ ]

            # core tables
            pack += [ ts_file, pcs_csv_fp ]

            # per-phase plots
            for r in results:
                pack += [ r [ "scatter_pdf" ], r [ "filt_pdf" ], r [ "full_pdf" ] ]

            # shared artifacts
            for shared in ("phase_thresholds.json", "index_training.csv", "training_config.json"):
                pack.append(pc_dir / shared)

            # regressor artifacts per phase
            for ph in phases:
                for fname in (
                    f"eofs_{ph}.csv",
                    f"coefficients_{ph}.csv",
                    f"glo_var_mask_{ph}.npy",
                    f"phase_mask_{ph}.csv",
                    f"pc_transform_{ph}.npz",
                ):
                    pack.append(pc_dir / fname)

            # operational artifacts
            pack += [ forecast_status_fp, forecast_out_fp ]

            # write unique existing files
            seen = set()
            for p in pack:
                p = Path(p)
                if p.exists() and p not in seen:
                    zipf.write(p, arcname=p.name)
                    seen.add(p)

        # Optional: copy to Desktop (best effort; keep your original behavior)
        try:
            desktop_path = Path.home() / "Desktop"
            target_file = extract_target_name(clim_file)
            months_string = ",".join(map(str, months))
            out_folder = desktop_path / f"{target_file}_{startyr}_{endyr}" / f"{target_file}_{indicator}_{glo_var_name}_{M}_{months_string}_{startyr}_{endyr}"
            out_folder.mkdir(parents=True, exist_ok=True)

            to_copy = [ts_file, pcs_csv_fp, zip_path]
            for out_id in ("scatter_plot", "glo_var_map"):
                f = getattr(response.outputs[out_id], "file", None)
                if f and Path(f).exists():
                    to_copy.append(Path(f))

            for src in to_copy:
                src = Path(src)
                if src.exists():
                    shutil.copy(src, out_folder / src.name)

            # copy pc_vs_hindcast folder
            dst_dir = out_folder / "pc_vs_hindcast"
            if pc_dir.exists():
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(pc_dir, dst_dir)

        except Exception as e:
            LOGGER.warning(f"Could not copy to Desktop: {e}")

        response.update_status("Drought process completed successfully!", 100)
        return response
