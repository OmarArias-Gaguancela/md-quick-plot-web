import os
import uuid
import shutil
import base64
import zipfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="MD Quick Plot")

BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# In-memory job store
jobs: dict = {}
jobs_lock = threading.Lock()


def _update_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_analysis(job_id: str, job_dir: Path, topology: str, trajectory: str,
                 protein_sel: str, ligand_sel: str, dt_in_ps: Optional[float],
                 analyses: list, temperature: int, reference_frame: int):
    try:
        from analyzer import MDAnalyzer

        _update_job(job_id, status="running", progress=5, message="Loading trajectory...")

        analyzer = MDAnalyzer(
            topology=topology,
            trajectory=trajectory,
            protein_selection=protein_sel,
            ligand_selection=ligand_sel,
            dt_in_ps=dt_in_ps if dt_in_ps and dt_in_ps > 0 else None,
        )

        info = analyzer.info()
        _update_job(job_id, info=info)

        results = {}
        total = len(analyses)
        step_size = 85 // max(total, 1)
        current_progress = 10

        analysis_map = {
            "rmsd": ("RMSD", lambda: analyzer.plot_rmsd(str(job_dir / "rmsd.png"))),
            "rmsf": ("RMSF", lambda: analyzer.plot_rmsf(str(job_dir / "rmsf.png"))),
            "rg":   ("Radius of Gyration", lambda: analyzer.plot_rg(str(job_dir / "rg.png"))),
            "fel":  ("Free Energy Landscape", lambda: analyzer.plot_free_energy_landscape(
                         str(job_dir / "fel.png"), temperature=temperature)),
            "binding": ("Binding Energy", lambda: analyzer.plot_binding_energy(
                            str(job_dir / "binding_energy.png"))),
            "distance": ("P-L Distance", lambda: analyzer.plot_protein_ligand_distance(
                             str(job_dir / "pl_distance.png"))),
        }

        for key in analyses:
            if key not in analysis_map:
                continue
            label, fn = analysis_map[key]

            if key in ("binding", "distance") and not info["has_ligand"]:
                results[key] = {"skipped": True, "reason": "No ligand detected"}
                current_progress += step_size
                continue

            _update_job(job_id, progress=current_progress, message=f"Running {label}...")
            stat = fn()
            if stat is None:
                results[key] = {"skipped": True, "reason": "No ligand detected"}
            else:
                plot_path = stat.pop("plot")
                stat["image"] = _img_to_b64(plot_path)
                results[key] = stat

            current_progress += step_size

        _update_job(job_id, status="done", progress=100,
                    message="Analysis complete!", results=results)

    except Exception as exc:
        _update_job(job_id, status="error", message=str(exc))


# ─────────────────────────── Routes ───────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    topology: UploadFile = File(...),
    trajectory: UploadFile = File(...),
    protein_sel: str = Form("protein"),
    ligand_sel: str = Form("resname LIG"),
    dt_in_ps: float = Form(0.0),
    analyses: str = Form("rmsd,rmsf,rg,fel,binding,distance"),
    temperature: int = Form(300),
    reference_frame: int = Form(0),
):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    # Save uploaded files
    topo_path = job_dir / topology.filename
    traj_path = job_dir / trajectory.filename

    with open(topo_path, "wb") as f:
        shutil.copyfileobj(topology.file, f)
    with open(traj_path, "wb") as f:
        shutil.copyfileobj(trajectory.file, f)

    selected = [a.strip() for a in analyses.split(",") if a.strip()]

    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "message": "Queued...",
                        "results": {}, "info": {}}

    background_tasks.add_task(
        run_analysis, job_id, job_dir,
        str(topo_path), str(traj_path),
        protein_sel, ligand_sel, dt_in_ps,
        selected, temperature, reference_frame,
    )

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job


@app.get("/download/{job_id}")
async def download(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        return JSONResponse({"error": "Job not found"}, status_code=404)

    zip_path = JOBS_DIR / f"{job_id}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for png in job_dir.glob("*.png"):
            zf.write(png, png.name)

    return FileResponse(str(zip_path), media_type="application/zip",
                        filename="md_quick_plot_results.zip")


@app.post("/extract-frame")
async def extract_frame(
    background_tasks: BackgroundTasks,
    topology: UploadFile = File(...),
    trajectory: UploadFile = File(...),
    time_ns: float = Form(...),
    selection: str = Form("protein or resname LIG"),
    output_name: str = Form("extracted_frame.pdb"),
):
    from analyzer import extract_frame_from_trajectory

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    topo_path = job_dir / topology.filename
    traj_path = job_dir / trajectory.filename
    with open(topo_path, "wb") as f:
        shutil.copyfileobj(topology.file, f)
    with open(traj_path, "wb") as f:
        shutil.copyfileobj(trajectory.file, f)

    if not output_name.endswith(".pdb"):
        output_name += ".pdb"
    out_path = job_dir / output_name

    try:
        frame_idx, actual_time = extract_frame_from_trajectory(
            str(topo_path), str(traj_path), time_ns,
            output_pdb=str(out_path), selection=selection,
        )
        return FileResponse(
            str(out_path), media_type="chemical/x-pdb",
            filename=output_name,
            headers={"X-Frame-Index": str(frame_idx),
                     "X-Actual-Time-NS": str(actual_time)},
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
