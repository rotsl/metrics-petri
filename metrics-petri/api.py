from __future__ import annotations

import base64
import json
import mimetypes
import os
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv

from .assets import assets_exist, get_assets_dir
from pipeline.reporting import generate_report

ROOT_DIR = Path.cwd()
INPUT_IMAGES_DIR = ROOT_DIR / "input_images"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ARCHIVES_DIR = ROOT_DIR / "archives"
ARCHIVED_REPORTS_DIR = ARCHIVES_DIR / "quarto_reports"
MODEL_DIR = ROOT_DIR / "models"

_JOB_LOCK = threading.RLock()
_ANALYSIS_JOBS: dict[str, "AnalysisJob"] = {}
_JOB_PROCESSES: dict[str, subprocess.Popen[str]] = {}
_SHUTTING_DOWN = False


@dataclass
class AnalysisJob:
    id: str
    engine: str
    filenames: list[str]
    experiment_name: str = ""
    tags: list[str] = field(default_factory=list)
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    progress: dict[str, Any] = field(
        default_factory=lambda: {"current": 0, "total": 0, "stage": "Queued"}
    )
    logs: list[str] = field(default_factory=list)
    pid: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "engine": self.engine,
            "filenames": self.filenames,
            "experimentName": self.experiment_name,
            "tags": self.tags,
            "status": self.status,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "progress": self.progress,
            "logs": self.logs,
            "pid": self.pid,
            "result": self.result,
            "error": self.error,
        }


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _slugify_segment(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")[:60]


def _normalize_tags(tags: Any) -> list[str]:
    if not isinstance(tags, list):
        return []
    normalized = [str(tag).strip() for tag in tags if isinstance(tag, str) and str(tag).strip()]
    return list(dict.fromkeys(normalized))


def _is_image_file(filename: str) -> bool:
    return bool(re.search(r"\.(png|jpe?g|bmp|tif?f|webp)$", filename, re.I))


def _extract_explicit_day(filename: str) -> int | None:
    match = re.search(r"(?:day|d)(\d+)", filename, re.I)
    return int(match.group(1)) if match else None


def _infer_image_days(filenames: list[str]) -> dict[str, int]:
    explicit: dict[str, int] = {}
    inferred: list[str] = []
    for filename in filenames:
        day = _extract_explicit_day(filename)
        if day is None:
            inferred.append(filename)
        else:
            explicit[filename] = day

    used_days = set(explicit.values())
    next_day = 1
    for filename in inferred:
        while next_day in used_days:
            next_day += 1
        explicit[filename] = next_day
        used_days.add(next_day)
        next_day += 1
    return explicit


def _sanitize_filename(filename: str) -> str:
    parsed = Path(filename)
    safe_base = re.sub(r"[^a-zA-Z0-9._-]+", "_", parsed.stem).strip("_") or "image"
    return f"{safe_base}{parsed.suffix.lower()}"


def _resolve_unique_upload_path(filename: str) -> Path:
    safe_name = _sanitize_filename(filename)
    candidate = INPUT_IMAGES_DIR / safe_name
    index = 1
    while candidate.exists():
        parsed = candidate.with_suffix("")
        candidate = INPUT_IMAGES_DIR / f"{parsed.name}_{index}{Path(safe_name).suffix}"
        index += 1
    return candidate


def _read_request_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    body = handler.rfile.read(length)
    if not body:
        return {}
    payload = json.loads(body.decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _send_json(handler: SimpleHTTPRequestHandler, payload: Any, status: int = HTTPStatus.OK) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_error(handler: SimpleHTTPRequestHandler, status: int, message: str) -> None:
    _send_json(handler, {"error": message}, status=status)


def _serve_file(handler: SimpleHTTPRequestHandler, base_dir: Path, relative: str) -> bool:
    candidate = (base_dir / relative.lstrip("/")).resolve()
    try:
        candidate.relative_to(base_dir.resolve())
    except ValueError:
        return False
    if not candidate.is_file():
        return False
    mime_type, _ = mimetypes.guess_type(candidate.name)
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", mime_type or "application/octet-stream")
    handler.send_header("Content-Length", str(candidate.stat().st_size))
    handler.end_headers()
    with candidate.open("rb") as stream:
        shutil.copyfileobj(stream, handler.wfile)
    return True


def _append_job_log(job: AnalysisJob, line: str) -> None:
    text = line.strip()
    if not text:
        return
    job.logs.append(text)
    if len(job.logs) > 300:
        job.logs = job.logs[-300:]
    job.updated_at = datetime.now(timezone.utc).isoformat()


def _get_job(job_id: str) -> AnalysisJob:
    try:
        return _ANALYSIS_JOBS[job_id]
    except KeyError as exc:
        raise KeyError(f"Analysis job {job_id} was not found.") from exc


def _active_job() -> AnalysisJob | None:
    active = [job for job in _ANALYSIS_JOBS.values() if job.status in {"queued", "running", "paused"}]
    return sorted(active, key=lambda job: job.created_at, reverse=True)[0] if active else None


def _write_run_metadata(run_dir: Path, experiment_name: str, tags: list[str]) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        return
    metadata = dict(manifest.get("metadata") or {})
    metadata["experiment_name"] = experiment_name
    metadata["tags"] = tags
    manifest["metadata"] = metadata
    _write_json(manifest_path, manifest)


def _delete_run_directory(run_id: str) -> bool:
    if not re.fullmatch(r"\d{8}T\d{6}Z_(local|gemini)", run_id):
        raise ValueError(f"Invalid output run id: {run_id}")
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        return False
    shutil.rmtree(run_dir, ignore_errors=True)
    return not run_dir.exists()


def _archive_run_directory(run_id: str) -> str | None:
    if not re.fullmatch(r"\d{8}T\d{6}Z_(local|gemini)", run_id):
        raise ValueError(f"Invalid output run id: {run_id}")
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        return None
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    archive_name = run_id
    archive_dir = ARCHIVES_DIR / archive_name
    if archive_dir.exists():
        archive_name = f"{run_id}_{int(datetime.now(timezone.utc).timestamp())}"
        archive_dir = ARCHIVES_DIR / archive_name
    shutil.move(str(run_dir), str(archive_dir))
    return archive_name


def _delete_quarto_report_bundle(run_id: str) -> bool:
    if not re.fullmatch(r"\d{8}T\d{6}Z_(local|gemini)", run_id):
        raise ValueError(f"Invalid output run id: {run_id}")
    report_dir = OUTPUTS_DIR / run_id / "quarto_report"
    if not report_dir.exists():
        return False
    shutil.rmtree(report_dir)
    return not report_dir.exists()


def _archive_quarto_report_bundle(run_id: str) -> str | bool:
    if not re.fullmatch(r"\d{8}T\d{6}Z_(local|gemini)", run_id):
        raise ValueError(f"Invalid output run id: {run_id}")
    report_dir = OUTPUTS_DIR / run_id / "quarto_report"
    if not report_dir.exists():
        return False
    ARCHIVED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(OUTPUTS_DIR / run_id / "manifest.json")
    experiment_name = ""
    if isinstance(manifest, dict):
        metadata = manifest.get("metadata")
        if isinstance(metadata, dict):
            experiment_name = str(metadata.get("experiment_name", "")).strip()
    experiment_slug = re.sub(r"[^a-z0-9]+", "-", experiment_name.lower()).strip("-")[:60] if experiment_name else ""
    archive_parent = ARCHIVED_REPORTS_DIR / "_".join(part for part in [run_id, experiment_slug] if part)
    archive_parent.mkdir(parents=True, exist_ok=True)
    archive_dir = archive_parent / "quarto_report"
    if archive_dir.exists():
        archive_dir = archive_parent / f"quarto_report_{int(datetime.now(timezone.utc).timestamp())}"
    shutil.move(str(report_dir), str(archive_dir))
    return str(archive_dir)


def _spawn_analysis_job(
    engine: str,
    filenames: list[str],
    experiment_name: str,
    tags: list[str],
    gemini_api_key: str,
) -> AnalysisJob:
    job_id = str(uuid.uuid4())
    job = AnalysisJob(
        id=job_id,
        engine=engine,
        filenames=filenames,
        experiment_name=experiment_name,
        tags=tags,
        progress={"current": 0, "total": len(filenames), "stage": "Queued"},
    )
    _ANALYSIS_JOBS[job_id] = job

    args = [
        "-m",
        "pipeline.cli",
        "--engine",
        engine,
        "--input-dir",
        str(INPUT_IMAGES_DIR),
        "--output-dir",
        str(OUTPUTS_DIR),
        "--json",
    ]
    for filename in filenames:
        args.extend(["--filename", filename])

    env = os.environ.copy()
    if gemini_api_key:
        env["GEMINI_API_KEY"] = gemini_api_key

    process = subprocess.Popen(
        [sys.executable, *args],
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    _JOB_PROCESSES[job_id] = process
    job.pid = process.pid
    job.status = "running"
    job.progress["stage"] = "Starting analysis"
    _append_job_log(job, f"[job] Starting {engine} analysis for {len(filenames)} image(s)")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _read_stream(stream, chunks: list[str], is_stderr: bool) -> None:
        assert stream is not None
        for line in stream:
            chunks.append(line)
            if is_stderr:
                _append_job_log(job, line)
                progress_match = re.match(r"^\[progress\]\s+(\d+)\/(\d+)\s+(.*)$", line.strip(), re.I)
                if progress_match:
                    job.progress = {
                        "current": int(progress_match.group(1)),
                        "total": int(progress_match.group(2)),
                        "stage": progress_match.group(3).strip(),
                    }

    stdout_thread = threading.Thread(target=_read_stream, args=(process.stdout, stdout_chunks, False), daemon=True)
    stderr_thread = threading.Thread(target=_read_stream, args=(process.stderr, stderr_chunks, True), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    def _finalize() -> None:
        exit_code = process.wait()
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        _JOB_PROCESSES.pop(job_id, None)

        stdout_text = "".join(stdout_chunks).strip()
        stderr_text = "".join(stderr_chunks).strip()
        if job.status == "stopped":
            job.progress["stage"] = "Stopped"
            _append_job_log(job, "[job] Analysis stopped by user")
        elif exit_code == 0:
            try:
                payload = json.loads(stdout_text) if stdout_text else {}
                if isinstance(payload, dict):
                    run = payload.get("run")
                    if isinstance(run, dict):
                        run_id = str(run.get("outputDir", "")).split("/")[-1]
                        if run_id:
                            merged_tags = list(dict.fromkeys([*tags, *("macbook" if sys.platform == "darwin" else [])]))
                            _write_run_metadata(OUTPUTS_DIR / run_id, experiment_name, merged_tags)
                            run["id"] = run_id
                            run["experimentName"] = experiment_name
                            run["tags"] = merged_tags
                            payload["run"] = run
                job.status = "completed"
                job.result = payload if isinstance(payload, dict) else {"run": payload}
                job.progress = {"current": job.progress.get("total", 0), "total": job.progress.get("total", 0), "stage": "Completed"}
                _append_job_log(job, "[job] Analysis completed successfully")
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.error = str(exc)
                _append_job_log(job, f"[job] {job.error}")
        else:
            job.status = "failed"
            job.error = stderr_text or f"Analysis failed with exit code {exit_code}"
            _append_job_log(job, f"[job] {job.error}")
        job.pid = None
        job.updated_at = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=_finalize, daemon=True).start()
    return job


def _list_images() -> list[dict[str, Any]]:
    INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    filenames = sorted(
        [entry.name for entry in INPUT_IMAGES_DIR.iterdir() if entry.is_file() and _is_image_file(entry.name)],
        key=str.lower,
    )
    day_map = _infer_image_days(filenames)
    return [
        {"filename": filename, "day": day_map.get(filename, 0), "imageUrl": f"/input_images/{filename}"}
        for filename in filenames
    ]


def _list_output_runs() -> list[dict[str, Any]]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for entry in sorted(OUTPUTS_DIR.iterdir(), key=lambda path: path.name, reverse=True):
        if not entry.is_dir() or not re.fullmatch(r"\d{8}T\d{6}Z_(local|gemini)", entry.name):
            continue
        manifest = _load_json(entry / "manifest.json")
        if not isinstance(manifest, dict):
            continue
        metadata = manifest.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        runs.append(
            {
                "id": entry.name,
                "engine": manifest.get("engine", "unknown"),
                "engineModel": manifest.get("engine_model", "unknown"),
                "createdAt": manifest.get("created_at", ""),
                "outputDir": manifest.get("output_dir", ""),
                "analysisJson": manifest.get("analysis_json", ""),
                "analysisCsv": manifest.get("analysis_csv", ""),
                "experimentName": metadata.get("experiment_name", ""),
                "tags": metadata.get("tags", []),
            }
        )
    return sorted(runs, key=lambda run: str(run.get("createdAt", "")), reverse=True)


def _list_quarto_reports() -> list[dict[str, Any]]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for entry in sorted(OUTPUTS_DIR.iterdir(), key=lambda path: path.name, reverse=True):
        if not entry.is_dir() or not re.fullmatch(r"\d{8}T\d{6}Z_(local|gemini)", entry.name):
            continue
        report_dir = entry / "quarto_report"
        if not report_dir.exists():
            continue
        bundle_path = report_dir / "report_bundle.json"
        bundle = _load_json(bundle_path)
        manifest = _load_json(entry / "manifest.json")
        metadata = {}
        if isinstance(manifest, dict):
            maybe_metadata = manifest.get("metadata")
            if isinstance(maybe_metadata, dict):
                metadata = maybe_metadata
        qmd_path = report_dir / "report.qmd"
        html_path = report_dir / "report.html"
        pdf_path = report_dir / "report.pdf"
        generated_at = str(bundle.get("generatedAt", "")) if isinstance(bundle, dict) else ""
        if not generated_at and bundle_path.exists():
            generated_at = datetime.fromtimestamp(bundle_path.stat().st_mtime, timezone.utc).isoformat()
        if not generated_at:
            generated_at = datetime.fromtimestamp(report_dir.stat().st_mtime, timezone.utc).isoformat()
        reports.append(
            {
                "runId": str(bundle.get("runId", entry.name)) if isinstance(bundle, dict) else entry.name,
                "template": str(bundle.get("template", "biology_report_v1")) if isinstance(bundle, dict) else "biology_report_v1",
                "generatedAt": generated_at,
                "assetsDir": str(bundle.get("assetsDir", f"/outputs/{entry.name}/report_assets")) if isinstance(bundle, dict) else f"/outputs/{entry.name}/report_assets",
                "graphCount": int(bundle.get("graphCount", 0) or 0) if isinstance(bundle, dict) else 0,
                "experimentName": str(bundle.get("experimentName", metadata.get("experiment_name", ""))) if isinstance(bundle, dict) else str(metadata.get("experiment_name", "")),
                "tags": bundle.get("tags", []) if isinstance(bundle, dict) and isinstance(bundle.get("tags", []), list) else [str(tag) for tag in metadata.get("tags", []) if isinstance(tag, str)] if isinstance(metadata, dict) else [],
                "qmdPath": str(bundle.get("qmdPath", f"/outputs/{entry.name}/quarto_report/{qmd_path.name}" if qmd_path.exists() else "")) if isinstance(bundle, dict) else (f"/outputs/{entry.name}/quarto_report/{qmd_path.name}" if qmd_path.exists() else ""),
                "quartoHtmlPath": str(bundle.get("quartoHtmlPath", f"/outputs/{entry.name}/quarto_report/{html_path.name}" if html_path.exists() else "")) if isinstance(bundle, dict) else (f"/outputs/{entry.name}/quarto_report/{html_path.name}" if html_path.exists() else ""),
                "quartoPdfPath": str(bundle.get("quartoPdfPath", f"/outputs/{entry.name}/quarto_report/{pdf_path.name}" if pdf_path.exists() else "")) if isinstance(bundle, dict) else (f"/outputs/{entry.name}/quarto_report/{pdf_path.name}" if pdf_path.exists() else ""),
                "quartoStatus": str(bundle.get("quartoStatus", "skipped")) if isinstance(bundle, dict) else "skipped",
                "quartoError": bundle.get("quartoError", None) if isinstance(bundle, dict) else None,
                "quartoContent": "",
            }
        )
    return sorted(reports, key=lambda report: str(report.get("generatedAt", "")), reverse=True)


def _route_api_get(handler: SimpleHTTPRequestHandler, path: str) -> bool:
    if path == "/api/images":
        _send_json(handler, {"images": _list_images()})
        return True
    if path == "/api/outputs":
        _send_json(handler, {"runs": _list_output_runs()})
        return True
    if path == "/api/reports":
        _send_json(handler, {"reports": _list_quarto_reports()})
        return True
    if path == "/api/analyze/active":
        active = _active_job()
        if active is None:
            _send_error(handler, HTTPStatus.NOT_FOUND, "No active analysis job was found.")
        else:
            _send_json(handler, active.to_dict())
        return True
    if path.startswith("/api/analyze/"):
        job_id = path.split("/")[3]
        job = _ANALYSIS_JOBS.get(job_id)
        if job is None:
            _send_error(handler, HTTPStatus.NOT_FOUND, f"Analysis job {job_id} was not found.")
        else:
            _send_json(handler, job.to_dict())
        return True
    if path.startswith("/api/outputs/"):
        parts = path.split("/")
        if len(parts) == 4:
            run_id = parts[3]
            if not re.fullmatch(r"\d{8}T\d{6}Z_(local|gemini)", run_id):
                _send_error(handler, HTTPStatus.BAD_REQUEST, "Invalid output run id")
                return True
            run_dir = OUTPUTS_DIR / run_id
            manifest = _load_json(run_dir / "manifest.json")
            results = _load_json(run_dir / "analysis.json")
            if not isinstance(manifest, dict) or not isinstance(results, list):
                _send_error(handler, HTTPStatus.NOT_FOUND, f"Output run {run_id} was not found.")
                return True
            metadata = manifest.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            _send_json(
                handler,
                {
                    "run": {
                        "id": run_id,
                        "engine": manifest.get("engine", "unknown"),
                        "engineModel": manifest.get("engine_model", "unknown"),
                        "createdAt": manifest.get("created_at", ""),
                        "outputDir": manifest.get("output_dir", ""),
                        "analysisJson": manifest.get("analysis_json", ""),
                        "analysisCsv": manifest.get("analysis_csv", ""),
                        "experimentName": metadata.get("experiment_name", ""),
                        "tags": metadata.get("tags", []),
                    },
                    "results": results,
                },
            )
            return True
    return False


def _route_api_post(handler: SimpleHTTPRequestHandler, path: str) -> bool:
    payload = _read_request_json(handler)
    if path == "/api/images/upload":
        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        uploads: list[dict[str, str]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            content_base64 = item.get("contentBase64")
            if not isinstance(name, str) or not isinstance(content_base64, str):
                continue
            if not _is_image_file(name):
                _send_error(handler, HTTPStatus.BAD_REQUEST, f"Unsupported image type for {name}")
                return True
            destination = _resolve_unique_upload_path(name)
            destination.write_bytes(base64.b64decode(content_base64))
            uploads.append({"filename": destination.name, "imageUrl": f"/input_images/{destination.name}"})
        if not uploads:
            _send_error(handler, HTTPStatus.BAD_REQUEST, "files must contain at least one image payload")
        else:
            _send_json(handler, {"uploaded": uploads})
        return True

    if path == "/api/analyze":
        engine = payload.get("engine")
        filenames = [str(item) for item in payload.get("filenames", []) if isinstance(item, str)]
        experiment_name = str(payload.get("experimentName", "")).strip()
        tags = _normalize_tags(payload.get("tags"))
        gemini_api_key = str(payload.get("geminiApiKey", "")).strip()
        if engine not in {"local", "gemini"}:
            _send_error(handler, HTTPStatus.BAD_REQUEST, 'engine must be "local" or "gemini"')
            return True
        job = _spawn_analysis_job(engine, filenames, experiment_name, tags, gemini_api_key)
        _send_json(handler, {"jobId": job.id, "status": job.status, "progress": job.progress}, status=HTTPStatus.ACCEPTED)
        return True

    if path == "/api/outputs/delete":
        run_ids = [str(item) for item in payload.get("runIds", []) if isinstance(item, str)]
        if not run_ids:
            _send_error(handler, HTTPStatus.BAD_REQUEST, "runIds must contain at least one output run id")
            return True
        deleted = [run_id for run_id in run_ids if _delete_run_directory(run_id)]
        _send_json(handler, {"deleted": deleted})
        return True

    if path == "/api/outputs/archive":
        run_ids = [str(item) for item in payload.get("runIds", []) if isinstance(item, str)]
        if not run_ids:
            _send_error(handler, HTTPStatus.BAD_REQUEST, "runIds must contain at least one output run id")
            return True
        archived: list[str] = []
        archived_as: dict[str, str] = {}
        for run_id in run_ids:
            archive_name = _archive_run_directory(run_id)
            if archive_name:
                archived.append(run_id)
                archived_as[run_id] = archive_name
        _send_json(handler, {"archived": archived, "archivedAs": archived_as})
        return True

    if path == "/api/reports/delete":
        run_ids = [str(item) for item in payload.get("runIds", []) if isinstance(item, str)]
        if not run_ids:
            _send_error(handler, HTTPStatus.BAD_REQUEST, "runIds must contain at least one Quarto report id")
            return True
        deleted = [run_id for run_id in run_ids if _delete_quarto_report_bundle(run_id)]
        _send_json(handler, {"deleted": deleted})
        return True

    if path == "/api/reports/archive":
        run_ids = [str(item) for item in payload.get("runIds", []) if isinstance(item, str)]
        if not run_ids:
            _send_error(handler, HTTPStatus.BAD_REQUEST, "runIds must contain at least one Quarto report id")
            return True
        archived: list[str] = []
        archived_as: dict[str, str] = {}
        for run_id in run_ids:
            archive_name = _archive_quarto_report_bundle(run_id)
            if archive_name:
                archived.append(run_id)
                archived_as[run_id] = archive_name
        _send_json(handler, {"archived": archived, "archivedAs": archived_as})
        return True

    if path.startswith("/api/outputs/") and path.endswith("/report"):
        run_id = path.split("/")[3]
        run_dir = OUTPUTS_DIR / run_id
        if not (run_dir / "manifest.json").exists() or not (run_dir / "analysis.json").exists():
            _send_error(handler, HTTPStatus.NOT_FOUND, f"Output run {run_id} was not found.")
            return True
        experiment_name = str(payload.get("experimentName", "")).strip()
        tags = _normalize_tags(payload.get("tags"))
        try:
            existing_manifest = _load_json(run_dir / "manifest.json")
            existing_tags = []
            if isinstance(existing_manifest, dict):
                metadata = existing_manifest.get("metadata") or {}
                if isinstance(metadata, dict):
                    existing_tags = [str(tag) for tag in metadata.get("tags", []) if isinstance(tag, str)]
            merged_tags = list(dict.fromkeys([*existing_tags, *tags]))
            _write_run_metadata(run_dir, experiment_name, merged_tags)
            report = generate_report(run_dir, experiment_override=experiment_name, tags_override=merged_tags)
            _send_json(handler, report)
        except Exception as exc:  # noqa: BLE001
            _send_error(handler, HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        return True

    if path.startswith("/api/reports/") and path.endswith("/archive"):
        run_id = path.split("/")[3]
        try:
            archived_as = _archive_quarto_report_bundle(run_id)
            if not archived_as:
                _send_error(handler, HTTPStatus.NOT_FOUND, f"Quarto report for {run_id} was not found.")
                return True
            _send_json(handler, {"archived": [run_id], "archivedAs": {run_id: archived_as}})
        except Exception as exc:  # noqa: BLE001
            _send_error(handler, HTTPStatus.BAD_REQUEST, str(exc))
        return True

    if path.startswith("/api/analyze/") and path.endswith("/pause"):
        job_id = path.split("/")[3]
        try:
            job = _get_job(job_id)
            if job.status != "running":
                raise ValueError(f"Only running jobs can be paused. Current status: {job.status}")
            process = _JOB_PROCESSES.get(job_id)
            if process is None or process.pid is None:
                raise ValueError(f"Analysis job {job_id} is not attached to a running process.")
            os.kill(process.pid, signal.SIGSTOP)
            job.status = "paused"
            job.progress["stage"] = "Paused"
            _append_job_log(job, "[job] Analysis paused by user")
            _send_json(handler, job.to_dict())
        except Exception as exc:  # noqa: BLE001
            _send_error(handler, HTTPStatus.BAD_REQUEST, str(exc))
        return True

    if path.startswith("/api/analyze/") and path.endswith("/resume"):
        job_id = path.split("/")[3]
        try:
            job = _get_job(job_id)
            if job.status != "paused":
                raise ValueError(f"Only paused jobs can be resumed. Current status: {job.status}")
            process = _JOB_PROCESSES.get(job_id)
            if process is None or process.pid is None:
                raise ValueError(f"Analysis job {job_id} is not attached to a running process.")
            os.kill(process.pid, signal.SIGCONT)
            job.status = "running"
            job.progress["stage"] = "Resuming analysis"
            _append_job_log(job, "[job] Analysis resumed by user")
            _send_json(handler, job.to_dict())
        except Exception as exc:  # noqa: BLE001
            _send_error(handler, HTTPStatus.BAD_REQUEST, str(exc))
        return True

    if path.startswith("/api/analyze/") and path.endswith("/stop"):
        job_id = path.split("/")[3]
        try:
            job = _get_job(job_id)
            if job.status not in {"queued", "running", "paused"}:
                raise ValueError(f"Only queued, running, or paused jobs can be stopped. Current status: {job.status}")
            process = _JOB_PROCESSES.get(job_id)
            if process is None or process.pid is None:
                raise ValueError(f"Analysis job {job_id} is not attached to a running process.")
            job.status = "stopped"
            job.progress["stage"] = "Stopping analysis"
            _append_job_log(job, "[job] Stop requested by user")
            os.kill(process.pid, signal.SIGTERM)
            _send_json(handler, job.to_dict())
        except Exception as exc:  # noqa: BLE001
            _send_error(handler, HTTPStatus.BAD_REQUEST, str(exc))
        return True

    if path == "/api/server/stop":
        restart_command = "metrics-petri"
        _send_json(
            handler,
            {
                "stoppedPort": handler.server.server_address[1],
                "stoppedPorts": [handler.server.server_address[1]],
                "restartCommand": restart_command,
                "message": f"GUI and API are shutting down. Restart with: {restart_command}",
            },
        )

        global _SHUTTING_DOWN
        if not _SHUTTING_DOWN:
            _SHUTTING_DOWN = True

            def _shutdown() -> None:
                handler.server.shutdown()

            threading.Thread(target=_shutdown, daemon=True).start()
        return True

    return False


def _route_api_delete(handler: SimpleHTTPRequestHandler, path: str) -> bool:
    if path.startswith("/api/outputs/"):
        run_id = path.split("/")[3]
        try:
            deleted = _delete_run_directory(run_id)
            if not deleted:
                _send_error(handler, HTTPStatus.NOT_FOUND, f"Output run {run_id} was not found.")
            else:
                _send_json(handler, {"deleted": [run_id]})
        except Exception as exc:  # noqa: BLE001
            _send_error(handler, HTTPStatus.BAD_REQUEST, str(exc))
        return True
    return False


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    """Create the combined static + API server used by the packaged app."""

    load_dotenv(ROOT_DIR / ".env")
    INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)

    assets_dir = get_assets_dir()
    if not assets_exist():
        raise FileNotFoundError(f"Frontend assets not found at {assets_dir}. Build the frontend before packaging.")

    class MetricsPetriHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(assets_dir), **kwargs)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _path(self) -> str:
            return unquote(urlsplit(self.path).path)

        def _serve_workspace_file(self, base_dir: Path, relative: str) -> bool:
            return _serve_file(self, base_dir, relative)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = self._path()
            if path.startswith("/api/"):
                if _route_api_get(self, path):
                    return
                _send_error(self, HTTPStatus.NOT_FOUND, f"Unknown API route: {path}")
                return
            if path.startswith("/input_images/"):
                if self._serve_workspace_file(INPUT_IMAGES_DIR, path.removeprefix("/input_images/")):
                    return
                _send_error(self, HTTPStatus.NOT_FOUND, f"Image not found: {path}")
                return
            if path.startswith("/outputs/"):
                if self._serve_workspace_file(OUTPUTS_DIR, path.removeprefix("/outputs/")):
                    return
                _send_error(self, HTTPStatus.NOT_FOUND, f"Output not found: {path}")
                return
            if path.startswith("/archives/"):
                if self._serve_workspace_file(ARCHIVES_DIR, path.removeprefix("/archives/")):
                    return
                _send_error(self, HTTPStatus.NOT_FOUND, f"Archive not found: {path}")
                return
            if path == "/" or "." not in posixpath.basename(path):
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            path = self._path()
            if path.startswith("/api/") and _route_api_post(self, path):
                return
            _send_error(self, HTTPStatus.NOT_FOUND, f"Unknown API route: {path}")

        def do_DELETE(self) -> None:  # noqa: N802
            path = self._path()
            if path.startswith("/api/") and _route_api_delete(self, path):
                return
            _send_error(self, HTTPStatus.NOT_FOUND, f"Unknown API route: {path}")

    return ThreadingHTTPServer((host, port), MetricsPetriHandler)
