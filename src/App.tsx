import React, {useEffect, useState} from 'react';
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  CirclePower,
  SquareTerminal,
  Download,
  FileText,
  FolderOpen,
  Image as ImageIcon,
  Loader2,
  Orbit,
  Pause,
  Play,
  RefreshCcw,
  Square,
  Upload,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import {AnimatePresence, motion} from 'motion/react';
import {cn} from './lib/utils';
import {
  archiveOutputRuns,
  deleteOutputRuns,
  generateOutputReport,
  getAnalysisJob,
  getLatestActiveAnalysisJob,
  getOutputRun,
  listInputImages,
  listOutputRuns,
  pauseAnalysisJob,
  resumeAnalysisJob,
  startAnalysis,
  stopAnalysisJob,
  stopLocalApp,
  uploadInputImages,
} from './services/analysisService';
import type {
  AnalysisEngine,
  AnalysisJob,
  AnalysisResult,
  AnalysisRun,
  GrowthData,
  InputImage,
  ReportBundle,
} from './types';
import {AnalysisOverlay} from './components/AnalysisOverlay';

export default function App() {
  const GEMINI_KEY_STORAGE = 'grayleafspot.geminiApiKey';
  const GEMINI_KEY_SAVE_STORAGE = 'grayleafspot.saveGeminiApiKey';
  const ACTIVE_JOB_STORAGE = 'grayleafspot.activeAnalysisJobId';
  const EXPERIMENT_NAME_STORAGE = 'grayleafspot.experimentName';
  const [images, setImages] = useState<InputImage[]>([]);
  const [outputRuns, setOutputRuns] = useState<AnalysisRun[]>([]);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [engine, setEngine] = useState<AnalysisEngine>('local');
  const [analyzing, setAnalyzing] = useState(false);
  const [loadingImages, setLoadingImages] = useState(false);
  const [loadingOutputs, setLoadingOutputs] = useState(false);
  const [loadingRunId, setLoadingRunId] = useState<string | null>(null);
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const [latestRun, setLatestRun] = useState<AnalysisRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisJob, setAnalysisJob] = useState<AnalysisJob | null>(null);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [jobActionPending, setJobActionPending] = useState<'pause' | 'resume' | 'stop' | 'port' | null>(null);
  const [reportBundle, setReportBundle] = useState<ReportBundle | null>(null);
  const [reportGenerating, setReportGenerating] = useState(false);
  const [experimentName, setExperimentName] = useState('');
  const [geminiApiKey, setGeminiApiKey] = useState('');
  const [saveGeminiApiKey, setSaveGeminiApiKey] = useState(false);
  const [uploadingImages, setUploadingImages] = useState(false);
  const [overlayConfig, setOverlayConfig] = useState({
    dish: true,
    mask: true,
    cracks: true,
    roi: true,
  });
  const frontendOrigin = window.location.origin;
  const expectedLocalFrontend =
    window.location.port === '3000' && ['127.0.0.1', 'localhost'].includes(window.location.hostname);
  const apiConnectionLabel =
    expectedLocalFrontend
      ? 'http://127.0.0.1:8000 via Vite proxy'
      : `${window.location.origin}/api`;
  const backendPortLabel = apiConnectionLabel.match(/:(\d+)/)?.[1] ?? '8000';
  const connectionStatusLabel = expectedLocalFrontend ? 'Connected' : 'Mismatch suspected';

  useEffect(() => {
    void refreshImages();
    void refreshOutputs();
  }, []);

  useEffect(() => {
    const storedKey = window.localStorage.getItem(GEMINI_KEY_STORAGE) ?? '';
    const shouldSave = window.localStorage.getItem(GEMINI_KEY_SAVE_STORAGE) === '1';
    setSaveGeminiApiKey(shouldSave);
    if (shouldSave && storedKey) {
      setGeminiApiKey(storedKey);
    }
    const storedExperiment = window.localStorage.getItem(EXPERIMENT_NAME_STORAGE) ?? '';
    if (storedExperiment) {
      setExperimentName(storedExperiment);
    }
  }, []);

  useEffect(() => {
    if (saveGeminiApiKey) {
      window.localStorage.setItem(GEMINI_KEY_STORAGE, geminiApiKey);
      window.localStorage.setItem(GEMINI_KEY_SAVE_STORAGE, '1');
      return;
    }
    window.localStorage.removeItem(GEMINI_KEY_STORAGE);
    window.localStorage.setItem(GEMINI_KEY_SAVE_STORAGE, '0');
  }, [geminiApiKey, saveGeminiApiKey]);

  useEffect(() => {
    if (experimentName.trim()) {
      window.localStorage.setItem(EXPERIMENT_NAME_STORAGE, experimentName);
      return;
    }
    window.localStorage.removeItem(EXPERIMENT_NAME_STORAGE);
  }, [experimentName]);

  useEffect(() => {
    void rehydrateActiveJob();
  }, []);

  useEffect(() => {
    if (analysisJob && ['queued', 'running', 'paused'].includes(analysisJob.status)) {
      window.localStorage.setItem(ACTIVE_JOB_STORAGE, analysisJob.id);
      return;
    }
    window.localStorage.removeItem(ACTIVE_JOB_STORAGE);
  }, [analysisJob]);

  async function refreshImages() {
    setLoadingImages(true);
    setError(null);
    try {
      const payload = await listInputImages();
      setImages(payload);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to list input images.');
    } finally {
      setLoadingImages(false);
    }
  }

  async function monitorAnalysisJob(jobId: string) {
    let currentJob = await getAnalysisJob(jobId);
    setAnalysisJob(currentJob);
    setAnalyzing(['queued', 'running', 'paused'].includes(currentJob.status));

    while (currentJob.status === 'queued' || currentJob.status === 'running' || currentJob.status === 'paused') {
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
      currentJob = await getAnalysisJob(jobId);
      setAnalysisJob(currentJob);
      setAnalyzing(['queued', 'running', 'paused'].includes(currentJob.status));
    }

    if (currentJob.status === 'failed') {
      throw new Error(currentJob.error || 'Analysis failed.');
    }

    if (currentJob.status === 'stopped') {
      setResults([]);
      setSelectedResult(null);
      await refreshOutputs(activeRunId ?? undefined);
      await refreshImages();
      return;
    }

    if (!currentJob.result) {
      throw new Error('Analysis finished without a result payload.');
    }

    setResults(currentJob.result.results);
    setLatestRun(currentJob.result.run);
    const preferredRunId =
      currentJob.result.run.id ?? currentJob.result.run.outputDir.split('/').filter(Boolean).pop();
    await refreshOutputs(preferredRunId);
    await refreshImages();
  }

  async function rehydrateActiveJob() {
    try {
      const storedJobId = window.localStorage.getItem(ACTIVE_JOB_STORAGE);
      const activeJob = storedJobId ? await getAnalysisJob(storedJobId).catch(() => null) : await getLatestActiveAnalysisJob();
      if (!activeJob || !['queued', 'running', 'paused'].includes(activeJob.status)) {
        return;
      }
      setAnalysisJob(activeJob);
      setAnalyzing(true);
      if (activeJob.experimentName) {
        setExperimentName(activeJob.experimentName);
      }
      setEngine(activeJob.engine);
      await monitorAnalysisJob(activeJob.id);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to reconnect to the active analysis job.');
      setAnalyzing(false);
    }
  }

  async function refreshOutputs(preferredRunId?: string) {
    setLoadingOutputs(true);
    setError(null);
    try {
      const runs = await listOutputRuns();
      setOutputRuns(runs);
      const nextRunId = preferredRunId ?? latestRun?.id ?? runs[0]?.id;
      if (nextRunId) {
        await handleSelectRun(nextRunId, runs);
      } else if (runs.length === 0) {
        setLatestRun(null);
        setResults([]);
        setSelectedResult(null);
        setReportBundle(null);
      }
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to list output runs.');
    } finally {
      setLoadingOutputs(false);
    }
  }

  function toggleImage(filename: string) {
    setImages((current) =>
      current.map((image) =>
        image.filename === filename ? {...image, selected: !image.selected} : image,
      ),
    );
  }

  async function handleRunAnalysis() {
    const selected = images.filter((image) => image.selected).map((image) => image.filename);
    if (selected.length === 0) {
      setError('Select at least one image from input_images before running analysis.');
      return;
    }

    setAnalyzing(true);
    setError(null);
    try {
      const trimmedExperimentName = experimentName.trim();
      const tags = trimmedExperimentName ? [trimmedExperimentName] : [];
      const started = await startAnalysis(
        engine,
        selected,
        trimmedExperimentName,
        tags,
        engine === 'gemini' ? geminiApiKey.trim() : undefined,
      );
      await monitorAnalysisJob(started.jobId);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Analysis failed.');
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleUploadImages(event: React.ChangeEvent<HTMLInputElement>) {
    const fileList = event.target.files;
    const files: File[] = [];
    if (fileList) {
      for (let index = 0; index < fileList.length; index += 1) {
        const file = fileList.item(index);
        if (file && file.type.startsWith('image/')) {
          files.push(file);
        }
      }
    }
    event.target.value = '';
    if (files.length === 0) {
      return;
    }

    setUploadingImages(true);
    setError(null);
    try {
      await uploadInputImages(files);
      await refreshImages();
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to upload images.');
    } finally {
      setUploadingImages(false);
    }
  }

  function handleClearGeminiApiKey() {
    setGeminiApiKey('');
    setSaveGeminiApiKey(false);
    window.localStorage.removeItem(GEMINI_KEY_STORAGE);
    window.localStorage.setItem(GEMINI_KEY_SAVE_STORAGE, '0');
  }

  async function handlePauseJob() {
    if (!analysisJob) {
      return;
    }
    setJobActionPending('pause');
    setError(null);
    try {
      const updatedJob = await pauseAnalysisJob(analysisJob.id);
      setAnalysisJob(updatedJob);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to pause analysis job.');
    } finally {
      setJobActionPending(null);
    }
  }

  async function handleResumeJob() {
    if (!analysisJob) {
      return;
    }
    setJobActionPending('resume');
    setError(null);
    try {
      const updatedJob = await resumeAnalysisJob(analysisJob.id);
      setAnalysisJob(updatedJob);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to resume analysis job.');
    } finally {
      setJobActionPending(null);
    }
  }

  async function handleStopJob() {
    if (!analysisJob) {
      return;
    }
    setJobActionPending('stop');
    setError(null);
    try {
      const updatedJob = await stopAnalysisJob(analysisJob.id);
      setAnalysisJob(updatedJob);
      setAnalyzing(false);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to stop analysis job.');
    } finally {
      setJobActionPending(null);
    }
  }

  async function handleStopPort() {
    setJobActionPending('port');
    setError(null);
    try {
      const payload = await stopLocalApp();
      window.alert(`${payload.message}\n\nRestart command:\n${payload.restartCommand}`);
      setAnalyzing(false);
      setAnalysisJob((current) =>
        current
          ? {
              ...current,
              status: current.status === 'completed' ? current.status : 'stopped',
              progress: {
                ...current.progress,
                stage: 'Frontend and API stopped',
              },
              logs: [
                ...current.logs,
                `[job] Local app stopped from GUI (${(payload.stoppedPorts ?? [payload.stoppedPort]).join(', ')})`,
              ],
            }
          : current,
      );
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to stop local app.');
    } finally {
      setJobActionPending(null);
    }
  }

  async function handleGenerateReport() {
    if (!activeRunId) {
      setError('Load a saved run before generating a report.');
      return;
    }
    setReportGenerating(true);
    setError(null);
    try {
      const bundle = await generateOutputReport(activeRunId, effectiveExperimentName, effectiveTags);
      setReportBundle(bundle);
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : 'Failed to generate report.');
    } finally {
      setReportGenerating(false);
    }
  }

  function toggleRunSelection(runId: string) {
    setSelectedRunIds((current) =>
      current.includes(runId) ? current.filter((item) => item !== runId) : [...current, runId],
    );
  }

  async function handleRunRelocation(runIds: string[], action: 'delete' | 'archive') {
    if (runIds.length === 0) {
      return;
    }
    setError(null);
    const remainingRuns = outputRuns.filter((run) => !run.id || !runIds.includes(run.id));
    try {
      if (action === 'delete') {
        await deleteOutputRuns(runIds);
      } else {
        await archiveOutputRuns(runIds);
      }
      setOutputRuns(remainingRuns);
      const activeId = latestRun?.id ?? latestRun?.outputDir.split('/').filter(Boolean).pop() ?? null;
      if (activeId && runIds.includes(activeId)) {
        const nextRun = remainingRuns[0] ?? null;
        if (nextRun?.id) {
          await handleSelectRun(nextRun.id, remainingRuns);
        } else {
          setLatestRun(null);
          setResults([]);
          setSelectedResult(null);
          setReportBundle(null);
        }
      }
      setSelectedRunIds((current) => current.filter((item) => !runIds.includes(item)));
      await refreshOutputs(activeId && runIds.includes(activeId) ? undefined : activeId ?? undefined);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : action === 'delete'
            ? 'Failed to delete output runs.'
            : 'Failed to archive output runs.',
      );
    }
  }

  async function handleDeleteRuns(runIds: string[]) {
    await handleRunRelocation(runIds, 'delete');
  }

  async function handleArchiveRuns(runIds: string[]) {
    await handleRunRelocation(runIds, 'archive');
  }

  async function handleSelectRun(runId: string, runs = outputRuns) {
    setLoadingRunId(runId);
    setError(null);
    try {
      const payload = await getOutputRun(runId);
      setLatestRun(payload.run);
      setResults(payload.results);
      setExperimentName((current) =>
        payload.run.experimentName && payload.run.experimentName.trim() ? payload.run.experimentName : current,
      );
      setSelectedResult(null);
      setReportBundle(null);
      if (runs.length > 0) {
        setOutputRuns(runs);
      }
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : `Failed to load output run ${runId}.`);
    } finally {
      setLoadingRunId(null);
    }
  }

  function exportToR() {
    if (results.length === 0) {
      return;
    }

    const runId = activeRunId ?? 'magnaporthe_run';
    const experimentSlug = effectiveExperimentName.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    const tags = effectiveTags;
    const tagSlug = tags.map((tag) => tag.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')).filter(Boolean).join('_');
    const dataJson = JSON.stringify(
      results.map((result) => ({
        id: result.id,
        filename: result.filename,
        day: result.day,
        area: result.morphology.areaMm2,
        radius: result.morphology.equivalentRadiusMm,
        diameter: result.morphology.diameterMm,
        perimeter: result.morphology.perimeterMm,
        circularity: result.morphology.circularity,
        eccentricity: result.morphology.eccentricity,
        edge_roughness: result.morphology.edgeRoughness,
        contrast: result.texture.contrast,
        correlation: result.texture.correlation,
        energy: result.texture.energy,
        homogeneity: result.texture.homogeneity,
        entropy: result.texture.entropy,
        center_edge_delta: result.texture.centerToEdgeDelta,
        density_index: result.texture.densityIndex,
        core: result.texture.radialZonation.core,
        middle: result.texture.radialZonation.middle,
        outer: result.texture.radialZonation.outer,
        crack_count: result.cracks.count,
        crack_coverage: result.cracks.coveragePct,
        proportional_crack_coverage: result.cracks.proportionalCoveragePct,
        velocity: result.kinematics.radialVelocity,
        area_growth_rate: result.kinematics.areaGrowthRate,
        relative_growth_rate: result.kinematics.relativeGrowthRate,
        radial_acceleration: result.kinematics.radialAcceleration,
        ring_spacing: result.rawAnalysis?.radial_profile?.ringSpacingMm ?? 0,
        radial_profile_radius_mm: result.rawAnalysis?.radial_profile?.radiusMm ?? [],
        radial_profile_mean_intensity: result.rawAnalysis?.radial_profile?.meanIntensity ?? [],
        qc_status: result.qcStatus,
        qc_notes: result.qcNotes,
      })),
      null,
      2,
    );

    const rScript = `
required_packages <- c("ggplot2", "dplyr", "tidyr", "readr", "jsonlite", "purrr", "stringr", "tibble")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Install required R packages first: ",
      paste(missing_packages, collapse = ", ")
    )
  )
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(jsonlite)
  library(purrr)
  library(stringr)
})

json_text <- ${JSON.stringify(dataJson)}
run_id <- "${runId}"
experiment_name <- ${JSON.stringify(effectiveExperimentName)}
tags <- c(${tags.map((tag) => JSON.stringify(tag)).join(', ')})
tags <- unique(tags[nzchar(tags)])

get_growth_data <- function() {
  df <- jsonlite::fromJSON(json_text, simplifyVector = TRUE)
  df <- tibble::as_tibble(df) %>%
    arrange(day, filename) %>%
    mutate(
      outer_core_delta = outer - core,
      middle_core_delta = middle - core,
      texture_spread = pmax(core, middle, outer) - pmin(core, middle, outer)
    )
  df
}

sanitize_label <- function(value) {
  gsub("[^A-Za-z0-9]+", "_", value)
}

ensure_output_dir <- function() {
  experiment_slug <- if (nzchar(experiment_name)) sanitize_label(experiment_name) else ""
  tag_values <- tags
  if (nzchar(experiment_name) && length(tag_values) > 0) {
    tag_values <- tag_values[tolower(tag_values) != tolower(experiment_name)]
  }
  experiment_suffix <- if (nzchar(experiment_slug)) paste0("_", experiment_slug) else ""
  tag_suffix <- if (length(tag_values) > 0) paste0("_", paste(vapply(tag_values, sanitize_label, character(1)), collapse = "_")) else ""
  output_dir <- file.path(getwd(), paste0("rstudio_export_", run_id, experiment_suffix, tag_suffix))
  plots_dir <- file.path(output_dir, "plots")
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  list(output_dir = output_dir, plots_dir = plots_dir)
}

save_plot <- function(plot, path, width = 9, height = 5) {
  ggplot2::ggsave(path, plot = plot, width = width, height = height, dpi = 180, bg = "white")
}

sanitize_filename <- function(value) {
  value <- gsub("/", "_", value, fixed = TRUE)
  value <- gsub(intToUtf8(92), "_", value, fixed = TRUE)
  value
}

apply_finder_tags <- function(target_paths, tag_values) {
  tag_values <- unique(tag_values[nzchar(tag_values)])
  if (length(tag_values) == 0) {
    return(invisible(NULL))
  }
  if (Sys.info()[["sysname"]] != "Darwin") {
    return(invisible(NULL))
  }
  python3_path <- Sys.which("python3")
  if (!nzchar(python3_path)) {
    return(invisible(NULL))
  }
  target_paths <- unique(target_paths[file.exists(target_paths)])
  if (length(target_paths) == 0) {
    return(invisible(NULL))
  }
  python_code <- paste(
    "import json, os, plistlib, subprocess, sys",
    "paths = json.loads(sys.argv[1])",
    "tags = [tag for tag in json.loads(sys.argv[2]) if tag]",
    "payload = plistlib.dumps([f'{tag}\\\\n0' for tag in tags], fmt=plistlib.FMT_BINARY).hex()",
    "finder_info = '00' * 32",
    "for path in paths:",
    "    if os.path.exists(path):",
    "        subprocess.run(['/usr/bin/xattr', '-wx', 'com.apple.FinderInfo', finder_info, path], check=False)",
    "        subprocess.run(['/usr/bin/xattr', '-wx', 'com.apple.metadata:_kMDItemUserTags', payload, path], check=False)",
    sep = "\\n"
  )
  python_script <- tempfile(fileext = ".py")
  writeLines(python_code, python_script)
  suppressWarnings(
    try(
      system2(
        python3_path,
        c(
          python_script,
          jsonlite::toJSON(as.character(target_paths), auto_unbox = TRUE),
          jsonlite::toJSON(as.character(tag_values), auto_unbox = TRUE)
        ),
        stdout = FALSE,
        stderr = FALSE
      ),
      silent = TRUE
    )
  )
  unlink(python_script)
  invisible(NULL)
}

refresh_macos_metadata <- function(target_paths) {
  if (Sys.info()[["sysname"]] != "Darwin") {
    return(invisible(NULL))
  }
  mdimport_path <- Sys.which("mdimport")
  if (!nzchar(mdimport_path)) {
    return(invisible(NULL))
  }
  for (target in unique(target_paths[file.exists(target_paths)])) {
    suppressWarnings(
      try(
        system2(mdimport_path, c("-f", normalizePath(target, winslash = "/", mustWork = FALSE)), stdout = FALSE, stderr = FALSE),
        silent = TRUE
      )
    )
  }
  invisible(NULL)
}

line_plot_data <- function(df, series) {
  purrr::imap_dfr(series, function(spec, idx) {
    tibble::tibble(
      day = df$day,
      value = df[[spec$key]],
      series = spec$label,
      colour = spec$colour
    )
  })
}

save_line_plot <- function(df, plots_dir, filename, title, xlabel, ylabel, series) {
  long_df <- line_plot_data(df, series)
  colour_map <- stats::setNames(vapply(series, function(item) item$colour, character(1)), vapply(series, function(item) item$label, character(1)))
  plot <- ggplot(long_df, aes(x = day, y = value, color = series)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    scale_color_manual(values = colour_map) +
    labs(title = title, x = xlabel, y = ylabel, color = NULL) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold")
    )
  save_plot(plot, file.path(plots_dir, filename))
}

save_histogram <- function(df, plots_dir, filename, title, column, xlabel, fill_colour) {
  plot <- ggplot(df, aes(x = .data[[column]])) +
    geom_histogram(bins = max(3, min(8, nrow(df))), fill = fill_colour, color = "white", alpha = 0.9) +
    labs(title = title, x = xlabel, y = "Image count") +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold"))
  save_plot(plot, file.path(plots_dir, filename), width = 8, height = 5)
}

save_scatter <- function(df, plots_dir, filename, title, x, y, xlabel, ylabel, size_key) {
  plot <- ggplot(df, aes(x = .data[[x]], y = .data[[y]], size = .data[[size_key]], color = day)) +
    geom_point(alpha = 0.8) +
    scale_color_gradient(low = "#67e8f9", high = "#1d4ed8") +
    labs(title = title, x = xlabel, y = ylabel, color = "Day", size = size_key) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold"))
  save_plot(plot, file.path(plots_dir, filename), width = 8, height = 5)
}

save_plate_feature_heatmaps <- function(df, plots_dir) {
  heatmap_dir <- file.path(plots_dir, "plate_feature_heatmaps")
  dir.create(heatmap_dir, recursive = TRUE, showWarnings = FALSE)
  feature_spec <- tibble::tribble(
    ~group, ~feature, ~column,
    "Morphology", "Area (mm²)", "area",
    "Morphology", "Diameter (mm)", "diameter",
    "Morphology", "Circularity", "circularity",
    "Morphology", "Edge roughness", "edge_roughness",
    "Texture", "Entropy (bits)", "entropy",
    "Texture", "Center-edge delta", "center_edge_delta",
    "Texture", "Density index", "density_index",
    "Texture", "Contrast", "contrast",
    "Stress", "Crack coverage (%)", "crack_coverage",
    "Stress", "Prop. crack cov. (%)", "proportional_crack_coverage",
    "Stress", "Crack count", "crack_count",
    "Stress", "Ring spacing (mm)", "ring_spacing",
    "Kinematics", "Velocity (mm/day)", "velocity",
    "Kinematics", "Area growth (mm²/day)", "area_growth_rate",
    "Kinematics", "Relative growth (1/day)", "relative_growth_rate",
    "Kinematics", "Acceleration", "radial_acceleration"
  )

  for (i in seq_len(nrow(df))) {
    row <- df[i, ]
    heatmap_df <- feature_spec %>%
      mutate(value = purrr::map_dbl(column, ~ as.numeric(row[[.x]]))) %>%
      mutate(group = factor(group, levels = c("Morphology", "Texture", "Stress", "Kinematics")))

    plot <- ggplot(heatmap_df, aes(x = feature, y = group, fill = value)) +
      geom_tile(color = "white") +
      geom_text(aes(label = sprintf("%.2f", value)), size = 3) +
      scale_fill_gradient2(low = "#f43f5e", mid = "#f8fafc", high = "#14b8a6") +
      labs(
        title = paste("Plate Feature Heatmap:", row$filename[[1]]),
        x = "Feature",
        y = "Feature group",
        fill = "Value"
      ) +
      theme_minimal(base_size = 11) +
      theme(
        axis.text.x = element_text(angle = 25, hjust = 1),
        panel.grid = element_blank(),
        plot.title = element_text(face = "bold")
      )

    safe_name <- sanitize_filename(row$filename[[1]])
    save_plot(plot, file.path(heatmap_dir, paste0(safe_name, "_feature_heatmap.png")) , width = 9, height = 5)
  }
}

save_radial_profiles <- function(df, plots_dir) {
  profile_dir <- file.path(plots_dir, "radial_profiles")
  dir.create(profile_dir, recursive = TRUE, showWarnings = FALSE)

  for (i in seq_len(nrow(df))) {
    row <- df[i, ]
    radius <- row$radial_profile_radius_mm[[1]]
    intensity <- row$radial_profile_mean_intensity[[1]]
    if (length(radius) == 0 || length(intensity) == 0) {
      next
    }
    profile_df <- tibble::tibble(radius_mm = radius, mean_intensity = intensity)
    plot <- ggplot(profile_df, aes(x = radius_mm, y = mean_intensity)) +
      geom_line(color = "#2563eb", linewidth = 1) +
      labs(
        title = paste("Radial Intensity Profile:", row$filename[[1]]),
        x = "Radius (mm)",
        y = "Mean grayscale intensity"
      ) +
      theme_minimal(base_size = 12) +
      theme(panel.grid.minor = element_blank(), plot.title = element_text(face = "bold"))

    safe_name <- sanitize_filename(row$filename[[1]])
    save_plot(plot, file.path(profile_dir, paste0(safe_name, "_radial_profile.png")), width = 8, height = 4.5)
  }
}

make_all_graphs <- function(df, plots_dir) {
  save_line_plot(df, plots_dir, "colony_expansion.png", "Colony Expansion", "Time (days)", "Measurement (mm² / mm)", list(
    list(key = "area", label = "Area (mm²)", colour = "#4f46e5"),
    list(key = "radius", label = "Radius (mm)", colour = "#059669")
  ))
  save_line_plot(df, plots_dir, "radial_texture_zonation.png", "Radial Texture Zonation", "Time (days)", "Texture SD (a.u.)", list(
    list(key = "core", label = "Core", colour = "#1b9e77"),
    list(key = "middle", label = "Middle", colour = "#d95f02"),
    list(key = "outer", label = "Outer", colour = "#7570b3")
  ))
  save_line_plot(df, plots_dir, "colony_morphogenesis.png", "Colony Morphogenesis", "Time (days)", "Measurement (mm / unitless)", list(
    list(key = "diameter", label = "Diameter (mm)", colour = "#e11d48"),
    list(key = "perimeter", label = "Perimeter (mm)", colour = "#f97316"),
    list(key = "circularity", label = "Circularity", colour = "#7c3aed")
  ))
  save_line_plot(df, plots_dir, "stress_remodeling.png", "Stress Remodeling", "Time (days)", "Coverage / count", list(
    list(key = "crack_coverage", label = "Crack Coverage (%)", colour = "#f59e0b"),
    list(key = "proportional_crack_coverage", label = "Proportional Crack Coverage (%)", colour = "#ef4444"),
    list(key = "crack_count", label = "Crack Count", colour = "#92400e")
  ))
  save_line_plot(df, plots_dir, "radial_differentiation.png", "Radial Differentiation", "Time (days)", "Radial texture delta (a.u.)", list(
    list(key = "outer_core_delta", label = "Outer - Core", colour = "#06b6d4"),
    list(key = "middle_core_delta", label = "Middle - Core", colour = "#0ea5e9"),
    list(key = "texture_spread", label = "Texture Spread", colour = "#1d4ed8")
  ))
  save_scatter(df, plots_dir, "shape_vs_stress.png", "Shape vs Stress Trajectory", "eccentricity", "crack_coverage", "Eccentricity", "Crack Coverage (%)", "diameter")
  save_line_plot(df, plots_dir, "growth_kinematics.png", "Growth Kinematics", "Time (days)", "Rate / acceleration", list(
    list(key = "velocity", label = "Radial Velocity (mm/day)", colour = "#1f78b4"),
    list(key = "radial_acceleration", label = "Radial Acceleration", colour = "#33a02c")
  ))
  save_scatter(df, plots_dir, "circularity_vs_crack_coverage.png", "Circularity vs Crack Coverage", "crack_coverage", "circularity", "Crack Coverage (%)", "Circularity", "day")
  save_line_plot(df, plots_dir, "texture_entropy_center_edge.png", "Texture Entropy and Center-to-Edge Intensity", "Time (days)", "Texture signal", list(
    list(key = "entropy", label = "Texture Entropy (bits)", colour = "#0f766e"),
    list(key = "center_edge_delta", label = "Center-to-Edge Intensity", colour = "#14b8a6")
  ))
  save_line_plot(df, plots_dir, "relative_growth_edge_roughness.png", "Relative Growth and Edge Roughness", "Time (days)", "Growth / roughness", list(
    list(key = "relative_growth_rate", label = "Relative Growth Rate (1/day)", colour = "#ea580c"),
    list(key = "edge_roughness", label = "Edge Roughness", colour = "#b45309")
  ))
  save_histogram(df, plots_dir, "area_distribution.png", "Area Distribution", "area", "Area (mm²)", "#6366f1")
  save_histogram(df, plots_dir, "growth_rate_distribution.png", "Growth Rate Distribution", "area_growth_rate", "Growth rate (mm²/day)", "#10b981")
  save_histogram(df, plots_dir, "entropy_distribution.png", "Entropy Distribution", "entropy", "Entropy (bits)", "#14b8a6")
  save_scatter(df, plots_dir, "area_vs_circularity.png", "Area vs Circularity", "area", "circularity", "Colony Area (mm²)", "Circularity", "day")
  save_plate_feature_heatmaps(df, plots_dir)
  save_radial_profiles(df, plots_dir)
}

write_summary <- function(df, output_dir) {
  lines <- c(
    paste0("Run ID: ", run_id),
    paste0("Experiment: ", ifelse(nzchar(experiment_name), experiment_name, "not provided")),
    paste0("Tags: ", if (length(tags) > 0) paste(tags, collapse = ", ") else "none"),
    paste0("Image count: ", nrow(df)),
    "Files written:",
    "- mgp_analysis_results.csv",
    "- plots/*.png",
    "- plots/plate_feature_heatmaps/*.png",
    "- plots/radial_profiles/*.png",
    "",
    "Open this R file in RStudio and run it to regenerate the figures with ggplot2."
  )
  writeLines(lines, file.path(output_dir, "README_RStudio.txt"))
}

df <- get_growth_data()
paths <- ensure_output_dir()
readr::write_csv(df, file.path(paths$output_dir, "mgp_analysis_results.csv"))
make_all_graphs(df, paths$plots_dir)
write_summary(df, paths$output_dir)
tag_targets <- c(
  paths$output_dir,
  list.files(paths$output_dir, recursive = TRUE, full.names = TRUE, all.files = FALSE)
)
apply_finder_tags(tag_targets, if (length(tags) > 0) tags else if (nzchar(experiment_name)) c(experiment_name) else character(0))
refresh_macos_metadata(tag_targets)
print(head(df))
message("Saved RStudio export to: ", paths$output_dir)
`.trim();

    const blob = new Blob([rScript], {type: 'text/x-rsrc'});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${runId}${experimentSlug ? `_${experimentSlug}` : ''}${tagSlug ? `_${tagSlug}` : ''}_rstudio_analysis.R`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const selectedCount = images.filter((image) => image.selected).length;
  const activeRunId = latestRun?.id ?? latestRun?.outputDir.split('/').filter(Boolean).pop() ?? null;
  const growthData: GrowthData[] = [...results]
    .sort((left, right) => left.day - right.day || left.filename.localeCompare(right.filename))
    .map((result) => ({
      day: result.day,
      area: result.morphology.areaMm2,
      radius: result.morphology.equivalentRadiusMm,
      diameter: result.morphology.diameterMm,
      perimeter: result.morphology.perimeterMm,
      circularity: result.morphology.circularity,
      eccentricity: result.morphology.eccentricity,
      edgeRoughness: result.morphology.edgeRoughness,
      contrast: result.texture.contrast,
      entropy: result.texture.entropy,
      centerEdgeDelta: result.texture.centerToEdgeDelta,
      densityIndex: result.texture.densityIndex,
      ringSpacing: result.rawAnalysis?.radial_profile?.ringSpacingMm ?? 0,
      crackCoverage: result.cracks.coveragePct,
      proportionalCrackCoverage: result.cracks.proportionalCoveragePct,
      crackCount: result.cracks.count,
      velocity: result.kinematics.radialVelocity,
      areaRate: result.kinematics.areaGrowthRate,
      relativeGrowthRate: result.kinematics.relativeGrowthRate,
      acceleration: result.kinematics.radialAcceleration,
      core: result.texture.radialZonation.core,
      middle: result.texture.radialZonation.middle,
      outer: result.texture.radialZonation.outer,
      outerCoreDelta: result.texture.radialZonation.outer - result.texture.radialZonation.core,
      middleCoreDelta: result.texture.radialZonation.middle - result.texture.radialZonation.core,
      textureSpread:
        Math.max(
          result.texture.radialZonation.core,
          result.texture.radialZonation.middle,
          result.texture.radialZonation.outer,
        ) -
        Math.min(
          result.texture.radialZonation.core,
          result.texture.radialZonation.middle,
          result.texture.radialZonation.outer,
        ),
    }));
  const uniqueDayCount = new Set(growthData.map((point) => point.day)).size;
  const lineType = uniqueDayCount >= 6 ? 'monotone' : 'linear';
  const areaHistogram = buildHistogram(growthData.map((point) => point.area), 6, 'mm²');
  const growthRateHistogram = buildHistogram(growthData.map((point) => point.areaRate), 6, 'mm²/day');
  const entropyHistogram = buildHistogram(growthData.map((point) => point.entropy), 6, 'bits');
  const singleAxisChartMargin = {top: 8, right: 18, left: 34, bottom: 22};
  const dualAxisChartMargin = {top: 8, right: 34, left: 34, bottom: 22};
  const scatterChartMargin = {top: 8, right: 24, left: 34, bottom: 22};
  const distributionChartMargin = {top: 8, right: 16, left: 38, bottom: 30};
  const axisLabelOffset = 10;
  const chartLegendProps = {verticalAlign: 'top' as const, wrapperStyle: {fontSize: 10, paddingBottom: 8}};
  const canPauseJob = analysisJob?.status === 'running';
  const canResumeJob = analysisJob?.status === 'paused';
  const canStopJob = analysisJob ? ['queued', 'running', 'paused'].includes(analysisJob.status) : false;
  const effectiveExperimentName =
    experimentName.trim() || latestRun?.experimentName?.trim() || analysisJob?.experimentName?.trim() || '';
  const effectiveTags = effectiveExperimentName ? [effectiveExperimentName] : [...new Set([...(latestRun?.tags ?? []), ...(analysisJob?.tags ?? [])])];

  return (
    <div className="min-h-screen bg-[#fafafa] text-slate-900 font-sans selection:bg-indigo-100">
      <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-start justify-between gap-4 px-6 py-4">
          <div className="flex min-w-0 items-start gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 shadow-lg shadow-indigo-200">
              <Activity className="h-6 w-6 text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="text-2xl font-bold leading-none tracking-tight text-slate-900">Gray Leaf Spot</h1>
              <p className="mt-1 text-xs font-medium uppercase tracking-[0.22em] text-slate-500">
                Magnaporthe Analyser
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] text-slate-400">
                <span
                  className={cn(
                    'rounded-full px-2 py-0.5 font-bold',
                    expectedLocalFrontend ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700',
                  )}
                >
                  {connectionStatusLabel}
                </span>
                <span>Frontend: {frontendOrigin}</span>
                <span>|</span>
                <span>API: {apiConnectionLabel}</span>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-end gap-4">
            <div className="rounded-lg border border-slate-200 bg-slate-100 px-3 py-1.5">
              <span className="mr-2 text-[10px] font-bold uppercase text-slate-500">Engine</span>
              <div className="inline-flex overflow-hidden rounded-md border border-slate-200 bg-white">
                <button
                  onClick={() => setEngine('local')}
                  className={cn(
                    'px-3 py-1 text-[10px] font-bold transition-colors',
                    engine === 'local' ? 'bg-emerald-600 text-white' : 'text-slate-600',
                  )}
                >
                  LOCAL GEMMA 4
                </button>
                <button
                  onClick={() => setEngine('gemini')}
                  className={cn(
                    'px-3 py-1 text-[10px] font-bold transition-colors',
                    engine === 'gemini' ? 'bg-indigo-600 text-white' : 'text-slate-600',
                  )}
                >
                  GEMINI API
                </button>
              </div>
            </div>

            {engine === 'gemini' && (
              <div className="flex flex-col gap-2 rounded-lg border border-indigo-100 bg-indigo-50/70 px-3 py-2">
                <div className="flex items-center gap-2">
                  <input
                    type="password"
                    value={geminiApiKey}
                    onChange={(event) => setGeminiApiKey(event.target.value)}
                    placeholder="Enter Gemini API key"
                    className="w-56 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm outline-none transition-colors placeholder:text-slate-400 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-100"
                  />
                  <button
                    onClick={handleClearGeminiApiKey}
                    type="button"
                    className="rounded-lg border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-700 transition-colors hover:bg-rose-50"
                  >
                    Clear API Key
                  </button>
                </div>
                <label className="flex items-center gap-2 text-[11px] text-slate-600">
                  <input
                    type="checkbox"
                    checked={saveGeminiApiKey}
                    onChange={(event) => setSaveGeminiApiKey(event.target.checked)}
                    className="h-3.5 w-3.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  />
                  Save in this browser
                </label>
              </div>
            )}

            <button
              onClick={() => {
                void refreshImages();
                void refreshOutputs();
              }}
              className="flex items-center gap-2 rounded-lg bg-slate-100 px-4 py-2 text-sm font-medium transition-colors hover:bg-slate-200"
            >
              {loadingImages || loadingOutputs ? (
                <Loader2 className="animate-spin" size={16} />
              ) : (
                <RefreshCcw size={16} />
              )}
              Refresh Workspace
            </button>

            <input
              value={experimentName}
              onChange={(event) => setExperimentName(event.target.value)}
              placeholder="Your name / experiment"
              className="w-52 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm outline-none transition-colors placeholder:text-slate-400 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-100"
            />

            <button
              onClick={() => void handleRunAnalysis()}
              disabled={selectedCount === 0 || analyzing}
              className="flex items-center gap-2 rounded-lg bg-indigo-600 px-6 py-2 text-sm font-semibold text-white shadow-md shadow-indigo-100 transition-all hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {analyzing ? <Loader2 className="animate-spin" size={16} /> : <Activity size={16} />}
              Run Analysis
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-12 gap-8 px-6 py-8">
        <div className="col-span-3 space-y-6">
          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-bold text-slate-800">
                <FolderOpen size={18} className="text-indigo-500" />
                Root Input Images
              </h2>
              <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-bold text-slate-600">
                {selectedCount}/{images.length}
              </span>
            </div>

            <p className="mb-4 text-xs leading-relaxed text-slate-500">
              Add source images to `input_images/` at the repo root, then refresh this list. All generated artifacts are written under `outputs/`.
            </p>

            <label className="mb-4 flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-indigo-200 bg-indigo-50/60 px-3 py-3 text-sm font-semibold text-indigo-700 transition-colors hover:bg-indigo-100">
              {uploadingImages ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
              {uploadingImages ? 'Uploading images...' : 'Upload Images'}
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={(event) => void handleUploadImages(event)}
                className="hidden"
              />
            </label>

            <div className="space-y-3 pr-2">
              {images.length === 0 && (
                <div className="rounded-xl border-2 border-dashed border-slate-100 py-12 text-center">
                  <p className="text-sm text-slate-400">No root input images found</p>
                </div>
              )}

              {images.map((image) => (
                <button
                  key={image.filename}
                  onClick={() => toggleImage(image.filename)}
                  className={cn(
                    'group w-full overflow-hidden rounded-xl border text-left transition-all',
                    image.selected
                      ? 'border-indigo-300 bg-indigo-50/40'
                      : 'border-slate-200 bg-slate-50 hover:border-slate-300',
                  )}
                >
                  <img src={image.imageUrl} alt={image.filename} className="h-32 w-full object-cover" />
                  <div className="border-t border-slate-100 bg-white p-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="truncate font-mono text-[10px] text-slate-500">{image.filename}</p>
                      <span
                        className={cn(
                          'rounded-full px-2 py-1 text-[10px] font-bold',
                          image.selected ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-500',
                        )}
                      >
                        {image.selected ? 'Selected' : 'Skipped'}
                      </span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-bold text-slate-800">
                <Orbit size={18} className="text-emerald-500" />
                Output Runs
              </h2>
              <div className="flex items-center gap-2">
                <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-bold text-slate-600">
                  {outputRuns.length}
                </span>
                {selectedRunIds.length > 0 && (
                  <>
                    <button
                      onClick={() => void handleArchiveRuns([...selectedRunIds])}
                      className="rounded-lg border border-amber-200 bg-amber-50 px-2 py-1 text-[10px] font-bold text-amber-700 transition-colors hover:bg-amber-100"
                    >
                      Archive {selectedRunIds.length}
                    </button>
                    <button
                      onClick={() => void handleDeleteRuns([...selectedRunIds])}
                      className="rounded-lg border border-rose-200 bg-rose-50 px-2 py-1 text-[10px] font-bold text-rose-700 transition-colors hover:bg-rose-100"
                    >
                      Clear {selectedRunIds.length}
                    </button>
                  </>
                )}
              </div>
            </div>

            <p className="mb-4 text-xs leading-relaxed text-slate-500">
              Browse saved runs from `outputs/`, load any past result set back into the interactive dashboard, or move runs into `archives/`.
            </p>

            <div className="space-y-3">
              {outputRuns.length === 0 && (
                <div className="rounded-xl border-2 border-dashed border-slate-100 py-8 text-center text-sm text-slate-400">
                  No output runs found yet
                </div>
              )}

              {outputRuns.map((run) => {
                const isActive = run.id === activeRunId;
                const isLoading = loadingRunId === run.id;
                return (
                  <div
                    key={run.id ?? run.outputDir}
                    onClick={() => run.id && !isLoading && void handleSelectRun(run.id)}
                    role="button"
                    tabIndex={run.id && !isLoading ? 0 : -1}
                    onKeyDown={(event) => {
                      if ((event.key === 'Enter' || event.key === ' ') && run.id && !isLoading) {
                        event.preventDefault();
                        void handleSelectRun(run.id);
                      }
                    }}
                    className={cn(
                      'w-full cursor-pointer rounded-xl border p-3 text-left transition-all',
                      isActive
                        ? 'border-indigo-300 bg-indigo-50/50'
                        : 'border-slate-200 bg-slate-50 hover:border-slate-300 hover:bg-white',
                      (!run.id || isLoading) && 'cursor-default opacity-70',
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <label
                        className="mt-0.5 flex shrink-0 items-center"
                        onClick={(event) => event.stopPropagation()}
                      >
                        <input
                          type="checkbox"
                          checked={selectedRunIds.includes(run.id ?? '')}
                          onClick={(event) => event.stopPropagation()}
                          onChange={() => run.id && toggleRunSelection(run.id)}
                          className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                        />
                      </label>
                      <div className="min-w-0 flex-1">
                        <div className="min-w-0">
                          <p className="truncate text-xs font-bold uppercase tracking-wide text-slate-700">
                            {run.engine === 'local' ? 'Local Gemma 4' : 'Gemini API'}
                          </p>
                          <p className="mt-1 break-all font-mono text-[10px] text-slate-500">
                            {run.id ?? run.outputDir}
                          </p>
                          <p className="mt-2 break-words text-[11px] text-slate-500">{run.engineModel}</p>
                        </div>

                        <div className="mt-3 flex flex-wrap items-center gap-2">
                          {isLoading ? (
                            <Loader2 size={14} className="animate-spin text-indigo-500" />
                          ) : (
                            <>
                              <button
                                onClick={(event) => {
                                  event.stopPropagation();
                                  if (run.id) {
                                    void handleArchiveRuns([run.id]);
                                  }
                                }}
                                className="rounded-full bg-amber-100 px-2.5 py-1 text-[10px] font-bold text-amber-700 transition-colors hover:bg-amber-200"
                              >
                                Archive
                              </button>
                              <button
                                onClick={(event) => {
                                  event.stopPropagation();
                                  if (run.id) {
                                    void handleDeleteRuns([run.id]);
                                  }
                                }}
                                className="rounded-full bg-rose-100 px-2.5 py-1 text-[10px] font-bold text-rose-700 transition-colors hover:bg-rose-200"
                              >
                                Clear
                              </button>
                              <span
                                className={cn(
                                  'rounded-full px-2.5 py-1 text-[10px] font-bold',
                                  isActive ? 'bg-indigo-600 text-white' : 'bg-slate-200 text-slate-600',
                                )}
                              >
                                {isActive ? 'Loaded' : 'Open'}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    <p className="mt-3 text-[10px] text-slate-400">
                      {new Date(run.createdAt).toLocaleString()}
                    </p>
                  </div>
                );
              })}
            </div>
          </section>

          {latestRun && (
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="mb-3 flex items-center gap-2 font-bold text-slate-800">
                <Download size={18} className="text-emerald-500" />
                Active Output
              </h2>
              <p className="text-xs text-slate-500">Engine: {latestRun.engineModel}</p>
              <p className="mt-1 break-all text-xs text-slate-500">Folder: {latestRun.outputDir}</p>
              <div className="mt-4 grid gap-2">
                <a
                  href={latestRun.analysisJson}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium transition-colors hover:bg-slate-50"
                >
                  Open `analysis.json`
                </a>
                <a
                  href={latestRun.analysisCsv}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium transition-colors hover:bg-slate-50"
                >
                  Open `analysis.csv`
                </a>
              </div>
            </section>
          )}

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h2 className="flex items-center gap-2 font-bold text-slate-800">
                <CirclePower size={18} className="text-slate-700" />
                Server Controls
              </h2>
            </div>
            <p className="mb-4 text-xs leading-relaxed text-slate-500">
              This control shuts down both the GUI port 3000 and the backend API port {backendPortLabel}. The popup shows one command to start the full local app again.
            </p>
            <button
              onClick={() => void handleStopPort()}
              disabled={jobActionPending !== null}
              className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-300 bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {jobActionPending === 'port' ? <Loader2 size={16} className="animate-spin" /> : <CirclePower size={16} />}
              Stop GUI + Backend
            </button>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h2 className="flex items-center gap-2 font-bold text-slate-800">
                <FileText size={18} className="text-indigo-500" />
                Markdown Maker
              </h2>
              {reportBundle && (
                <span className="rounded-full bg-indigo-50 px-2 py-1 text-[10px] font-bold text-indigo-700">
                  {reportBundle.graphCount} graphs
                </span>
              )}
            </div>
            <p className="mb-4 text-xs leading-relaxed text-slate-500">
              Build a templated markdown report for the active run and compile the same content into a clean PDF with embedded graphs and metric summaries.
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => void handleGenerateReport()}
                disabled={!activeRunId || reportGenerating}
                className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {reportGenerating ? <Loader2 size={16} className="animate-spin" /> : <FileText size={16} />}
                Generate Markdown + PDF
              </button>
              {reportBundle && (
                <>
                  <a
                    href={reportBundle.markdownPath}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium transition-colors hover:bg-slate-50"
                  >
                    Open Markdown
                  </a>
                  <a
                    href={reportBundle.pdfPath}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium transition-colors hover:bg-slate-50"
                    download
                  >
                    Download PDF
                  </a>
                </>
              )}
            </div>
            {reportBundle && (
              <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3">
                <p className="mb-2 text-[11px] font-bold uppercase tracking-wide text-slate-500">
                  Template Preview: {reportBundle.template}
                </p>
                <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-5 text-slate-700">
                  {reportBundle.markdownContent}
                </pre>
              </div>
            )}
          </section>

          {analysisJob && (
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-4 flex items-center justify-between gap-3">
                <h2 className="flex items-center gap-2 font-bold text-slate-800">
                  <SquareTerminal size={18} className="text-slate-700" />
                  Backend Progress
                </h2>
                <span
                  className={cn(
                    'rounded-full px-2 py-1 text-[10px] font-bold',
                    analysisJob.status === 'completed' && 'bg-emerald-100 text-emerald-700',
                    analysisJob.status === 'failed' && 'bg-rose-100 text-rose-700',
                    analysisJob.status === 'paused' && 'bg-sky-100 text-sky-700',
                    analysisJob.status === 'stopped' && 'bg-slate-200 text-slate-700',
                    (analysisJob.status === 'running' || analysisJob.status === 'queued') &&
                      'bg-amber-100 text-amber-700',
                  )}
                >
                  {analysisJob.status}
                </span>
              </div>

              <div className="mb-4 flex flex-wrap items-center gap-2">
                <button
                  onClick={() => void handlePauseJob()}
                  disabled={!canPauseJob || jobActionPending !== null}
                  className="flex items-center gap-1.5 rounded-lg border border-sky-200 bg-sky-50 px-3 py-1.5 text-[11px] font-bold text-sky-700 transition-colors hover:bg-sky-100 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {jobActionPending === 'pause' ? <Loader2 size={13} className="animate-spin" /> : <Pause size={13} />}
                  Pause
                </button>
                <button
                  onClick={() => void handleResumeJob()}
                  disabled={!canResumeJob || jobActionPending !== null}
                  className="flex items-center gap-1.5 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-[11px] font-bold text-emerald-700 transition-colors hover:bg-emerald-100 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {jobActionPending === 'resume' ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                  Resume
                </button>
                <button
                  onClick={() => void handleStopJob()}
                  disabled={!canStopJob || jobActionPending !== null}
                  className="flex items-center gap-1.5 rounded-lg border border-rose-200 bg-rose-50 px-3 py-1.5 text-[11px] font-bold text-rose-700 transition-colors hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {jobActionPending === 'stop' ? <Loader2 size={13} className="animate-spin" /> : <Square size={13} />}
                  Stop Run
                </button>
              </div>

              <div className="mb-2 flex items-center justify-between text-[11px] text-slate-500">
                <span>{analysisJob.progress.stage}</span>
                <span>
                  {analysisJob.progress.current}/{analysisJob.progress.total}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full bg-indigo-600 transition-all"
                  style={{
                    width: `${analysisJob.progress.total > 0 ? (analysisJob.progress.current / analysisJob.progress.total) * 100 : 0}%`,
                  }}
                />
              </div>
              <div className="mt-4 rounded-xl bg-slate-950 p-3 font-mono text-[11px] text-emerald-300">
                <div className="mb-2 text-slate-400">$ pipeline.cli --engine {analysisJob.engine}</div>
                <div className="max-h-48 overflow-y-auto whitespace-pre-wrap break-words">
                  {analysisJob.logs.length > 0 ? analysisJob.logs.join('\n') : 'Waiting for backend logs...'}
                </div>
              </div>
            </section>
          )}
        </div>

        <div className="col-span-9 space-y-8">
          {error && (
            <section className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
              {error}
            </section>
          )}

          {results.length > 0 ? (
            <>
              <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <h2 className="font-bold text-slate-800">Completed run</h2>
                    <p className="text-sm text-slate-500">
                      {latestRun?.engine === 'local'
                        ? 'Interactive local Gemma 4 outputs loaded from root outputs.'
                        : `Cloud analysis using ${latestRun?.engineModel ?? 'Gemini API'} on the same root input_images set.`}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {latestRun && (
                      <a
                        href={latestRun.analysisCsv}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-lg bg-slate-100 p-2 text-slate-600 transition-colors hover:bg-slate-200"
                        title="Open server-generated CSV"
                      >
                        <Download size={18} />
                      </a>
                    )}
                    <button
                      onClick={exportToR}
                      className="flex items-center gap-2 rounded-lg border border-emerald-100 bg-emerald-50 px-3 py-2 text-xs font-bold text-emerald-700 transition-colors hover:bg-emerald-100"
                    >
                      <FileText size={14} />
                      R-STUDIO
                    </button>
                  </div>
                </div>
              </section>

              <div className="grid grid-cols-2 gap-6">
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <BarChart3 size={16} className="text-indigo-500" />
                    Colony Expansion (Area & Radius)
                  </h3>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={growthData} margin={dualAxisChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="day" tick={{fontSize: 10}} label={{value: 'Time (days)', position: 'insideBottom', offset: -4}} />
                        <YAxis yAxisId="left" width={72} tick={{fontSize: 10}} label={{value: 'Area (mm²)', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <YAxis yAxisId="right" orientation="right" width={72} tick={{fontSize: 10}} label={{value: 'Radius (mm)', angle: 90, position: 'right', offset: axisLabelOffset}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Line yAxisId="left" type={lineType} dataKey="area" stroke="#6366f1" strokeWidth={2} name="Area" />
                        <Line yAxisId="right" type={lineType} dataKey="radius" stroke="#10b981" strokeWidth={2} name="Radius" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <Activity size={16} className="text-emerald-500" />
                    Radial Texture Zonation
                  </h3>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={growthData} margin={singleAxisChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="day" tick={{fontSize: 10}} label={{value: 'Time (days)', position: 'insideBottom', offset: -4}} />
                        <YAxis width={78} tick={{fontSize: 10}} label={{value: 'Texture SD (a.u.)', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Line type={lineType} dataKey="core" stroke="#1b9e77" strokeWidth={2} name="Core" />
                        <Line type={lineType} dataKey="middle" stroke="#d95f02" strokeWidth={2} name="Middle" />
                        <Line type={lineType} dataKey="outer" stroke="#7570b3" strokeWidth={2} name="Outer" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <BarChart3 size={16} className="text-rose-500" />
                    Colony Morphogenesis
                  </h3>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={growthData} margin={dualAxisChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="day" tick={{fontSize: 10}} label={{value: 'Time (days)', position: 'insideBottom', offset: -4}} />
                        <YAxis yAxisId="left" width={72} tick={{fontSize: 10}} label={{value: 'Length (mm)', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <YAxis yAxisId="right" orientation="right" width={78} domain={[0, 1]} tick={{fontSize: 10}} label={{value: 'Circularity', angle: 90, position: 'right', offset: axisLabelOffset}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Line
                          yAxisId="left"
                          type={lineType}
                          dataKey="diameter"
                          stroke="#e11d48"
                          strokeWidth={2}
                          name="Diameter (mm)"
                        />
                        <Line
                          yAxisId="left"
                          type={lineType}
                          dataKey="perimeter"
                          stroke="#f97316"
                          strokeWidth={2}
                          name="Perimeter (mm)"
                        />
                        <Line
                          yAxisId="right"
                          type={lineType}
                          dataKey="circularity"
                          stroke="#7c3aed"
                          strokeWidth={2}
                          name="Circularity"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <AlertCircle size={16} className="text-amber-500" />
                    Stress Remodeling
                  </h3>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={growthData} margin={dualAxisChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="day" tick={{fontSize: 10}} label={{value: 'Time (days)', position: 'insideBottom', offset: -4}} />
                        <YAxis yAxisId="left" width={88} tick={{fontSize: 10}} label={{value: 'Coverage (%)', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <YAxis yAxisId="right" orientation="right" width={74} tick={{fontSize: 10}} label={{value: 'Crack Count', angle: 90, position: 'right', offset: axisLabelOffset}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Line
                          yAxisId="left"
                          type={lineType}
                          dataKey="crackCoverage"
                          stroke="#f59e0b"
                          strokeWidth={2}
                          name="Crack Coverage (%)"
                        />
                        <Line
                          yAxisId="left"
                          type={lineType}
                          dataKey="proportionalCrackCoverage"
                          stroke="#ef4444"
                          strokeWidth={2}
                          name="Proportional Crack Coverage (%)"
                        />
                        <Line
                          yAxisId="right"
                          type={lineType}
                          dataKey="crackCount"
                          stroke="#92400e"
                          strokeWidth={2}
                          name="Crack Count"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <Orbit size={16} className="text-cyan-500" />
                    Radial Differentiation
                  </h3>
                  <div className="relative h-[250px] pl-8">
                    <div className="pointer-events-none absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-slate-600">
                      Radial Texture Delta (a.u.)
                    </div>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={growthData} margin={singleAxisChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="day" tick={{fontSize: 10}} label={{value: 'Time (days)', position: 'insideBottom', offset: -4}} />
                        <YAxis width={74} tick={{fontSize: 10}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Line
                          type={lineType}
                          dataKey="outerCoreDelta"
                          stroke="#06b6d4"
                          strokeWidth={2}
                          name="Outer - Core Texture"
                        />
                        <Line
                          type={lineType}
                          dataKey="middleCoreDelta"
                          stroke="#0ea5e9"
                          strokeWidth={2}
                          name="Middle - Core Texture"
                        />
                        <Line
                          type={lineType}
                          dataKey="textureSpread"
                          stroke="#1d4ed8"
                          strokeWidth={2}
                          name="Texture Spread"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <ImageIcon size={16} className="text-fuchsia-500" />
                    Shape vs Stress Trajectory
                  </h3>
                  <div className="relative h-[250px] pl-8">
                    <div className="pointer-events-none absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-slate-600">
                      Crack Coverage (%)
                    </div>
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={scatterChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          type="number"
                          dataKey="eccentricity"
                          name="Eccentricity"
                          domain={[0, 1]}
                          tick={{fontSize: 10}}
                          label={{value: 'Eccentricity', position: 'insideBottom', offset: -4}}
                        />
                        <YAxis
                          type="number"
                          dataKey="crackCoverage"
                          name="Crack Coverage"
                          unit="%"
                          width={72}
                          tick={{fontSize: 10}}
                        />
                        <ZAxis type="number" dataKey="diameter" range={[60, 420]} name="Diameter" />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Scatter name="Shape-Stress State" data={growthData} fill="#d946ef" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <Activity size={16} className="text-amber-500" />
                    Growth Kinematics
                  </h3>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={growthData} margin={singleAxisChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="day" tick={{fontSize: 10}} label={{value: 'Time (days)', position: 'insideBottom', offset: -4}} />
                        <YAxis width={84} tick={{fontSize: 10}} label={{value: 'Rate / Acceleration', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Line type={lineType} dataKey="velocity" stroke="#1f78b4" strokeWidth={2} name="Velocity" />
                        <Line type={lineType} dataKey="acceleration" stroke="#33a02c" strokeWidth={2} name="Acceleration" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <ImageIcon size={16} className="text-indigo-500" />
                    Circularity vs Crack Coverage
                  </h3>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={scatterChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis type="number" dataKey="crackCoverage" name="Crack Coverage" unit="%" tick={{fontSize: 10}} label={{value: 'Crack Coverage (%)', position: 'insideBottom', offset: -4}} />
                        <YAxis type="number" dataKey="circularity" name="Circularity" domain={[0, 1]} width={76} tick={{fontSize: 10}} label={{value: 'Circularity', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <ZAxis type="number" dataKey="day" range={[50, 400]} name="Day" />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Scatter name="Colony State" data={growthData} fill="#6366f1" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <Activity size={16} className="text-teal-500" />
                    Texture Entropy and Center-Edge Intensity
                  </h3>
                  <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={growthData} margin={dualAxisChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="day" tick={{fontSize: 10}} label={{value: 'Time (days)', position: 'insideBottom', offset: -4}} />
                        <YAxis yAxisId="left" width={82} tick={{fontSize: 10}} label={{value: 'Entropy (bits)', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <YAxis yAxisId="right" orientation="right" width={90} tick={{fontSize: 10}} label={{value: 'Edge - Center Intensity', angle: 90, position: 'right', offset: axisLabelOffset}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Line yAxisId="left" type={lineType} dataKey="entropy" stroke="#0f766e" strokeWidth={2} name="Texture Entropy" />
                        <Line yAxisId="right" type={lineType} dataKey="centerEdgeDelta" stroke="#14b8a6" strokeWidth={2} name="Center-to-Edge Intensity" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                    <AlertCircle size={16} className="text-orange-500" />
                    Relative Growth and Edge Roughness
                  </h3>
                  <div className="relative h-[250px] pl-8 pr-8">
                    <div className="pointer-events-none absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-slate-600">
                      Relative Growth (1/day)
                    </div>
                    <div className="pointer-events-none absolute right-0 top-1/2 translate-y-[-50%] rotate-90 text-[10px] text-slate-600">
                      Edge Roughness Index
                    </div>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={growthData} margin={dualAxisChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="day" tick={{fontSize: 10}} label={{value: 'Time (days)', position: 'insideBottom', offset: -4}} />
                        <YAxis yAxisId="left" width={74} tick={{fontSize: 10}} />
                        <YAxis yAxisId="right" orientation="right" width={70} tick={{fontSize: 10}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Legend {...chartLegendProps} />
                        <Line yAxisId="left" type={lineType} dataKey="relativeGrowthRate" stroke="#ea580c" strokeWidth={2} name="Relative Growth Rate" />
                        <Line yAxisId="right" type={lineType} dataKey="edgeRoughness" stroke="#b45309" strokeWidth={2} name="Edge Roughness" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>

              <div className="grid grid-cols-3 gap-6">
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-4 text-sm font-bold text-slate-800">Area Distribution</h3>
                  <div className="h-[220px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={areaHistogram} margin={distributionChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="label" tick={{fontSize: 9}} angle={-15} textAnchor="end" height={50} label={{value: 'Area bins (mm²)', position: 'insideBottom', offset: -2}} />
                        <YAxis width={84} tick={{fontSize: 10}} label={{value: 'Image count', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Bar dataKey="count" fill="#6366f1" name="Images" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-4 text-sm font-bold text-slate-800">Growth Rate Distribution</h3>
                  <div className="h-[220px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={growthRateHistogram} margin={distributionChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="label" tick={{fontSize: 9}} angle={-15} textAnchor="end" height={50} label={{value: 'Growth-rate bins (mm²/day)', position: 'insideBottom', offset: -2}} />
                        <YAxis width={84} tick={{fontSize: 10}} label={{value: 'Image count', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Bar dataKey="count" fill="#10b981" name="Images" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="mb-4 text-sm font-bold text-slate-800">Entropy Distribution</h3>
                  <div className="h-[220px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={entropyHistogram} margin={distributionChartMargin}>
                        <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="label" tick={{fontSize: 9}} angle={-15} textAnchor="end" height={50} label={{value: 'Entropy bins (bits)', position: 'insideBottom', offset: -2}} />
                        <YAxis width={84} tick={{fontSize: 10}} label={{value: 'Image count', angle: -90, position: 'left', offset: axisLabelOffset}} />
                        <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                        <Bar dataKey="count" fill="#14b8a6" name="Images" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>

              <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <h3 className="mb-6 flex items-center gap-2 text-sm font-bold text-slate-800">
                  <ImageIcon size={16} className="text-sky-500" />
                  Area vs Circularity
                </h3>
                <div className="h-[250px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={scatterChartMargin}>
                      <CartesianGrid stroke="#f1f5f9" strokeDasharray="3 3" vertical={false} />
                      <XAxis type="number" dataKey="area" name="Area" unit=" mm²" tick={{fontSize: 10}} label={{value: 'Colony Area (mm²)', position: 'insideBottom', offset: -4}} />
                      <YAxis type="number" dataKey="circularity" name="Circularity" domain={[0, 1]} width={76} tick={{fontSize: 10}} label={{value: 'Circularity', angle: -90, position: 'left', offset: axisLabelOffset}} />
                      <ZAxis type="number" dataKey="day" range={[50, 300]} name="Day" />
                      <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                      <Scatter name="Area-Circularity State" data={growthData} fill="#0ea5e9" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="flex items-center justify-between border-b border-slate-100 p-6">
                  <h3 className="flex items-center gap-2 font-bold text-slate-800">
                    <FileText size={18} className="text-indigo-500" />
                    Analysis Summary & QC
                  </h3>
                  {latestRun && (
                    <a
                      href={latestRun.analysisCsv}
                      target="_blank"
                      rel="noreferrer"
                      className="flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium transition-colors hover:bg-slate-50"
                    >
                      <Download size={16} />
                      Open CSV
                    </a>
                  )}
                </div>
                <div className="border-b border-slate-100 px-6 py-4 text-xs text-slate-500">
                  Click any row to inspect the saved output interactively, including the overlay drawn from `analysis.json`.
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-slate-50 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                      <tr>
                        <th className="px-6 py-4">Day</th>
                        <th className="px-6 py-4">Filename</th>
                        <th className="px-6 py-4">Area (mm²)</th>
                        <th className="px-6 py-4">Circularity</th>
                        <th className="px-6 py-4">Cracks</th>
                        <th className="px-6 py-4">QC</th>
                        <th className="px-6 py-4"></th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {results.map((result) => (
                        <tr
                          key={result.id}
                          className="group cursor-pointer transition-colors hover:bg-slate-50"
                          onClick={() => setSelectedResult(result)}
                        >
                          <td className="px-6 py-4 font-bold text-indigo-600">D{result.day}</td>
                          <td className="px-6 py-4 font-mono text-xs text-slate-500">{result.filename}</td>
                          <td className="px-6 py-4 font-medium">
                            {result.morphology.areaMm2.toLocaleString(undefined, {maximumFractionDigits: 2})}
                          </td>
                          <td className="px-6 py-4">{result.morphology.circularity.toFixed(3)}</td>
                          <td className="px-6 py-4">
                            <span
                              className={cn(
                                'rounded-full px-2 py-1 text-[10px] font-bold',
                                result.cracks.count > 0 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-500',
                              )}
                            >
                              {result.cracks.count} detected
                            </span>
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-1.5">
                              {result.qcStatus === 'pass' ? (
                                <CheckCircle2 size={14} className="text-emerald-500" />
                              ) : (
                                <AlertCircle size={14} className="text-amber-500" />
                              )}
                              <span className="capitalize font-medium">{result.qcStatus}</span>
                            </div>
                          </td>
                          <td className="px-6 py-4 text-right">
                            <ChevronRight size={16} className="text-slate-300 transition-colors group-hover:text-indigo-500" />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          ) : (
            <div className="flex h-[600px] flex-col items-center justify-center space-y-6 rounded-3xl border-2 border-dashed border-slate-200 bg-white text-center">
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-slate-50">
                <FolderOpen size={32} className="text-slate-300" />
              </div>
              <div className="max-w-xl">
                <h3 className="text-xl font-bold text-slate-800">Ready for root-folder analysis</h3>
                <p className="mt-2 text-slate-500">
                  Put petri dish images in `input_images/`, choose Local Gemma 4 or Gemini API, and run the pipeline. Results and server-generated exports will be written into `outputs/`.
                </p>
              </div>
            </div>
          )}
        </div>
      </main>

      <AnimatePresence>
        {selectedResult && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
            <motion.div
              initial={{opacity: 0}}
              animate={{opacity: 1}}
              exit={{opacity: 0}}
              onClick={() => setSelectedResult(null)}
              className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
            />
            <motion.div
              initial={{scale: 0.95, opacity: 0, y: 20}}
              animate={{scale: 1, opacity: 1, y: 0}}
              exit={{scale: 0.95, opacity: 0, y: 20}}
              className="relative flex h-[80vh] w-full max-w-5xl overflow-hidden rounded-3xl bg-white shadow-2xl"
            >
              <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-slate-900">
                <AnalysisOverlay
                  imageUrl={selectedResult.imageUrl}
                  analysis={selectedResult.rawAnalysis}
                  showDish={overlayConfig.dish}
                  showMask={overlayConfig.mask}
                  showCracks={overlayConfig.cracks}
                  showROI={overlayConfig.roi}
                />

                <div className="absolute left-6 top-6 flex max-w-[70%] flex-wrap gap-2">
                  {(['dish', 'mask', 'cracks', 'roi'] as const).map((key, index) => (
                    <button
                      key={key}
                      onClick={() => setOverlayConfig((current) => ({...current, [key]: !current[key]}))}
                      className={cn(
                        'rounded-full border px-3 py-1 text-[10px] font-bold backdrop-blur-md transition-all',
                        overlayConfig[key]
                          ? 'border-indigo-400 bg-indigo-500 text-white'
                          : 'border-white/10 bg-black/40 text-slate-400',
                      )}
                    >
                      {index + 1}. {key.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex w-96 flex-col border-l border-slate-100 bg-white">
                <div className="border-b border-slate-100 p-6">
                  <div className="mb-2 flex items-center justify-between">
                    <h4 className="text-lg font-bold">Analysis Details</h4>
                    <button
                      onClick={() => setSelectedResult(null)}
                      className="rounded-lg p-1 transition-colors hover:bg-slate-100"
                    >
                      <AlertCircle size={20} className="rotate-45 text-slate-400" />
                    </button>
                  </div>
                  <p className="text-xs font-mono text-slate-400">{selectedResult.filename}</p>
                </div>

                <div className="custom-scrollbar flex-1 space-y-8 overflow-y-auto p-6">
                  <div>
                    <h5 className="mb-4 text-[10px] font-bold uppercase tracking-widest text-slate-400">
                      Morphology Measurements
                    </h5>
                    <div className="grid grid-cols-2 gap-4">
                      <MetricCard label="Area" value={`${selectedResult.morphology.areaMm2.toFixed(2)} mm²`} />
                      <MetricCard label="Radius" value={`${selectedResult.morphology.equivalentRadiusMm.toFixed(2)} mm`} />
                      <MetricCard label="Perimeter" value={`${selectedResult.morphology.perimeterMm.toFixed(2)} mm`} />
                      <MetricCard label="Circularity" value={selectedResult.morphology.circularity.toFixed(3)} />
                      <MetricCard label="Eccentricity" value={selectedResult.morphology.eccentricity.toFixed(3)} />
                      <MetricCard label="Edge Roughness" value={selectedResult.morphology.edgeRoughness.toFixed(3)} />
                    </div>
                  </div>

                  <SegmentationHealth result={selectedResult} />

                  <div>
                    <h5 className="mb-4 text-[10px] font-bold uppercase tracking-widest text-slate-400">
                      GLCM Texture Features
                    </h5>
                    <div className="space-y-3">
                      {Object.entries(selectedResult.texture).map(([key, value]) => {
                        if (key === 'radialZonation') {
                          return null;
                        }
                        return (
                          <div key={key} className="flex items-center justify-between">
                            <span className="text-xs capitalize text-slate-600">{key}</span>
                            <span className="text-xs font-mono font-bold">{(value as number).toFixed(3)}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <PlateFeatureHeatmap result={selectedResult} />

                  <div>
                    <h5 className="mb-4 text-[10px] font-bold uppercase tracking-widest text-slate-400">
                      Radial Zonation
                    </h5>
                    <div className="grid grid-cols-3 gap-2">
                      <MetricCard label="Core" value={selectedResult.texture.radialZonation.core.toFixed(1)} compact />
                      <MetricCard label="Middle" value={selectedResult.texture.radialZonation.middle.toFixed(1)} compact />
                      <MetricCard label="Outer" value={selectedResult.texture.radialZonation.outer.toFixed(1)} compact />
                    </div>
                  </div>

                  {selectedResult.rawAnalysis?.radial_profile?.radiusMm?.length ? (
                    <div>
                      <h5 className="mb-4 text-[10px] font-bold uppercase tracking-widest text-slate-400">
                        Radial Intensity Profile
                      </h5>
                      <div className="h-[180px] rounded-xl bg-slate-50 p-2">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            margin={{top: 8, right: 14, left: 30, bottom: 22}}
                            data={(selectedResult.rawAnalysis.radial_profile.radiusMm ?? []).map((radiusMm, index) => ({
                              radiusMm,
                              meanIntensity: selectedResult.rawAnalysis?.radial_profile?.meanIntensity?.[index] ?? 0,
                            }))}
                          >
                            <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
                            <XAxis
                              dataKey="radiusMm"
                              type="number"
                              tick={{fontSize: 10}}
                              label={{value: 'Radius from colony center (mm)', position: 'insideBottom', offset: -4}}
                            />
                            <YAxis
                              width={74}
                              tick={{fontSize: 10}}
                              label={{value: 'Mean intensity', angle: -90, position: 'left', offset: 8}}
                            />
                            <Tooltip contentStyle={{borderRadius: '12px', border: 'none', fontSize: 10}} />
                            <Line type="linear" dataKey="meanIntensity" stroke="#0f766e" strokeWidth={2} dot={false} name="Mean intensity" />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="mt-3 grid grid-cols-2 gap-3">
                        <MetricCard
                          label="Ring Spacing"
                          value={`${(selectedResult.rawAnalysis.radial_profile.ringSpacingMm ?? 0).toFixed(2)} mm`}
                          compact
                        />
                        <MetricCard
                          label="Density Index"
                          value={(selectedResult.texture.densityIndex ?? 0).toFixed(3)}
                          compact
                        />
                      </div>
                    </div>
                  ) : null}

                  <div className="rounded-2xl border border-amber-100 bg-amber-50 p-4">
                    <h5 className="mb-2 text-[10px] font-bold uppercase tracking-widest text-amber-700">
                      Crack Metrics
                    </h5>
                    <div className="mb-3 grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-[8px] uppercase text-amber-600">Coverage</p>
                        <p className="text-xs font-bold">{selectedResult.cracks.coveragePct.toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-[8px] uppercase text-amber-600">Prop. Coverage</p>
                        <p className="text-xs font-bold">
                          {selectedResult.cracks.proportionalCoveragePct.toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <p className="text-[10px] italic leading-relaxed text-amber-700">
                      {selectedResult.cracks.internalBandSummary}
                    </p>
                  </div>

                  <div>
                    <h5 className="mb-4 text-[10px] font-bold uppercase tracking-widest text-slate-400">QC Notes</h5>
                    <p className="text-xs leading-relaxed text-slate-600">{selectedResult.qcNotes}</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #e2e8f0;
          border-radius: 10px;
        }
      `}</style>
      <footer className="border-t border-slate-200 bg-white/80 px-6 py-5 text-center text-sm text-slate-500">
        <p>
          Designed by{' '}
          <a
            href="https://github.com/rotsl/grayleafspot"
            target="_blank"
            rel="noreferrer"
            className="font-medium text-indigo-600 hover:text-indigo-700"
          >
            Rohan R
          </a>
        </p>
        <p>Apache 2.0 License</p>
      </footer>
    </div>
  );
}

function buildHistogram(values: number[], bins: number, unit: string) {
  if (values.length === 0) {
    return [];
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    return [{label: `${min.toFixed(2)} ${unit}`, count: values.length}];
  }
  const step = (max - min) / bins;
  const counts = Array.from({length: bins}, () => 0);
  for (const value of values) {
    const rawIndex = Math.floor((value - min) / step);
    const index = Math.max(0, Math.min(bins - 1, rawIndex));
    counts[index] += 1;
  }
  return counts.map((count, index) => {
    const start = min + index * step;
    const end = start + step;
    return {
      label: `${start.toFixed(1)}-${end.toFixed(1)}`,
      count,
    };
  });
}

function MetricCard({label, value, compact = false}: {label: string; value: string; compact?: boolean}) {
  return (
    <div className={cn('rounded-xl bg-slate-50 p-3', compact && 'p-2 text-center')}>
      <p className="mb-1 text-[10px] text-slate-500">{label}</p>
      <p className={cn('text-sm font-bold', compact && 'text-xs')}>{value}</p>
    </div>
  );
}

function getPlateFeatureHeatmapRows(result: AnalysisResult) {
  return [
    {
      label: 'Morphology',
      cells: [
        {label: 'Area (mm²)', value: result.morphology.areaMm2},
        {label: 'Diameter (mm)', value: result.morphology.diameterMm},
        {label: 'Circularity', value: result.morphology.circularity},
        {label: 'Roughness', value: result.morphology.edgeRoughness},
      ],
    },
    {
      label: 'Texture',
      cells: [
        {label: 'Entropy', value: result.texture.entropy},
        {label: 'Center-Edge', value: result.texture.centerToEdgeDelta},
        {label: 'Density', value: result.texture.densityIndex},
        {label: 'Contrast', value: result.texture.contrast},
      ],
    },
    {
      label: 'Stress',
      cells: [
        {label: 'Crack cov. (%)', value: result.cracks.coveragePct},
        {label: 'Prop. crack (%)', value: result.cracks.proportionalCoveragePct},
        {label: 'Crack count', value: result.cracks.count},
        {label: 'Ring spacing (mm)', value: result.rawAnalysis?.radial_profile?.ringSpacingMm ?? 0},
      ],
    },
    {
      label: 'Kinematics',
      cells: [
        {label: 'Velocity', value: result.kinematics.radialVelocity},
        {label: 'Area rate', value: result.kinematics.areaGrowthRate},
        {label: 'Rel. growth', value: result.kinematics.relativeGrowthRate},
        {label: 'Acceleration', value: result.kinematics.radialAcceleration},
      ],
    },
  ];
}

function PlateFeatureHeatmap({result}: {result: AnalysisResult}) {
  const rows = getPlateFeatureHeatmapRows(result);
  const maxAbs = Math.max(
    1,
    ...rows.flatMap((row) => row.cells.map((cell) => Math.abs(cell.value))),
  );

  const colorFor = (value: number) => {
    const intensity = Math.min(1, Math.abs(value) / maxAbs);
    if (value >= 0) {
      return {
        backgroundColor: `rgba(20, 184, 166, ${0.12 + intensity * 0.55})`,
        color: intensity > 0.6 ? '#ffffff' : '#134e4a',
      };
    }
    return {
      backgroundColor: `rgba(244, 63, 94, ${0.12 + intensity * 0.55})`,
      color: intensity > 0.6 ? '#ffffff' : '#881337',
    };
  };

  return (
    <div>
      <div className="mb-3">
        <h5 className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
          Plate Feature Heatmap
        </h5>
        <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
          Single-plate overview of morphology, texture, stress, and growth features. Higher absolute values
          are shown with stronger color for this plate only.
        </p>
      </div>
      <div className="overflow-x-auto">
        <div
          className="grid min-w-[320px] gap-2"
          style={{gridTemplateColumns: `92px repeat(${rows[0]?.cells.length ?? 0}, minmax(64px, 1fr))`}}
        >
          <div />
          {rows[0].cells.map((cell) => (
            <div key={cell.label} className="px-1 text-center text-[10px] font-bold text-slate-500">
              {cell.label}
            </div>
          ))}
          {rows.map((row) => (
            <React.Fragment key={row.label}>
              <div className="flex items-center px-1 text-[10px] font-bold text-slate-500">{row.label}</div>
              {row.cells.map((cell) => (
                <div
                  key={`${row.label}-${cell.label}`}
                  className="rounded-lg px-2 py-3 text-center text-[10px] font-bold"
                  style={colorFor(cell.value)}
                  title={`${row.label} | ${cell.label}: ${cell.value.toFixed(3)}`}
                >
                  {cell.value.toFixed(2)}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

function SegmentationHealth({result}: {result: AnalysisResult}) {
  const diagnostics = result.rawAnalysis?.segmentation_diagnostics;
  const crackAnalysis = result.rawAnalysis?.crack_analysis;
  if (!diagnostics) {
    return null;
  }

  const priorIou = diagnostics.refinement_iou_with_model_prior ?? 0;
  const areaRatio = diagnostics.refinement_area_ratio ?? 0;
  const centroidShift = diagnostics.refinement_centroid_shift_px ?? 0;
  const classicalUnetIou = diagnostics.classical_unet_iou ?? 0;
  const hybridStrategy = diagnostics.hybrid_strategy ?? 'n/a';
  const samDecision = diagnostics.sam_decision ?? 'not_run';

  let statusLabel = 'Stable';
  let statusClasses = 'bg-emerald-100 text-emerald-700';
  if (
    priorIou < 0.55 ||
    centroidShift > 60 ||
    areaRatio > 1.6 ||
    areaRatio < 0.65 ||
    classicalUnetIou < 0.35
  ) {
    statusLabel = 'Review';
    statusClasses = 'bg-amber-100 text-amber-700';
  }
  if (
    priorIou < 0.35 ||
    centroidShift > 100 ||
    areaRatio > 2.0 ||
    areaRatio < 0.45 ||
    classicalUnetIou < 0.2
  ) {
    statusLabel = 'Low Confidence';
    statusClasses = 'bg-rose-100 text-rose-700';
  }

  return (
    <div className="rounded-2xl border border-indigo-100 bg-indigo-50/70 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h5 className="text-[10px] font-bold uppercase tracking-widest text-indigo-700">
          Segmentation Confidence
        </h5>
        <span className={cn('rounded-full px-2 py-1 text-[10px] font-bold', statusClasses)}>
          {statusLabel}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <MetricCard label="Prior IoU" value={priorIou.toFixed(3)} compact />
        <MetricCard label="Classical vs U-Net" value={classicalUnetIou.toFixed(3)} compact />
        <MetricCard label="Area Ratio" value={areaRatio.toFixed(2)} compact />
        <MetricCard label="Centroid Shift" value={`${centroidShift.toFixed(1)} px`} compact />
        <MetricCard
          label="Dish Diameter"
          value={`${(diagnostics.petri_dish_diameter_mm ?? 90).toFixed(0)} mm`}
          compact
        />
      </div>

      <div className="mt-4 space-y-2 text-xs text-slate-600">
        <div className="flex items-center justify-between">
          <span>Hybrid strategy</span>
          <span className="font-mono font-bold">{hybridStrategy}</span>
        </div>
        <div className="flex items-center justify-between">
          <span>SAM decision</span>
          <span className="font-mono font-bold">{samDecision}</span>
        </div>
        <div className="flex items-center justify-between">
          <span>Model-prior mask area</span>
          <span className="font-mono font-bold">
            {(diagnostics.initial_mask_area_px ?? 0).toLocaleString()} px
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span>Classical mask area</span>
          <span className="font-mono font-bold">
            {(diagnostics.classical_mask_area_px ?? 0).toLocaleString()} px
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span>U-Net mask area</span>
          <span className="font-mono font-bold">
            {(diagnostics.unet_mask_area_px ?? 0).toLocaleString()} px
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span>Pre-SAM hybrid area</span>
          <span className="font-mono font-bold">
            {(diagnostics.hybrid_pre_sam_area_px ?? 0).toLocaleString()} px
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span>Refined mask area</span>
          <span className="font-mono font-bold">
            {(diagnostics.refined_mask_area_px ?? 0).toLocaleString()} px
          </span>
        </div>
        {crackAnalysis && (
          <>
            <div className="flex items-center justify-between">
              <span>Internal crack band</span>
              <span className="font-mono font-bold">
                {(crackAnalysis.analysis_band_mm ?? 0).toFixed(2)} mm
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span>Crack segments</span>
              <span className="font-mono font-bold">
                {(crackAnalysis.num_segments ?? 0).toLocaleString()}
              </span>
            </div>
          </>
        )}
      </div>

      <p className="mt-4 text-[11px] leading-relaxed text-slate-500">
        This tracks the full local hybrid path: Gemma prior, classical segmentation, U-Net agreement,
        step-8 strategy selection, SAM refinement, and internal-band crack analysis.
      </p>
    </div>
  );
}
