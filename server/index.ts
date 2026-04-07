import crypto from 'node:crypto';
import {ChildProcessWithoutNullStreams, execSync, spawn, spawnSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import dotenv from 'dotenv';
import express from 'express';

dotenv.config({path: '.env'});

const app = express();

const rootDir = process.cwd();
const inputImagesDir = path.join(rootDir, 'input_images');
const outputsDir = path.join(rootDir, 'outputs');
const archivesDir = path.join(rootDir, 'archives');
const pythonPath = path.join(rootDir, 'venv', 'bin', 'python');
const port = Number.parseInt(process.env.API_PORT || '8000', 10);
const frontendPort = Number.parseInt(process.env.FRONTEND_PORT || '3000', 10);

fs.mkdirSync(inputImagesDir, {recursive: true});
fs.mkdirSync(outputsDir, {recursive: true});
fs.mkdirSync(archivesDir, {recursive: true});

app.use(express.json({limit: '80mb'}));
app.use('/input_images', express.static(inputImagesDir));
app.use('/outputs', express.static(outputsDir));

type AnalysisJobStatus = 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';

interface AnalysisJob {
  id: string;
  engine: 'local' | 'gemini';
  filenames: string[];
  experimentName?: string;
  tags?: string[];
  status: AnalysisJobStatus;
  createdAt: string;
  updatedAt: string;
  progress: {
    current: number;
    total: number;
    stage: string;
  };
  logs: string[];
  pid?: number;
  result?: unknown;
  error?: string;
}

function slugifySegment(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
}

function normalizeTags(tags: unknown): string[] {
  if (!Array.isArray(tags)) {
    return [];
  }
  return [...new Set(tags.filter((item): item is string => typeof item === 'string').map((item) => item.trim()).filter(Boolean))];
}

function readRunManifest(runDir: string) {
  return readJsonFile<Record<string, unknown>>(path.join(runDir, 'manifest.json'));
}

function writeRunMetadata(runDir: string, metadata: {experimentName?: string; tags?: string[]}) {
  const manifestPath = path.join(runDir, 'manifest.json');
  const manifest = readJsonFile<Record<string, unknown>>(manifestPath);
  if (!manifest) {
    return;
  }
  const nextMetadata = {
    ...((manifest.metadata as Record<string, unknown> | undefined) ?? {}),
    experiment_name: metadata.experimentName ?? '',
    tags: metadata.tags ?? [],
  };
  manifest.metadata = nextMetadata;
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
}

const analysisJobs = new Map<string, AnalysisJob>();
const jobProcesses = new Map<string, ChildProcessWithoutNullStreams>();
let shuttingDown = false;

function isImageFile(filename: string) {
  return /\.(png|jpe?g|bmp|tif?f|webp)$/i.test(filename);
}

function sanitizeUploadFilename(filename: string) {
  const parsed = path.parse(filename);
  const safeBase = parsed.name.replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'image';
  const safeExt = parsed.ext.toLowerCase();
  return `${safeBase}${safeExt}`;
}

function resolveUniqueUploadPath(filename: string) {
  const safeName = sanitizeUploadFilename(filename);
  const parsed = path.parse(safeName);
  let candidate = path.join(inputImagesDir, safeName);
  let index = 1;
  while (fs.existsSync(candidate)) {
    candidate = path.join(inputImagesDir, `${parsed.name}_${index}${parsed.ext}`);
    index += 1;
  }
  return candidate;
}

function extractExplicitDay(filename: string) {
  const match =
    filename.match(/day(\d+)/i) ??
    filename.match(/d(\d+)/i);
  return match ? Number.parseInt(match[1], 10) : null;
}

function inferImageDays(filenames: string[]) {
  const explicit = new Map<string, number>();
  const inferred: string[] = [];
  for (const filename of filenames) {
    const day = extractExplicitDay(filename);
    if (day == null) {
      inferred.push(filename);
    } else {
      explicit.set(filename, day);
    }
  }

  const usedDays = new Set(explicit.values());
  let nextDay = 1;
  for (const filename of inferred) {
    while (usedDays.has(nextDay)) {
      nextDay += 1;
    }
    explicit.set(filename, nextDay);
    usedDays.add(nextDay);
    nextDay += 1;
  }
  return explicit;
}

function isOutputRunDirectory(entry: fs.Dirent) {
  return entry.isDirectory() && /\d{8}T\d{6}Z_(local|gemini)$/.test(entry.name);
}

function readJsonFile<T>(filePath: string): T | null {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(filePath, 'utf-8')) as T;
}

function appendJobLog(job: AnalysisJob, line: string) {
  const text = line.trimEnd();
  if (!text.trim()) {
    return;
  }
  job.logs.push(text);
  if (job.logs.length > 300) {
    job.logs = job.logs.slice(-300);
  }
  job.updatedAt = new Date().toISOString();
}

function getMutableJob(jobId: string) {
  const job = analysisJobs.get(jobId);
  if (!job) {
    throw new Error(`Analysis job ${jobId} was not found.`);
  }
  return job;
}

function requireActiveProcess(jobId: string) {
  const child = jobProcesses.get(jobId);
  if (!child || !child.pid) {
    throw new Error(`Analysis job ${jobId} is not attached to a running process.`);
  }
  return child;
}

function killProcessesListeningOnPort(targetPort: number, excludePid?: number) {
  try {
    const stdout = execSync(`lsof -ti tcp:${targetPort}`, {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    const pids = stdout
      .split(/\r?\n/)
      .map((value) => value.trim())
      .filter(Boolean)
      .map((value) => Number.parseInt(value, 10))
      .filter((value) => Number.isFinite(value) && value > 0 && value !== excludePid);
    for (const pid of pids) {
      try {
        process.kill(pid, 'SIGTERM');
      } catch {
        // ignore individual kill failures
      }
    }
  } catch {
    // no process found on that port
  }
}

function deleteRunDirectory(runId: string) {
  if (!/^\d{8}T\d{6}Z_(local|gemini)$/.test(runId)) {
    throw new Error(`Invalid output run id: ${runId}`);
  }
  const runDir = path.join(outputsDir, runId);
  if (!fs.existsSync(runDir)) {
    return false;
  }
  fs.rmSync(runDir, {recursive: true, force: true});
  return !fs.existsSync(runDir);
}

function archiveRunDirectory(runId: string) {
  if (!/^\d{8}T\d{6}Z_(local|gemini)$/.test(runId)) {
    throw new Error(`Invalid output run id: ${runId}`);
  }
  const runDir = path.join(outputsDir, runId);
  if (!fs.existsSync(runDir)) {
    return false;
  }
  let archiveName = runId;
  let archiveDir = path.join(archivesDir, archiveName);
  if (fs.existsSync(archiveDir)) {
    archiveName = `${runId}_${Date.now()}`;
    archiveDir = path.join(archivesDir, archiveName);
  }
  fs.renameSync(runDir, archiveDir);
  return archiveName;
}

app.get('/api/images', (_req, res) => {
  const filenames = fs
    .readdirSync(inputImagesDir, {withFileTypes: true})
    .filter((entry) => entry.isFile() && isImageFile(entry.name))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));
  const dayMap = inferImageDays(filenames);
  const images = filenames.map((filename) => ({
    filename,
    day: dayMap.get(filename) ?? 0,
    imageUrl: `/input_images/${encodeURIComponent(filename)}`,
  }));

  res.json({images});
});

app.post('/api/images/upload', (req, res) => {
  const files = Array.isArray(req.body?.files)
    ? req.body.files.filter(
        (item: unknown): item is {name: string; contentBase64: string} =>
          Boolean(item) &&
          typeof item === 'object' &&
          typeof (item as {name?: unknown}).name === 'string' &&
          typeof (item as {contentBase64?: unknown}).contentBase64 === 'string',
      )
    : [];

  if (files.length === 0) {
    res.status(400).json({error: 'files must contain at least one image payload'});
    return;
  }

  try {
    const uploaded: {filename: string; imageUrl: string}[] = [];
    for (const file of files) {
      if (!isImageFile(file.name)) {
        throw new Error(`Unsupported image type for ${file.name}`);
      }
      const destination = resolveUniqueUploadPath(file.name);
      const buffer = Buffer.from(file.contentBase64, 'base64');
      fs.writeFileSync(destination, buffer);
      uploaded.push({
        filename: path.basename(destination),
        imageUrl: `/input_images/${encodeURIComponent(path.basename(destination))}`,
      });
    }
    res.json({uploaded});
  } catch (error) {
    res.status(400).json({error: error instanceof Error ? error.message : 'Failed to upload images'});
  }
});

app.get('/api/outputs', (_req, res) => {
  const runs = fs
    .readdirSync(outputsDir, {withFileTypes: true})
    .filter(isOutputRunDirectory)
    .map((entry) => {
      const manifestPath = path.join(outputsDir, entry.name, 'manifest.json');
      const manifest = readJsonFile<{
        engine: 'local' | 'gemini';
        engine_model: string;
        created_at: string;
        output_dir: string;
        analysis_json: string;
        analysis_csv: string;
        metadata?: {
          experiment_name?: string;
          tags?: string[];
        };
      }>(manifestPath);

      if (!manifest) {
        return null;
      }

      return {
        id: entry.name,
        engine: manifest.engine,
        engineModel: manifest.engine_model,
        createdAt: manifest.created_at,
        outputDir: manifest.output_dir,
        analysisJson: manifest.analysis_json,
        analysisCsv: manifest.analysis_csv,
        experimentName: manifest.metadata?.experiment_name,
        tags: manifest.metadata?.tags ?? [],
      };
    })
    .filter((run): run is NonNullable<typeof run> => Boolean(run))
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));

  res.json({runs});
});

app.get('/api/outputs/:runId', (req, res) => {
  const runId = req.params.runId;
  if (!/^\d{8}T\d{6}Z_(local|gemini)$/.test(runId)) {
    res.status(400).json({error: 'Invalid output run id'});
    return;
  }

  const runDir = path.join(outputsDir, runId);
  const manifestPath = path.join(runDir, 'manifest.json');
  const analysisPath = path.join(runDir, 'analysis.json');
  const manifest = readJsonFile<{
    engine: 'local' | 'gemini';
    engine_model: string;
    created_at: string;
    output_dir: string;
    analysis_json: string;
    analysis_csv: string;
    metadata?: {
      experiment_name?: string;
      tags?: string[];
    };
  }>(manifestPath);
  const results = readJsonFile<unknown[]>(analysisPath);

  if (!manifest || !results) {
    res.status(404).json({error: `Output run ${runId} was not found.`});
    return;
  }

  res.json({
    run: {
      id: runId,
      engine: manifest.engine,
      engineModel: manifest.engine_model,
      createdAt: manifest.created_at,
      outputDir: manifest.output_dir,
      analysisJson: manifest.analysis_json,
      analysisCsv: manifest.analysis_csv,
      experimentName: manifest.metadata?.experiment_name,
      tags: manifest.metadata?.tags ?? [],
    },
    results,
  });
});

app.post('/api/outputs/:runId/report', (req, res) => {
  const runId = req.params.runId;
  if (!/^\d{8}T\d{6}Z_(local|gemini)$/.test(runId)) {
    res.status(400).json({error: 'Invalid output run id'});
    return;
  }

  const runDir = path.join(outputsDir, runId);
  if (!fs.existsSync(path.join(runDir, 'manifest.json')) || !fs.existsSync(path.join(runDir, 'analysis.json'))) {
    res.status(404).json({error: `Output run ${runId} was not found.`});
    return;
  }

  const existingManifest = readRunManifest(runDir);
  const experimentName =
    typeof req.body?.experimentName === 'string' && req.body.experimentName.trim()
      ? req.body.experimentName.trim()
      : typeof existingManifest?.metadata === 'object' &&
          existingManifest?.metadata &&
          typeof (existingManifest.metadata as Record<string, unknown>).experiment_name === 'string'
        ? String((existingManifest.metadata as Record<string, unknown>).experiment_name)
        : '';
  const tags = normalizeTags(req.body?.tags);
  const mergedTags = [...new Set([...(Array.isArray((existingManifest?.metadata as Record<string, unknown> | undefined)?.tags) ? ((existingManifest?.metadata as Record<string, unknown>).tags as string[]) : []), ...tags])];

  writeRunMetadata(runDir, {experimentName, tags: mergedTags});

  const reportArgs = ['-m', 'pipeline.reporting', '--run-dir', runDir, '--json'];
  if (experimentName) {
    reportArgs.push('--experiment-name', experimentName);
  }
  if (mergedTags.length > 0) {
    reportArgs.push('--tags', mergedTags.join(','));
  }

  const child = spawnSync(pythonPath, reportArgs, {
    cwd: rootDir,
    env: {
      ...process.env,
      MPLCONFIGDIR: path.join(rootDir, '.mplconfig'),
    },
    encoding: 'utf-8',
  });

  if (child.status !== 0) {
    res.status(500).json({
      error: child.stderr?.trim() || child.stdout?.trim() || `Report generation failed for ${runId}`,
    });
    return;
  }

  try {
    res.json(JSON.parse(child.stdout));
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to parse report generator output',
    });
  }
});

app.delete('/api/outputs/:runId', (req, res) => {
  try {
    const deleted = deleteRunDirectory(req.params.runId);
    if (!deleted) {
      res.status(404).json({error: `Output run ${req.params.runId} was not found.`});
      return;
    }
    res.json({deleted: [req.params.runId]});
  } catch (error) {
    res.status(400).json({error: error instanceof Error ? error.message : 'Failed to delete output run'});
  }
});

app.post('/api/outputs/delete', (req, res) => {
  const runIds = Array.isArray(req.body?.runIds)
    ? req.body.runIds.filter((item: unknown): item is string => typeof item === 'string')
    : [];

  if (runIds.length === 0) {
    res.status(400).json({error: 'runIds must contain at least one output run id'});
    return;
  }

  try {
    const deleted = runIds.filter((runId) => deleteRunDirectory(runId));
    res.json({deleted});
  } catch (error) {
    res.status(400).json({error: error instanceof Error ? error.message : 'Failed to delete output runs'});
  }
});

app.post('/api/outputs/:runId/archive', (req, res) => {
  try {
    const archivedAs = archiveRunDirectory(req.params.runId);
    if (!archivedAs) {
      res.status(404).json({error: `Output run ${req.params.runId} was not found.`});
      return;
    }
    res.json({archived: [req.params.runId], archivedAs: {[req.params.runId]: archivedAs}});
  } catch (error) {
    res.status(400).json({error: error instanceof Error ? error.message : 'Failed to archive output run'});
  }
});

app.post('/api/outputs/archive', (req, res) => {
  const runIds = Array.isArray(req.body?.runIds)
    ? req.body.runIds.filter((item: unknown): item is string => typeof item === 'string')
    : [];

  if (runIds.length === 0) {
    res.status(400).json({error: 'runIds must contain at least one output run id'});
    return;
  }

  try {
    const archived: string[] = [];
    const archivedAs: Record<string, string> = {};
    for (const runId of runIds) {
      const archiveName = archiveRunDirectory(runId);
      if (archiveName) {
        archived.push(runId);
        archivedAs[runId] = archiveName;
      }
    }
    res.json({archived, archivedAs});
  } catch (error) {
    res.status(400).json({error: error instanceof Error ? error.message : 'Failed to archive output runs'});
  }
});

app.post('/api/analyze', (req, res) => {
  const engine = req.body?.engine;
  const filenames = Array.isArray(req.body?.filenames)
    ? req.body.filenames.filter((item: unknown): item is string => typeof item === 'string')
    : [];
  const experimentName = typeof req.body?.experimentName === 'string' ? req.body.experimentName.trim() : '';
  const requestTags = normalizeTags(req.body?.tags);
  const geminiApiKey = typeof req.body?.geminiApiKey === 'string' ? req.body.geminiApiKey.trim() : '';

  if (engine !== 'local' && engine !== 'gemini') {
    res.status(400).json({error: 'engine must be "local" or "gemini"'});
    return;
  }

  const jobId = crypto.randomUUID();
  const job: AnalysisJob = {
    id: jobId,
    engine,
    filenames,
    experimentName,
    tags: requestTags,
    status: 'queued',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    progress: {
      current: 0,
      total: filenames.length,
      stage: 'Queued',
    },
    logs: [],
  };
  analysisJobs.set(jobId, job);

  const args = [
    '-m',
    'pipeline.cli',
    '--engine',
    engine,
    '--input-dir',
    inputImagesDir,
    '--output-dir',
    outputsDir,
    '--json',
  ];

  for (const filename of filenames) {
    args.push('--filename', filename);
  }

  const child = spawn(pythonPath, args, {
    cwd: rootDir,
    env: {
      ...process.env,
      ...(geminiApiKey ? {GEMINI_API_KEY: geminiApiKey} : {}),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  jobProcesses.set(jobId, child);

  job.status = 'running';
  job.progress.stage = 'Starting analysis';
  job.pid = child.pid;
  appendJobLog(
    job,
    `[job] Starting ${engine} analysis for ${filenames.length} image(s)${experimentName ? ` [${experimentName}]` : ''}`,
  );

  let stdoutBuffer = '';
  let stderrBuffer = '';

  child.stdout.setEncoding('utf-8');
  child.stderr.setEncoding('utf-8');

  child.stdout.on('data', (chunk: string) => {
    stdoutBuffer += chunk;
  });

  child.stderr.on('data', (chunk: string) => {
    stderrBuffer += chunk;
    const lines = stderrBuffer.split(/\r?\n/);
    stderrBuffer = lines.pop() ?? '';

    for (const line of lines) {
      appendJobLog(job, line);
      const match = line.match(/^\[progress\]\s+(\d+)\/(\d+)\s+(.*)$/i);
      if (match) {
        job.progress = {
          current: Number.parseInt(match[1], 10),
          total: Number.parseInt(match[2], 10),
          stage: match[3].trim(),
        };
      }
    }
  });

  child.on('close', (code) => {
    jobProcesses.delete(job.id);
    if (stderrBuffer.trim()) {
      appendJobLog(job, stderrBuffer.trim());
    }

    if (job.status === 'stopped') {
      job.progress = {
        current: job.progress.current,
        total: job.progress.total,
        stage: 'Stopped',
      };
      appendJobLog(job, '[job] Analysis stopped by user');
    } else if (code === 0) {
      try {
        const payload = JSON.parse(stdoutBuffer) as {
          run?: {
            outputDir?: string;
            id?: string;
            experimentName?: string;
            tags?: string[];
          };
        };
        const runId = payload.run?.outputDir?.split('/').filter(Boolean).pop();
        if (payload.run && runId) {
          const mergedTags = [...new Set([...requestTags, ...(process.platform === 'darwin' ? ['macbook'] : [])])];
          writeRunMetadata(path.join(outputsDir, runId), {experimentName, tags: mergedTags});
          payload.run = {
            ...payload.run,
            id: runId,
            experimentName,
            tags: mergedTags,
          };
        }
        job.status = 'completed';
        job.result = payload;
        job.progress = {
          current: job.progress.total,
          total: job.progress.total,
          stage: 'Completed',
        };
        appendJobLog(job, '[job] Analysis completed successfully');
      } catch (error) {
        job.status = 'failed';
        job.error = error instanceof Error ? error.message : 'Failed to parse analysis output';
        appendJobLog(job, `[job] ${job.error}`);
      }
    } else {
      job.status = 'failed';
      job.error = stderrBuffer.trim() || `Analysis failed with exit code ${code ?? 'unknown'}`;
      appendJobLog(job, `[job] ${job.error}`);
    }
    job.pid = undefined;
    job.updatedAt = new Date().toISOString();
  });

  res.status(202).json({
    jobId: job.id,
    status: job.status,
    progress: job.progress,
  });
});

app.get('/api/analyze/active', (_req, res) => {
  const activeJobs = [...analysisJobs.values()]
    .filter((job) => ['queued', 'running', 'paused'].includes(job.status))
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt));

  if (activeJobs.length === 0) {
    res.status(404).json({error: 'No active analysis job was found.'});
    return;
  }

  res.json(activeJobs[0]);
});

app.get('/api/analyze/:jobId', (req, res) => {
  const job = analysisJobs.get(req.params.jobId);
  if (!job) {
    res.status(404).json({error: `Analysis job ${req.params.jobId} was not found.`});
    return;
  }
  res.json(job);
});

app.post('/api/analyze/:jobId/pause', (req, res) => {
  try {
    const job = getMutableJob(req.params.jobId);
    if (job.status !== 'running') {
      res.status(400).json({error: `Only running jobs can be paused. Current status: ${job.status}`});
      return;
    }
    const child = requireActiveProcess(job.id);
    process.kill(child.pid!, 'SIGSTOP');
    job.status = 'paused';
    job.progress.stage = 'Paused';
    appendJobLog(job, '[job] Analysis paused by user');
    res.json(job);
  } catch (error) {
    res.status(400).json({error: error instanceof Error ? error.message : 'Failed to pause analysis job'});
  }
});

app.post('/api/analyze/:jobId/resume', (req, res) => {
  try {
    const job = getMutableJob(req.params.jobId);
    if (job.status !== 'paused') {
      res.status(400).json({error: `Only paused jobs can be resumed. Current status: ${job.status}`});
      return;
    }
    const child = requireActiveProcess(job.id);
    process.kill(child.pid!, 'SIGCONT');
    job.status = 'running';
    job.progress.stage = 'Resuming analysis';
    appendJobLog(job, '[job] Analysis resumed by user');
    res.json(job);
  } catch (error) {
    res.status(400).json({error: error instanceof Error ? error.message : 'Failed to resume analysis job'});
  }
});

app.post('/api/analyze/:jobId/stop', (req, res) => {
  try {
    const job = getMutableJob(req.params.jobId);
    if (!['queued', 'running', 'paused'].includes(job.status)) {
      res.status(400).json({error: `Only queued, running, or paused jobs can be stopped. Current status: ${job.status}`});
      return;
    }
    const child = requireActiveProcess(job.id);
    job.status = 'stopped';
    job.progress.stage = 'Stopping analysis';
    appendJobLog(job, '[job] Stop requested by user');
    process.kill(child.pid!, 'SIGTERM');
    res.json(job);
  } catch (error) {
    res.status(400).json({error: error instanceof Error ? error.message : 'Failed to stop analysis job'});
  }
});

const server = app.listen(port, '127.0.0.1', () => {
  console.log(`Magnaporthe API listening on http://127.0.0.1:${port}`);
});

app.post('/api/server/stop', (_req, res) => {
  const restartCommand = 'make run-app';
  res.json({
    stoppedPort: port,
    stoppedPorts: [frontendPort, port],
    restartCommand,
    message: `Frontend port ${frontendPort} and API port ${port} are shutting down. Restart with: ${restartCommand}`,
  });

  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  setTimeout(() => {
    killProcessesListeningOnPort(frontendPort, process.pid);
    server.close(() => {
      process.exit(0);
    });
  }, 250);
});
