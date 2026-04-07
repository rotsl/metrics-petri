import type {
  AnalysisEngine,
  AnalysisJob,
  AnalysisRun,
  InputImage,
  OutputRunPayload,
  ReportBundle,
} from '../types';

interface AnalyzeStartResponse {
  jobId: string;
  status: AnalysisJob['status'];
  progress: AnalysisJob['progress'];
}

interface OutputsResponse {
  runs: AnalysisRun[];
}

interface StopServerResponse {
  stoppedPort: number;
  stoppedPorts?: number[];
  restartCommand: string;
  message: string;
}

export async function listInputImages(): Promise<InputImage[]> {
  const response = await fetch('/api/images');
  if (!response.ok) {
    throw new Error(`Failed to list input images: ${response.statusText}`);
  }

  const payload = (await response.json()) as {images: {filename: string; imageUrl: string; day: number}[]};
  return payload.images.map((image) => ({
    ...image,
    selected: false,
  }));
}

export async function uploadInputImages(files: File[]) {
  const payloadFiles = await Promise.all(
    files.map(async (file) => ({
      name: file.name,
      contentBase64: await fileToBase64(file),
    })),
  );

  const response = await fetch('/api/images/upload', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({files: payloadFiles}),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || 'Failed to upload images');
  }

  return (await response.json()) as {uploaded: {filename: string; imageUrl: string}[]};
}

export async function startAnalysis(
  engine: AnalysisEngine,
  filenames: string[],
  experimentName?: string,
  tags?: string[],
  geminiApiKey?: string,
) {
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({engine, filenames, experimentName, tags, geminiApiKey}),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || 'Analysis failed');
  }

  return (await response.json()) as AnalyzeStartResponse;
}

export async function getAnalysisJob(jobId: string): Promise<AnalysisJob> {
  const response = await fetch(`/api/analyze/${encodeURIComponent(jobId)}`);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || `Failed to load analysis job ${jobId}`);
  }
  return (await response.json()) as AnalysisJob;
}

export async function getLatestActiveAnalysisJob(): Promise<AnalysisJob | null> {
  const response = await fetch('/api/analyze/active');
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || 'Failed to load active analysis job');
  }
  return (await response.json()) as AnalysisJob;
}

async function postJobAction(jobId: string, action: 'pause' | 'resume' | 'stop'): Promise<AnalysisJob> {
  const response = await fetch(`/api/analyze/${encodeURIComponent(jobId)}/${action}`, {
    method: 'POST',
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || `Failed to ${action} analysis job ${jobId}`);
  }
  return (await response.json()) as AnalysisJob;
}

export function pauseAnalysisJob(jobId: string) {
  return postJobAction(jobId, 'pause');
}

export function resumeAnalysisJob(jobId: string) {
  return postJobAction(jobId, 'resume');
}

export function stopAnalysisJob(jobId: string) {
  return postJobAction(jobId, 'stop');
}

export async function stopLocalApp(): Promise<StopServerResponse> {
  const response = await fetch('/api/server/stop', {
    method: 'POST',
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || 'Failed to stop local app');
  }
  return (await response.json()) as StopServerResponse;
}

export async function listOutputRuns(): Promise<AnalysisRun[]> {
  const response = await fetch('/api/outputs');
  if (!response.ok) {
    throw new Error(`Failed to list outputs: ${response.statusText}`);
  }

  const payload = (await response.json()) as OutputsResponse;
  return payload.runs;
}

export async function getOutputRun(runId: string): Promise<OutputRunPayload> {
  const response = await fetch(`/api/outputs/${encodeURIComponent(runId)}`);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || `Failed to load output run ${runId}`);
  }

  return (await response.json()) as OutputRunPayload;
}

export async function deleteOutputRuns(runIds: string[]) {
  const response = await fetch('/api/outputs/delete', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({runIds}),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || 'Failed to delete output runs');
  }
  return (await response.json()) as {deleted: string[]};
}

export async function archiveOutputRuns(runIds: string[]) {
  const response = await fetch('/api/outputs/archive', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({runIds}),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || 'Failed to archive output runs');
  }
  return (await response.json()) as {archived: string[]; archivedAs: Record<string, string>};
}

export async function generateOutputReport(runId: string, experimentName?: string, tags?: string[]): Promise<ReportBundle> {
  const response = await fetch(`/api/outputs/${encodeURIComponent(runId)}/report`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({experimentName, tags}),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({error: response.statusText}));
    throw new Error(payload.error || `Failed to generate report for ${runId}`);
  }
  return (await response.json()) as ReportBundle;
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === 'string' ? reader.result : '';
      const [, base64 = ''] = result.split(',', 2);
      resolve(base64);
    };
    reader.onerror = () => {
      reject(reader.error ?? new Error(`Failed to read ${file.name}`));
    };
    reader.readAsDataURL(file);
  });
}
