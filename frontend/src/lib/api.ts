/**
 * TacticalVision AI — API client.
 *
 * All requests go directly to FastAPI (no Next.js proxy) so that
 * HTTP 206 byte-range headers pass through cleanly to <video>.
 */

import type {
  MatchCard,
  Telemetry,
  JobStatus,
  ModelsSuiteResponse,
  ActionSpottingMatchesResponse,
  ActionSpottingAnalyzeResponse,
} from "./types";

export const API_BASE =
  typeof window !== "undefined"
    ? (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000")
    : (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000");

// ── Match listing ────────────────────────────────────────────────

export async function fetchMatches(): Promise<MatchCard[]> {
  const res = await fetch(`${API_BASE}/api/matches`);
  if (!res.ok) throw new Error(`Failed to fetch matches: ${res.status}`);
  return res.json();
}

export async function fetchMatch(matchId: string): Promise<MatchCard> {
  const res = await fetch(`${API_BASE}/api/matches/${matchId}`);
  if (!res.ok) throw new Error(`Failed to fetch match: ${res.status}`);
  return res.json();
}

// ── Full telemetry ───────────────────────────────────────────────

export async function fetchTelemetry(matchId: string): Promise<Telemetry> {
  const res = await fetch(`${API_BASE}/api/matches/${matchId}/telemetry`);
  if (!res.ok) throw new Error(`Failed to fetch telemetry: ${res.status}`);
  return res.json();
}

// ── URL builders (used as src attributes, not fetched directly) ──

export function getVideoUrl(
  matchId: string,
  type: "broadcast" | "radar",
): string {
  return `${API_BASE}/api/matches/${matchId}/video/${type}`;
}

export function getThumbnailUrl(matchId: string): string {
  return `${API_BASE}/api/matches/${matchId}/thumbnail`;
}

export function getExportUrl(
  matchId: string,
  format: "csv" | "json",
  scope: "players" | "events",
): string {
  return `${API_BASE}/api/matches/${matchId}/export?format=${format}&scope=${scope}`;
}

// ── Job management & Upload ──────────────────────────────────────

export async function uploadMatch(formData: FormData): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/api/matches/analyze`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `Error ${res.status}` }));
    throw new Error(errorData.detail || `Upload failed with status ${res.status}`);
  }
  return res.json();
}

export async function fetchJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `Error ${res.status}` }));
    throw new Error(errorData.detail || `Failed to fetch job status: ${res.status}`);
  }
  return res.json();
}

export async function cancelJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/cancel`, {
    method: "POST",
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `Error ${res.status}` }));
    throw new Error(errorData.detail || `Failed to cancel job: ${res.status}`);
  }
  return res.json();
}

export async function deleteMatch(matchId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/matches/${matchId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `Error ${res.status}` }));
    throw new Error(errorData.detail || `Failed to delete match: ${res.status}`);
  }
}

// ── 4-Model AI Suite & Action Spotting (Models 3 & 4) ────────────

export async function fetchModelsSuite(): Promise<ModelsSuiteResponse> {
  const res = await fetch(`${API_BASE}/api/models/suite`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch models suite: ${res.status}`);
  }
  return res.json();
}

export async function fetchActionSpottingMatches(): Promise<ActionSpottingMatchesResponse> {
  const res = await fetch(`${API_BASE}/api/action-spotting/matches`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch action spotting matches: ${res.status}`);
  }
  return res.json();
}

export async function fetchActionSpottingResult(
  runId: string,
): Promise<ActionSpottingAnalyzeResponse> {
  const res = await fetch(`${API_BASE}/api/action-spotting/results/${encodeURIComponent(runId)}`);
  if (!res.ok) {
    throw new Error(`Failed to load action spotting result: ${res.status}`);
  }
  return res.json();
}

export async function analyzeActionSpotting(params: {
  match_id?: string;
  half?: "half1" | "half2" | "both";
  min_conf?: number;
  nms_window_sec?: number;
  file?: File | null;
}): Promise<ActionSpottingAnalyzeResponse> {
  const fd = new FormData();
  if (params.match_id) fd.append("match_id", params.match_id);
  fd.append("half", params.half ?? "half1");
  fd.append("min_conf", String(params.min_conf ?? 0.35));
  fd.append("nms_window_sec", String(params.nms_window_sec ?? 10.0));
  if (params.file) fd.append("file", params.file);

  const res = await fetch(`${API_BASE}/api/action-spotting/analyze`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(errorData.detail || `Action spotting analysis failed (${res.status})`);
  }
  return res.json();
}
