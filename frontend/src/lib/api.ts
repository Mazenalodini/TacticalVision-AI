/**
 * TacticalVision AI — API client.
 *
 * All requests go directly to FastAPI (no Next.js proxy) so that
 * HTTP 206 byte-range headers pass through cleanly to <video>.
 */

import type { MatchCard, Telemetry, JobStatus } from "./types";

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
