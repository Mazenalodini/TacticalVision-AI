"use client";

import { useCallback, useEffect, useState } from "react";

export interface PlayerIdentityOverride {
  jersey_number?: number | null;
  custom_name?: string;
}

export type MatchOverridesMap = Record<number, PlayerIdentityOverride>;

const STORAGE_PREFIX = "tv_jersey_overrides_v1:";
const CHANGE_EVENT = "tv:jersey-overrides-changed";

export function loadMatchOverrides(matchId?: string): MatchOverridesMap {
  if (!matchId || typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(`${STORAGE_PREFIX}${matchId}`);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function saveMatchOverrides(
  matchId: string,
  map: MatchOverridesMap
): void {
  if (!matchId || typeof window === "undefined") return;
  try {
    window.localStorage.setItem(`${STORAGE_PREFIX}${matchId}`, JSON.stringify(map));
    window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: { matchId } }));
  } catch {
    // Ignore quota/storage errors
  }
}

/**
 * React hook keeping Coach jersey/name overrides synchronized across all War Room components
 * (2D Pitch Radar, Spotlight, Leaderboard, Pass Network, and Pro Event Feed).
 */
export function usePlayerOverrides(matchId?: string) {
  const [overrides, setOverrides] = useState<MatchOverridesMap>(() =>
    loadMatchOverrides(matchId)
  );

  useEffect(() => {
    setOverrides(loadMatchOverrides(matchId));
    if (!matchId || typeof window === "undefined") return;

    const handleSync = (e: Event) => {
      const detail = (e as CustomEvent<{ matchId?: string }>).detail;
      if (!detail?.matchId || detail.matchId === matchId) {
        setOverrides(loadMatchOverrides(matchId));
      }
    };
    window.addEventListener(CHANGE_EVENT, handleSync);
    window.addEventListener("storage", handleSync);
    return () => {
      window.removeEventListener(CHANGE_EVENT, handleSync);
      window.removeEventListener("storage", handleSync);
    };
  }, [matchId]);

  const setPlayerOverride = useCallback(
    (trackId: number, patch: PlayerIdentityOverride | null) => {
      if (!matchId) return;
      const cur = loadMatchOverrides(matchId);
      const next = { ...cur };
      if (
        !patch ||
        ((patch.jersey_number == null || patch.jersey_number <= 0) &&
          !patch.custom_name?.trim())
      ) {
        delete next[trackId];
      } else {
        const cleanNum =
          patch.jersey_number != null &&
          patch.jersey_number >= 1 &&
          patch.jersey_number <= 99
            ? Math.round(patch.jersey_number)
            : null;
        next[trackId] = {
          jersey_number: cleanNum,
          custom_name: patch.custom_name?.trim() || undefined,
        };
      }
      saveMatchOverrides(matchId, next);
      setOverrides(next);
    },
    [matchId]
  );

  return { overrides, setPlayerOverride };
}

export interface ResolvedIdentity {
  /** Strictly `#jn` when verified/overridden, else compact display fallback `T{1..99}`. */
  badgeText: string;
  /** Compact canvas label (strictly 1..99 digits, never >99 raw tracker clutter). */
  canvasLabel: string;
  isVerified: boolean;
  isCoachOverride: boolean;
  jerseyNumber: number | null;
  customName: string | null;
}

// Per-match deterministic collision-free squad slot cache for unverified track IDs > 99
const _matchSquadMaps = new Map<string, Map<number, number>>();
let _activeSquadMatchId: string | null = null;

interface SquadSyncTelemetryLike {
  players?: Array<{
    track_id: number;
    team: number;
    jersey_number?: number | null;
    jersey_verified?: boolean | null;
  }>;
  frames?: Array<{
    players?: Array<{
      id: number;
      team: number;
      role?: string;
      jn?: number | null;
      jv?: boolean | null;
    }>;
  }>;
}

/**
 * Deterministically maps any unverified raw ByteTrack ID > 99 into a clean collision-free
 * football squad slot in [1..99] per match+team, avoiding verified jersey numbers and active
 * teammates so 2D Radar, Pass Network, Spotlight, and Leaderboard stay 100% consistent.
 */
export function syncMatchSquadNumbers(
  matchId: string | undefined,
  telemetry: SquadSyncTelemetryLike | null | undefined,
  overrides?: MatchOverridesMap
): Map<number, number> {
  const key = matchId || "__default__";
  const slotMap = new Map<number, number>();
  if (!telemetry) {
    _matchSquadMaps.set(key, slotMap);
    _activeSquadMatchId = key;
    return slotMap;
  }

  const usedByTeam: Record<number, Set<number>> = { 0: new Set(), 1: new Set() };
  const over99ByTeam: Record<number, Set<number>> = { 0: new Set(), 1: new Set() };

  const registerEntity = (
    tid: number,
    team: number,
    jv?: boolean | null,
    jn?: number | null
  ) => {
    const tm = team === 1 ? 1 : 0;
    const ov = overrides?.[tid];
    const ovNum =
      ov?.jersey_number != null && ov.jersey_number >= 1 && ov.jersey_number <= 99
        ? ov.jersey_number
        : null;
    if (ovNum !== null) {
      usedByTeam[tm].add(ovNum);
      return;
    }
    if (jv && jn != null && jn >= 1 && jn <= 99) {
      usedByTeam[tm].add(jn);
      return;
    }
    if (tid >= 1 && tid <= 99) {
      usedByTeam[tm].add(tid);
      slotMap.set(tid, tid);
    } else if (tid > 99) {
      over99ByTeam[tm].add(tid);
    }
  };

  for (const p of telemetry.players || []) {
    registerEntity(p.track_id, p.team, p.jersey_verified, p.jersey_number);
  }
  for (const fr of telemetry.frames || []) {
    for (const fp of fr.players || []) {
      if (fp.role === "referee") continue;
      registerEntity(fp.id, fp.team, fp.jv, fp.jn);
    }
  }

  for (const tm of [0, 1] as const) {
    const used = usedByTeam[tm];
    const sortedHigh = Array.from(over99ByTeam[tm]).sort((a, b) => a - b);
    for (const tid of sortedHigh) {
      // Prefer a deterministic unused slot in 1..99 starting from (tid % 99) + 1
      const pref = ((tid - 1) % 99) + 1;
      let chosen = pref;
      if (used.has(chosen)) {
        for (let offset = 1; offset < 99; offset++) {
          const cand = ((pref - 1 + offset) % 99) + 1;
          if (!used.has(cand)) {
            chosen = cand;
            break;
          }
        }
      }
      used.add(chosen);
      slotMap.set(tid, chosen);
    }
  }

  _matchSquadMaps.set(key, slotMap);
  _activeSquadMatchId = key;
  return slotMap;
}

export function resolveCompactTrackNumber(
  trackId: number,
  matchId?: string
): number {
  if (trackId >= 1 && trackId <= 99) return trackId;
  const key = matchId || _activeSquadMatchId || "__default__";
  const mapped = _matchSquadMaps.get(key)?.get(trackId);
  if (mapped != null && mapped >= 1 && mapped <= 99) return mapped;
  return ((Math.max(1, Math.abs(Math.round(trackId))) - 1) % 99) + 1;
}

export function resolvePlayerIdentity(
  trackId: number,
  source?: {
    jersey_verified?: boolean | null;
    jersey_number?: number | null;
    jv?: boolean | null;
    jn?: number | null;
  } | null,
  overrides?: MatchOverridesMap,
  matchId?: string
): ResolvedIdentity {
  const ov = overrides?.[trackId];
  const ovNum =
    ov?.jersey_number != null && ov.jersey_number >= 1 && ov.jersey_number <= 99
      ? ov.jersey_number
      : null;
  const customName = ov?.custom_name?.trim() || null;

  if (ovNum !== null) {
    return {
      badgeText: `#${ovNum}`,
      canvasLabel: `${ovNum}`,
      isVerified: true,
      isCoachOverride: true,
      jerseyNumber: ovNum,
      customName,
    };
  }

  const modelVerified = Boolean(source?.jersey_verified || source?.jv);
  const modelNum = source?.jersey_number ?? source?.jn ?? null;

  if (modelVerified && modelNum != null && modelNum >= 1 && modelNum <= 99) {
    return {
      badgeText: `#${modelNum}`,
      canvasLabel: `${modelNum}`,
      isVerified: true,
      isCoachOverride: false,
      jerseyNumber: modelNum,
      customName,
    };
  }

  const compactNum = resolveCompactTrackNumber(trackId, matchId);
  return {
    badgeText: `T${compactNum}`,
    canvasLabel: `${compactNum}`,
    isVerified: false,
    isCoachOverride: Boolean(customName),
    jerseyNumber: null,
    customName,
  };
}
