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
  /** Strictly `#jn` when verified/overridden, else honest ByteTrack fallback `T{id}`. */
  badgeText: string;
  /** Compact canvas label without `#` clutter when drawn inside 20px circles. */
  canvasLabel: string;
  isVerified: boolean;
  isCoachOverride: boolean;
  jerseyNumber: number | null;
  customName: string | null;
}

export function resolvePlayerIdentity(
  trackId: number,
  source?: {
    jersey_verified?: boolean | null;
    jersey_number?: number | null;
    jv?: boolean | null;
    jn?: number | null;
  } | null,
  overrides?: MatchOverridesMap
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

  if (modelVerified && modelNum != null) {
    return {
      badgeText: `#${modelNum}`,
      canvasLabel: `${modelNum}`,
      isVerified: true,
      isCoachOverride: false,
      jerseyNumber: modelNum,
      customName,
    };
  }

  return {
    badgeText: `T${trackId}`,
    canvasLabel: `${trackId}`,
    isVerified: false,
    isCoachOverride: Boolean(customName),
    jerseyNumber: null,
    customName,
  };
}
