"use client";

import { useEffect, useState, useCallback } from "react";
import type { MatchCard, Telemetry } from "@/lib/types";
import { fetchMatch, fetchTelemetry } from "@/lib/api";

interface UseMatchDataReturn {
  match: MatchCard | null;
  telemetry: Telemetry | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export function useMatchData(matchId: string): UseMatchDataReturn {
  const [match, setMatch] = useState<MatchCard | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    if (!matchId) return;
    setIsLoading(true);
    setError(null);
    try {
      const [m, tel] = await Promise.all([
        fetchMatch(matchId),
        fetchTelemetry(matchId),
      ]);
      setMatch(m);
      setTelemetry(tel);
    } catch (err: unknown) {
      console.error(`Failed to load match ${matchId}:`, err);
      setError(err instanceof Error ? err.message : "Failed to load match data");
    } finally {
      setIsLoading(false);
    }
  }, [matchId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  return {
    match,
    telemetry,
    isLoading,
    error,
    refetch: loadData,
  };
}
