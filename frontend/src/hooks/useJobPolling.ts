"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchJobStatus } from "@/lib/api";
import type { JobStatus } from "@/lib/types";

export function useJobPolling(initialJobId: string | null = null) {
  const [jobId, setJobId] = useState<string | null>(initialJobId);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const poll = useCallback(
    async (id: string) => {
      try {
        const data = await fetchJobStatus(id);
        setJobStatus(data);
        setError(null);

        if (
          data.status === "completed" ||
          data.status === "failed" ||
          data.status === "cancelled"
        ) {
          setIsPolling(false);
          clearTimer();
        }
      } catch (err: any) {
        setError(err?.message || "Failed to poll job status");
      }
    },
    [clearTimer]
  );

  const startPolling = useCallback(
    (id: string) => {
      clearTimer();
      setJobId(id);
      setIsPolling(true);
      setError(null);
      setJobStatus(null);

      // Immediate first poll
      poll(id);

      // Start 1.5s interval
      timerRef.current = setInterval(() => {
        poll(id);
      }, 1500);
    },
    [clearTimer, poll]
  );

  const cancelPolling = useCallback(() => {
    clearTimer();
    setIsPolling(false);
  }, [clearTimer]);

  const resetJob = useCallback(() => {
    clearTimer();
    setIsPolling(false);
    setJobId(null);
    setJobStatus(null);
    setError(null);
  }, [clearTimer]);

  useEffect(() => {
    if (initialJobId) {
      startPolling(initialJobId);
    }
    return () => clearTimer();
  }, [initialJobId, startPolling, clearTimer]);

  return {
    jobId,
    jobStatus,
    setJobStatus,
    isPolling,
    error,
    startPolling,
    cancelPolling,
    resetJob,
  };
}
