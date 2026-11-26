"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Cpu,
  Upload,
  Play,
  Loader2,
  AlertCircle,
  Target,
  SlidersHorizontal,
  FileCode2,
  CheckCircle2,
  Clock,
  Sparkles,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import {
  analyzeActionSpotting,
  fetchActionSpottingMatches,
} from "@/lib/api";
import type {
  ActionCategory,
  ActionSpottingAnalyzeResponse,
  ActionSpottingMatchesResponse,
} from "@/lib/types";
import {
  ACTION_CATEGORY_ORDER,
  buildFilteredDensityBins,
  getActionCategoryLabel,
  getActionClassLabel,
  getCategoryMeta,
} from "@/lib/actionSpottingHelpers";
import {
  D3ActionTimeline,
  eventUniqueKey,
} from "@/components/action-spotting/D3ActionTimeline";
import { D3ActionDensityChart } from "@/components/action-spotting/D3ActionDensityChart";
import { BidiNum } from "@/components/ui/BidiNum";
import { Skeleton } from "@/components/ui/Skeleton";

export default function ActionSpottingPage() {
  const { t } = useLang();

  const [matchesMeta, setMatchesMeta] =
    useState<ActionSpottingMatchesResponse | null>(null);
  const [sourceMode, setSourceMode] = useState<"demo" | "upload">("demo");
  const [selectedMatchId, setSelectedMatchId] = useState<string>("test_000");
  const [halfMode, setHalfMode] = useState<"half1" | "half2" | "both">("half1");
  const [featureFile, setFeatureFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<ActionSpottingAnalyzeResponse | null>(null);

  // Live interactive post-inference filters
  const [confThreshold, setConfThreshold] = useState<number>(0.35);
  const [categoryFilter, setCategoryFilter] = useState<"all" | ActionCategory>("all");
  const [classFilter, setClassFilter] = useState<string>("all");
  const [halfFilter, setHalfFilter] = useState<"all" | "1" | "2">("all");
  const [showGtOverlay, setShowGtOverlay] = useState<boolean>(true);
  const [selectedEventKey, setSelectedEventKey] = useState<string | null>(null);

  const hasAutoRunRef = useRef<boolean>(false);

  const executeInference = useCallback(
    async (opts?: {
      mode?: "demo" | "upload";
      matchId?: string;
      half?: "half1" | "half2" | "both";
      file?: File | null;
      minConf?: number;
    }) => {
      const m = opts?.mode ?? sourceMode;
      const mid = opts?.matchId ?? selectedMatchId;
      const h = opts?.half ?? halfMode;
      const f = opts?.file !== undefined ? opts.file : featureFile;

      if (m === "upload" && !f) {
        setErrorMsg(t("as.upload_prompt"));
        return;
      }

      setIsRunning(true);
      setErrorMsg(null);
      setSelectedEventKey(null);
      setClassFilter("all");
      setCategoryFilter("all");
      setHalfFilter("all");

      try {
        // Request base floor 0.15 so live slider [0.15 -> 0.85] reshapes smoothly client-side
        const baseConf = Math.min(0.2, opts?.minConf ?? 0.2);
        const res = await analyzeActionSpotting({
          match_id: m === "demo" ? mid : undefined,
          half: h,
          min_conf: baseConf,
          nms_window_sec: 10.0,
          file: m === "upload" ? f : undefined,
        });
        setResult(res);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setErrorMsg(msg);
      } finally {
        setIsRunning(false);
      }
    },
    [featureFile, halfMode, selectedMatchId, sourceMode, t],
  );

  useEffect(() => {
    let active = true;
    fetchActionSpottingMatches()
      .then((data) => {
        if (!active) return;
        setMatchesMeta(data);
        const firstReady = data.demo_matches.find((m) => m.features_ready);
        if (firstReady) {
          setSelectedMatchId(firstReady.match_id);
          if (!hasAutoRunRef.current) {
            hasAutoRunRef.current = true;
            executeInference({
              mode: "demo",
              matchId: firstReady.match_id,
              half: "half1",
              minConf: 0.2,
            });
          }
        }
      })
      .catch(() => {
        // Non-fatal: user can still retry manually
      });
    return () => {
      active = false;
    };
  }, [executeInference]);

  // Filtered events reacting instantaneously to confidence slider + category/class/half chips
  const filteredEvents = useMemo(() => {
    if (!result) return [];
    return result.events.filter((ev) => {
      if (ev.confidence < confThreshold) return false;
      if (halfFilter !== "all" && String(ev.half) !== halfFilter) return false;
      if (categoryFilter !== "all" && ev.category !== categoryFilter) return false;
      if (classFilter !== "all" && ev.label !== classFilter) return false;
      return true;
    });
  }, [categoryFilter, classFilter, confThreshold, halfFilter, result]);

  const filteredBins = useMemo(() => {
    if (!result) return [];
    return buildFilteredDensityBins(result.density_bins, filteredEvents, halfFilter);
  }, [filteredEvents, halfFilter, result]);

  const rankedClassCounts = useMemo(() => {
    const map = new Map<string, { label: string; category: string; count: number }>();
    for (const ev of filteredEvents) {
      const prev = map.get(ev.label);
      if (prev) prev.count += 1;
      else map.set(ev.label, { label: ev.label, category: ev.category, count: 1 });
    }
    return Array.from(map.values()).sort((a, b) => b.count - a.count);
  }, [filteredEvents]);

  // Live Precision/Recall recalculated at active slider threshold when GT exists
  const liveGtStats = useMemo(() => {
    if (!result?.ground_truth_events || result.ground_truth_events.length === 0) {
      return null;
    }
    const gtFiltered = result.ground_truth_events.filter((g) =>
      halfFilter === "all" ? true : String(g.half ?? 1) === halfFilter,
    );
    const tol = result.nms_window_sec || 10.0;
    const candidates: Array<{ dt: number; pIdx: number; gIdx: number }> = [];
    filteredEvents.forEach((pred, pIdx) => {
      gtFiltered.forEach((gt, gIdx) => {
        if ((gt.half ?? 1) !== pred.half) return;
        if (gt.label !== pred.label) return;
        const dt = Math.abs(pred.time_s - gt.time_s);
        if (dt <= tol) candidates.push({ dt, pIdx, gIdx });
      });
    });
    candidates.sort((a, b) => a.dt - b.dt);
    const usedP = new Set<number>();
    const usedG = new Set<number>();
    for (const c of candidates) {
      if (!usedP.has(c.pIdx) && !usedG.has(c.gIdx)) {
        usedP.add(c.pIdx);
        usedG.add(c.gIdx);
      }
    }
    const matched = usedG.size;
    const gtTotal = gtFiltered.length;
    const predTotal = filteredEvents.length;
    return {
      tolerance_sec: tol,
      matched,
      gtTotal,
      precision_pct: predTotal > 0 ? Math.round((matched / predTotal) * 1000) / 10 : 0,
      recall_pct: gtTotal > 0 ? Math.round((matched / gtTotal) * 1000) / 10 : 0,
    };
  }, [filteredEvents, halfFilter, result]);

  const totalAnalyzedSec = useMemo(() => {
    if (!result) return 0;
    return result.halves.reduce((acc, h) => acc + h.duration_s, 0);
  }, [result]);

  return (
    <div className="max-w-[1650px] mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6">
      {/* ── Hero Control Deck ── */}
      <section className="rounded-3xl obsidian-card p-5 sm:p-7 space-y-6 relative overflow-hidden">
        <div className="absolute -top-24 -end-24 w-72 h-72 rounded-full bg-tv-mint/[0.06] blur-3xl pointer-events-none" />

        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
          <div className="space-y-2 max-w-2xl">
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-bold bg-tv-mint/10 border border-tv-mint/30 text-tv-mint">
              <Sparkles className="w-3.5 h-3.5 shrink-0" />
              <span dir="ltr">{t("as.badge")}</span>
            </span>
            <h1 className="text-xl sm:text-3xl font-extrabold text-white tracking-tight">
              {t("as.title")}
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
              {t("as.subtitle")}
            </p>
          </div>

          {/* Source Switcher Pills */}
          <div className="flex items-center p-1 rounded-xl obsidian-well border border-white/[0.08] self-start lg:self-auto">
            <button
              type="button"
              onClick={() => {
                setSourceMode("demo");
                setErrorMsg(null);
              }}
              className={`px-3.5 py-2 rounded-lg text-xs font-bold transition-colors cursor-pointer flex items-center gap-1.5 ${
                sourceMode === "demo"
                  ? "bg-tv-cyan text-slate-950 shadow"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              <Activity className="w-3.5 h-3.5 shrink-0" />
              <span>{t("as.source_demo")}</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setSourceMode("upload");
                setErrorMsg(null);
              }}
              className={`px-3.5 py-2 rounded-lg text-xs font-bold transition-colors cursor-pointer flex items-center gap-1.5 ${
                sourceMode === "upload"
                  ? "bg-tv-cyan text-slate-950 shadow"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              <Upload className="w-3.5 h-3.5 shrink-0" />
              <span>{t("as.source_upload")}</span>
            </button>
          </div>
        </div>

        {/* Source Configurator & Live Inference Trigger */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center pt-4 border-t border-white/[0.08]">
          {sourceMode === "demo" ? (
            <div className="lg:col-span-5 space-y-1.5">
              <label className="text-[11px] font-semibold text-slate-400 block">
                {t("as.source_demo")}
              </label>
              <select
                value={selectedMatchId}
                onChange={(e) => setSelectedMatchId(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-xl obsidian-well border border-white/10 text-xs sm:text-sm font-semibold text-white focus:outline-none focus:border-tv-cyan"
              >
                {(matchesMeta?.demo_matches || []).map((m) => (
                  <option key={m.match_id} value={m.match_id} className="bg-slate-950">
                    {m.title} ({m.match_id})
                  </option>
                ))}
                {(!matchesMeta || matchesMeta.demo_matches.length === 0) && (
                  <option value="test_000" className="bg-slate-950">
                    Manchester United 1 - 1 Arsenal (test_000)
                  </option>
                )}
              </select>
            </div>
          ) : (
            <div className="lg:col-span-5">
              <input
                ref={fileInputRef}
                type="file"
                accept=".npy,.npz"
                className="hidden"
                onChange={(e) => {
                  const chosen = e.target.files?.[0] ?? null;
                  setFeatureFile(chosen);
                  setErrorMsg(null);
                }}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="w-full p-3 rounded-xl border border-dashed border-white/20 obsidian-well hover:border-tv-cyan/60 transition-colors text-start cursor-pointer flex items-center gap-3"
              >
                <div className="w-9 h-9 rounded-lg bg-tv-cyan/15 text-tv-cyan flex items-center justify-center shrink-0">
                  <FileCode2 className="w-4 h-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-bold text-white truncate">
                    {featureFile ? featureFile.name : t("as.upload_prompt")}
                  </p>
                  <p className="text-[11px] text-slate-400 truncate mt-0.5">
                    {t("as.upload_hint")}
                  </p>
                </div>
              </button>
            </div>
          )}

          {/* Half Selector */}
          <div className="lg:col-span-4 space-y-1.5">
            <label className="text-[11px] font-semibold text-slate-400 block">
              {t("as.half_label")}
            </label>
            <div className="grid grid-cols-3 gap-1.5 p-1 rounded-xl obsidian-well border border-white/[0.06]">
              {(
                [
                  { id: "half1", lbl: t("as.half_1") },
                  { id: "half2", lbl: t("as.half_2") },
                  { id: "both", lbl: t("as.half_both") },
                ] as const
              ).map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setHalfMode(item.id)}
                  className={`py-2 px-2 rounded-lg text-[11px] font-bold transition-colors cursor-pointer truncate ${
                    halfMode === item.id
                      ? "bg-white/[0.14] text-white border border-white/15"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {item.lbl}
                </button>
              ))}
            </div>
          </div>

          {/* Primary ONNX Execution Button */}
          <div className="lg:col-span-3 flex items-end pt-4 lg:pt-5">
            <button
              type="button"
              disabled={isRunning}
              onClick={() => executeInference()}
              className="w-full py-3 px-4 rounded-xl bg-tv-mint text-slate-950 hover:bg-tv-mint/90 disabled:opacity-60 font-extrabold text-xs sm:text-sm shadow-lg shadow-tv-mint/20 transition-all cursor-pointer flex items-center justify-center gap-2"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                  <span>{t("as.running_btn")}</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current shrink-0" />
                  <span>{t("as.run_btn")}</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Graceful Validation / API Error Banner */}
        {errorMsg && (
          <div className="p-3.5 rounded-xl bg-tv-coral/12 border border-tv-coral/35 text-tv-coral text-xs font-medium flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
      </section>

      {/* ── Execution Loading State ── */}
      {isRunning && !result && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Skeleton className="h-24 rounded-2xl" />
          <Skeleton className="h-24 rounded-2xl" />
          <Skeleton className="h-24 rounded-2xl" />
          <Skeleton className="h-24 rounded-2xl" />
        </div>
      )}

      {/* ── Live Inference Results Cockpit ── */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
          className="space-y-6"
        >
          {/* 1. Real Runtime KPI Strip */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 sm:gap-4">
            <div className="rounded-2xl obsidian-card p-4 space-y-1">
              <span className="text-[11px] font-semibold text-slate-400 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-tv-mint" />
                <span>{t("as.kpi_backend")}</span>
              </span>
              <div className="pt-1">
                <BidiNum className="px-2.5 py-1 rounded-lg bg-tv-mint/15 border border-tv-mint/35 text-tv-mint text-xs sm:text-sm font-bold inline-block">
                  {result.backend_used}
                </BidiNum>
              </div>
            </div>

            <div className="rounded-2xl obsidian-card p-4 space-y-1">
              <span className="text-[11px] font-semibold text-slate-400 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-tv-cyan" />
                <span>{t("as.kpi_latency")}</span>
              </span>
              <p className="text-xl sm:text-2xl font-extrabold text-white">
                <BidiNum>{`${(result.total_inference_ms / 1000).toFixed(2)}s`}</BidiNum>
              </p>
              <p className="text-[11px] text-slate-400">
                <BidiNum>{`${result.total_inference_ms.toFixed(0)} ms`}</BidiNum>
              </p>
            </div>

            <div className="rounded-2xl obsidian-card p-4 space-y-1">
              <span className="text-[11px] font-semibold text-slate-400">
                {t("as.kpi_duration")}
              </span>
              <p className="text-xl sm:text-2xl font-extrabold text-white inline-flex items-baseline gap-1">
                <BidiNum>{Math.round(totalAnalyzedSec / 60)}</BidiNum>
                <span className="text-xs font-semibold text-slate-400">
                  {t("as.minutes_unit")}
                </span>
              </p>
              <p className="text-[11px] text-slate-400">
                <BidiNum>{`${result.halves.reduce((s, h) => s + h.num_frames, 0)}f @ ${result.fps} FPS`}</BidiNum>
              </p>
            </div>

            <div className="rounded-2xl obsidian-card p-4 space-y-1">
              <span className="text-[11px] font-semibold text-slate-400">
                {t("as.kpi_events")}
              </span>
              <p className="text-xl sm:text-2xl font-extrabold text-tv-cyan">
                <BidiNum>{filteredEvents.length}</BidiNum>
              </p>
              <p className="text-[11px] text-slate-400">
                {t("as.conf_slider")}{" "}
                <BidiNum className="text-white font-semibold">
                  {`≥ ${Math.round(confThreshold * 100)}%`}
                </BidiNum>
              </p>
            </div>

            <div className="col-span-2 lg:col-span-1 rounded-2xl obsidian-card p-4 space-y-1">
              <span className="text-[11px] font-semibold text-slate-400 flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5 text-tv-amber" />
                <span>{t("as.kpi_gt_align")}</span>
              </span>
              {liveGtStats ? (
                <>
                  <div className="flex items-baseline gap-3 pt-0.5">
                    <div>
                      <span className="text-[10px] text-slate-400 block">
                        {t("as.kpi_precision")}
                      </span>
                      <BidiNum className="text-lg font-extrabold text-tv-mint">
                        {`${liveGtStats.precision_pct}%`}
                      </BidiNum>
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 block">
                        {t("as.kpi_recall")}
                      </span>
                      <BidiNum className="text-lg font-extrabold text-tv-amber">
                        {`${liveGtStats.recall_pct}%`}
                      </BidiNum>
                    </div>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    {t("as.kpi_matched")}{" "}
                    <BidiNum className="text-white font-semibold">
                      {`${liveGtStats.matched}/${liveGtStats.gtTotal}`}
                    </BidiNum>
                  </p>
                </>
              ) : (
                <p className="text-xs text-slate-500 pt-2">
                  <BidiNum>{`±${result.nms_window_sec}s NMS`}</BidiNum>
                </p>
              )}
            </div>
          </div>

          {/* 2. Real-Time Confidence Slider + Category Filter Ribbon */}
          <div className="rounded-2xl obsidian-card p-4 sm:p-5 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex-1 space-y-2 max-w-md">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-white flex items-center gap-2">
                    <SlidersHorizontal className="w-3.5 h-3.5 text-tv-cyan" />
                    <span>{t("as.conf_slider")}</span>
                  </span>
                  <BidiNum className="px-2 py-0.5 rounded-md bg-tv-cyan/15 text-tv-cyan font-bold">
                    {confThreshold.toFixed(2)}
                  </BidiNum>
                </div>
                <input
                  type="range"
                  min={0.15}
                  max={0.85}
                  step={0.05}
                  value={confThreshold}
                  onChange={(e) => setConfThreshold(Number(e.target.value))}
                  className="w-full accent-tv-cyan cursor-pointer"
                />
              </div>

              {result.halves.length > 1 && (
                <div className="flex items-center gap-1.5 p-1 rounded-xl obsidian-well self-start">
                  {(
                    [
                      { k: "all", lbl: t("as.half_both") },
                      { k: "1", lbl: t("as.half_1") },
                      { k: "2", lbl: t("as.half_2") },
                    ] as const
                  ).map((hf) => (
                    <button
                      key={hf.k}
                      type="button"
                      onClick={() => setHalfFilter(hf.k)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-colors cursor-pointer ${
                        halfFilter === hf.k
                          ? "bg-tv-cyan text-slate-950"
                          : "text-slate-400 hover:text-white"
                      }`}
                    >
                      {hf.lbl}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Category Filter Chips */}
            <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-white/[0.06]">
              <button
                type="button"
                onClick={() => {
                  setCategoryFilter("all");
                  setClassFilter("all");
                }}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold border transition-colors cursor-pointer ${
                  categoryFilter === "all" && classFilter === "all"
                    ? "bg-white/15 border-white/30 text-white"
                    : "obsidian-well text-slate-400 hover:text-white"
                }`}
              >
                {t("as.cat.all")}
              </button>

              {ACTION_CATEGORY_ORDER.map((cat) => {
                const meta = getCategoryMeta(cat);
                const isActive = categoryFilter === cat;
                return (
                  <button
                    key={`cat-${cat}`}
                    type="button"
                    onClick={() => {
                      setCategoryFilter(isActive ? "all" : cat);
                      setClassFilter("all");
                    }}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all cursor-pointer flex items-center gap-1.5 ${
                      isActive
                        ? `${meta.bgClass} ${meta.borderClass} ${meta.textClass} font-bold`
                        : "obsidian-well text-slate-300 hover:text-white"
                    }`}
                  >
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: meta.color }}
                    />
                    <span>{getActionCategoryLabel(cat, t)}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 3. Signature D3 Timeline */}
          <D3ActionTimeline
            events={filteredEvents}
            halves={result.halves}
            groundTruthEvents={result.ground_truth_events}
            nmsWindowSec={result.nms_window_sec}
            selectedEventKey={selectedEventKey}
            onSelectEvent={setSelectedEventKey}
            showGtOverlay={showGtOverlay}
            onToggleGtOverlay={() => setShowGtOverlay((p) => !p)}
          />

          {/* 4. 5-Minute Intensity Chart + Class Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            <div className="lg:col-span-7">
              <D3ActionDensityChart bins={filteredBins} />
            </div>

            <div className="lg:col-span-5 rounded-2xl obsidian-card p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm sm:text-base font-bold text-white">
                  {t("as.class_breakdown")}
                </h3>
                <BidiNum className="px-2 py-0.5 rounded-md obsidian-well text-xs font-bold text-tv-cyan">
                  {rankedClassCounts.length}
                </BidiNum>
              </div>

              {rankedClassCounts.length === 0 ? (
                <p className="text-xs text-slate-500 py-8 text-center">
                  {t("as.empty_filtered")}
                </p>
              ) : (
                <div className="space-y-2.5 max-h-72 overflow-y-auto pe-1">
                  {rankedClassCounts.map((item) => {
                    const maxC = rankedClassCounts[0]?.count || 1;
                    const pct = Math.round((item.count / maxC) * 100);
                    const meta = getCategoryMeta(item.category);
                    const isFocused = classFilter === item.label;

                    return (
                      <button
                        key={item.label}
                        type="button"
                        onClick={() =>
                          setClassFilter(isFocused ? "all" : item.label)
                        }
                        className={`w-full text-start p-2 rounded-xl transition-colors cursor-pointer border ${
                          isFocused
                            ? "bg-white/[0.08] border-tv-cyan/40"
                            : "border-transparent hover:bg-white/[0.04]"
                        }`}
                      >
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span className="font-semibold text-slate-200 flex items-center gap-2">
                            <span
                              className="w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: meta.color }}
                            />
                            <span>{getActionClassLabel(item.label, t)}</span>
                          </span>
                          <BidiNum className="font-bold text-white">
                            {item.count}
                          </BidiNum>
                        </div>
                        <div className="h-1.5 w-full rounded-full bg-white/[0.06] overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${pct}%`,
                              backgroundColor: meta.color,
                            }}
                          />
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* 5. Chronological Event Ledger */}
          <div className="rounded-2xl obsidian-card p-5 space-y-4">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-tv-mint shrink-0" />
                <span>{t("as.ledger_title")}</span>
              </h3>
              <span className="text-xs text-slate-400">
                {t("as.visible_count")}:{" "}
                <BidiNum className="text-white font-bold">
                  {filteredEvents.length}
                </BidiNum>
              </span>
            </div>

            {filteredEvents.length === 0 ? (
              <p className="text-xs text-slate-500 py-8 text-center">
                {t("as.empty_filtered")}
              </p>
            ) : (
              <div className="overflow-x-auto max-h-96 rounded-xl border border-white/[0.08]">
                <table className="w-full text-start text-xs border-collapse">
                  <thead className="sticky top-0 bg-[#0A101D] border-b border-white/[0.08] text-slate-300 text-[11px] font-semibold">
                    <tr>
                      <th className="py-2.5 px-3 text-start">{t("as.col_time")}</th>
                      <th className="py-2.5 px-3 text-start">{t("as.col_half")}</th>
                      <th className="py-2.5 px-3 text-start">{t("as.col_event")}</th>
                      <th className="py-2.5 px-3 text-start">{t("as.col_category")}</th>
                      <th className="py-2.5 px-3 text-end">{t("as.col_frame")}</th>
                      <th className="py-2.5 px-3 text-end">{t("as.col_conf")}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.05]">
                    {filteredEvents.map((ev, idx) => {
                      const key = eventUniqueKey(ev, idx);
                      const meta = getCategoryMeta(ev.category);
                      const isSel = selectedEventKey === key;

                      return (
                        <tr
                          key={key}
                          onClick={() => setSelectedEventKey(isSel ? null : key)}
                          className={`cursor-pointer transition-colors ${
                            isSel
                              ? "bg-tv-cyan/15"
                              : "hover:bg-white/[0.03]"
                          }`}
                        >
                          <td className="py-2.5 px-3 font-bold text-white">
                            <BidiNum>{ev.minute_str}</BidiNum>
                          </td>
                          <td className="py-2.5 px-3 text-slate-300">
                            <BidiNum>{`H${ev.half}`}</BidiNum>
                          </td>
                          <td className="py-2.5 px-3 font-bold text-white">
                            <span className="inline-flex items-center gap-2">
                              <span
                                className="w-2 h-2 rounded-full shrink-0"
                                style={{ backgroundColor: meta.color }}
                              />
                              <span>{getActionClassLabel(ev.label, t)}</span>
                            </span>
                          </td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`inline-block px-2 py-0.5 rounded-md text-[10px] font-bold border ${meta.bgClass} ${meta.borderClass} ${meta.textClass}`}
                            >
                              {getActionCategoryLabel(ev.category, t)}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-end text-slate-400">
                            <BidiNum>{ev.frame_idx}</BidiNum>
                          </td>
                          <td className="py-2.5 px-3 text-end">
                            <BidiNum className="font-bold text-tv-mint">
                              {`${(ev.confidence * 100).toFixed(1)}%`}
                            </BidiNum>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
}
