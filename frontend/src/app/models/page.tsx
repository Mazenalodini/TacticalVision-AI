"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import * as d3 from "d3";
import { motion } from "framer-motion";
import {
  Cpu,
  CheckCircle2,
  AlertTriangle,
  HardDrive,
  RefreshCw,
  Compass,
  Activity,
  Eye,
  Sparkles,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import { fetchModelsSuite } from "@/lib/api";
import type { ModelsSuiteResponse } from "@/lib/types";
import { D3DetectionComparisonChart } from "@/components/models/D3DetectionComparisonChart";
import { D3ActionClassesChart } from "@/components/models/D3ActionClassesChart";
import { D3JerseyAndBenchmarksChart } from "@/components/models/D3JerseyAndBenchmarksChart";
import { BidiNum } from "@/components/ui/BidiNum";
import { Skeleton } from "@/components/ui/Skeleton";

export default function ModelsLabPage() {
  const { t } = useLang();
  const [suite, setSuite] = useState<ModelsSuiteResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const loadSuite = useCallback(async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const data = await fetchModelsSuite();
      setSuite(data);
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : t("models.load_error"));
    } finally {
      setIsLoading(false);
    }
  }, [t]);

  useEffect(() => {
    loadSuite();
  }, [loadSuite]);

  const m1 = suite?.metrics.model_1_players_detector;
  const m1Base = suite?.metrics.model_1_coco_baseline;
  const m2 = suite?.metrics.model_2_pitch_keypoints;
  const m3 = suite?.metrics.model_3_action_spotting;
  const m4 = suite?.metrics.model_4_jersey_recognition;

  // Model 2 D3 PCK tolerance curve points
  const pckLinePath = useMemo(() => {
    if (!m2) return "";
    const pts: Array<[number, number]> = [
      [5, m2.pck_5px_pct],
      [10, m2.pck_10px_pct],
      [20, m2.pck_20px_pct],
    ];
    const x = d3.scaleLinear().domain([4, 21]).range([36, 300]);
    const y = d3.scaleLinear().domain([0, 100]).range([112, 16]);
    return (
      d3
        .line<[number, number]>()
        .x((d) => x(d[0]))
        .y((d) => y(d[1]))
        .curve(d3.curveMonotoneX)(pts) ?? ""
    );
  }, [m2]);

  if (isLoading) {
    return (
      <div className="max-w-[1650px] mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        <Skeleton className="h-36 rounded-3xl" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Skeleton className="h-28 rounded-2xl" />
          <Skeleton className="h-28 rounded-2xl" />
          <Skeleton className="h-28 rounded-2xl" />
          <Skeleton className="h-28 rounded-2xl" />
        </div>
        <Skeleton className="h-80 rounded-2xl" />
      </div>
    );
  }

  if (errorMsg || !suite) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-tv-coral/15 border border-tv-coral/35 text-tv-coral flex items-center justify-center mx-auto">
          <AlertTriangle className="w-6 h-6" />
        </div>
        <h2 className="text-lg font-bold text-white">{t("models.load_error")}</h2>
        {errorMsg && <p className="text-xs text-slate-400">{errorMsg}</p>}
        <button
          type="button"
          onClick={loadSuite}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-tv-cyan text-slate-950 font-bold text-xs cursor-pointer"
        >
          <RefreshCw className="w-4 h-4" />
          <span>{t("models.retry")}</span>
        </button>
      </div>
    );
  }

  const runtimeChecks = [
    {
      label: t("models.runtime_ort"),
      ok: suite.runtime.onnxruntime_available,
      detail: suite.runtime.model_3_preferred_backend,
    },
    {
      label: t("models.runtime_torch"),
      ok: suite.runtime.torch_available,
      detail: "CPU",
    },
    {
      label: t("models.runtime_timm"),
      ok: suite.runtime.timm_available,
      detail: "MobileNetV3",
    },
    {
      label: t("models.runtime_parseq"),
      ok: suite.runtime.strhub_ocr_available,
      detail: suite.runtime.strhub_ocr_available ? t("models.active") : t("models.standby"),
    },
    {
      label: t("models.runtime_demo"),
      ok: suite.runtime.demo_features_available,
      detail: "test_000.npz",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="max-w-[1650px] mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6"
    >
      {/* ── Header Banner ── */}
      <section className="rounded-3xl obsidian-card p-5 sm:p-7 space-y-3 relative overflow-hidden">
        <div className="absolute -top-24 -end-24 w-72 h-72 rounded-full bg-tv-amber/[0.06] blur-3xl pointer-events-none" />
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-bold bg-tv-amber/12 border border-tv-amber/30 text-tv-amber">
          <Sparkles className="w-3.5 h-3.5 shrink-0" />
          <span dir="ltr">{t("models.badge")}</span>
        </span>
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
          <div className="space-y-1.5 max-w-3xl">
            <h1 className="text-xl sm:text-3xl font-extrabold text-white tracking-tight">
              {t("models.title")}
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
              {t("models.subtitle")}
            </p>
          </div>
          <button
            type="button"
            onClick={loadSuite}
            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl obsidian-well hover:bg-white/[0.08] border border-white/10 text-xs font-bold text-slate-200 cursor-pointer self-start lg:self-auto"
          >
            <RefreshCw className="w-3.5 h-3.5 text-tv-cyan" />
            <span>{t("models.retry")}</span>
          </button>
        </div>
      </section>

      {/* ── Section 1: Live Runtime & Disk Weight Inventory ── */}
      <section className="rounded-2xl obsidian-card p-5 space-y-4">
        <div>
          <h2 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-tv-cyan shrink-0" />
            <span>{t("models.readiness_title")}</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">{t("models.readiness_desc")}</p>
        </div>

        {/* Runtime Stack Pills */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5">
          {runtimeChecks.map((rc) => (
            <div
              key={rc.label}
              className={`p-3 rounded-xl border flex items-center justify-between gap-2 ${
                rc.ok
                  ? "bg-tv-mint/[0.06] border-tv-mint/30"
                  : "obsidian-well border-white/[0.08]"
              }`}
            >
              <div className="min-w-0">
                <p className="text-xs font-bold text-white truncate">{rc.label}</p>
                <p className="text-[11px] text-slate-400 truncate font-mono">{rc.detail}</p>
              </div>
              {rc.ok ? (
                <CheckCircle2 className="w-4 h-4 text-tv-mint shrink-0" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-tv-amber shrink-0" />
              )}
            </div>
          ))}
        </div>

        {/* Disk Weights Table */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
          {Object.entries(suite.disk_inventory).map(([modelKey, files]) => (
            <div key={modelKey} className="p-3.5 rounded-xl obsidian-well space-y-2">
              <p className="text-xs font-extrabold text-tv-cyan font-mono truncate">
                {modelKey}
              </p>
              <div className="space-y-1.5">
                {Object.entries(files).map(([variant, info]) => (
                  <div
                    key={variant}
                    className="flex items-center justify-between gap-2 text-[11px]"
                  >
                    <span className="text-slate-300 font-mono truncate">
                      {info.filename}
                    </span>
                    {info.exists ? (
                      <BidiNum className="px-1.5 py-0.5 rounded bg-tv-mint/15 text-tv-mint font-bold shrink-0">
                        {`${info.size_mb ?? 0} MB`}
                      </BidiNum>
                    ) : (
                      <span className="px-1.5 py-0.5 rounded bg-tv-coral/15 text-tv-coral font-bold shrink-0">
                        {t("models.missing")}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Section 2: Model 1 & Model 2 Spatial Stack ── */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* Model 1 Card */}
        {m1 && m1Base && (
          <section className="xl:col-span-7 rounded-2xl obsidian-card p-5 space-y-4">
            <div className="space-y-1">
              <h2 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
                <Eye className="w-4 h-4 text-tv-cyan shrink-0" />
                <span>{t("models.m1_title")}</span>
              </h2>
              <p className="text-xs text-slate-400">{t("models.m1_desc")}</p>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="p-3 rounded-xl obsidian-well">
                <span className="text-[11px] text-slate-400 block">
                  {t("models.m1_overall_map")}
                </span>
                <BidiNum className="text-lg sm:text-xl font-extrabold text-white">
                  {`${(m1.overall_mAP50 * 100).toFixed(2)}%`}
                </BidiNum>
              </div>
              <div className="p-3 rounded-xl bg-tv-cyan/[0.08] border border-tv-cyan/30">
                <span className="text-[11px] text-tv-cyan block font-semibold">
                  {t("models.m1_ball_map")}
                </span>
                <BidiNum className="text-lg sm:text-xl font-extrabold text-tv-cyan">
                  {`${((m1.per_class_map.ball?.mAP50 ?? 0) * 100).toFixed(2)}%`}
                </BidiNum>
                <span className="text-[10px] text-slate-400 block">
                  {t("models.m1_ball_vs")}
                </span>
              </div>
              <div className="p-3 rounded-xl obsidian-well">
                <span className="text-[11px] text-slate-400 block">
                  {t("models.m1_player_f1")}
                </span>
                <BidiNum className="text-lg sm:text-xl font-extrabold text-tv-mint">
                  {`${((m1.per_class_calibrated_iou50.player?.f1 ?? 0) * 100).toFixed(2)}%`}
                </BidiNum>
              </div>
            </div>

            <D3DetectionComparisonChart tuned={m1} baseline={m1Base} />
          </section>
        )}

        {/* Model 2 Card */}
        {m2 && (
          <section className="xl:col-span-5 rounded-2xl obsidian-card p-5 space-y-4 flex flex-col justify-between">
            <div className="space-y-4">
              <div className="space-y-1">
                <h2 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
                  <Compass className="w-4 h-4 text-tv-mint shrink-0" />
                  <span>{t("models.m2_title")}</span>
                </h2>
                <p className="text-xs text-slate-400">{t("models.m2_desc")}</p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl obsidian-well">
                  <span className="text-[11px] text-slate-400 block">
                    {t("models.m2_map50")}
                  </span>
                  <BidiNum className="text-xl font-extrabold text-white">
                    {`${(m2.pose_mAP50 * 100).toFixed(2)}%`}
                  </BidiNum>
                </div>
                <div className="p-3 rounded-xl obsidian-well">
                  <span className="text-[11px] text-slate-400 block">
                    {t("models.m2_ready")}
                  </span>
                  <BidiNum className="text-xl font-extrabold text-tv-mint">
                    {`${m2.homography_ready_frames_pct.toFixed(2)}%`}
                  </BidiNum>
                </div>
                <div className="p-3 rounded-xl obsidian-well">
                  <span className="text-[11px] text-slate-400 block">
                    {t("models.m2_pck20")}
                  </span>
                  <BidiNum className="text-xl font-extrabold text-tv-cyan">
                    {`${m2.pck_20px_pct.toFixed(2)}%`}
                  </BidiNum>
                </div>
                <div className="p-3 rounded-xl obsidian-well">
                  <span className="text-[11px] text-slate-400 block">
                    {t("models.m2_err")}
                  </span>
                  <BidiNum className="text-xl font-extrabold text-tv-amber">
                    {`${m2.median_pixel_error_px.toFixed(2)} px`}
                  </BidiNum>
                </div>
              </div>
            </div>

            {/* Declarative D3 PCK Precision Curve */}
            <div className="p-4 rounded-xl obsidian-well space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-slate-200">PCK Spatial Accuracy Curve</span>
                <BidiNum className="text-tv-mint font-bold">
                  {`5px → 20px`}
                </BidiNum>
              </div>
              <div dir="ltr">
                <svg viewBox="0 0 330 135" className="w-full h-auto block">
                  {[25, 50, 75, 100].map((v) => {
                    const y = d3.scaleLinear().domain([0, 100]).range([112, 16])(v);
                    return (
                      <g key={v}>
                        <line
                          x1={34}
                          y1={y}
                          x2={310}
                          y2={y}
                          stroke="rgba(148,163,184,0.12)"
                          strokeDasharray="3 3"
                        />
                        <text
                          x={28}
                          y={y + 3}
                          textAnchor="end"
                          fill="rgba(148,163,184,0.65)"
                          fontSize={9}
                          fontFamily="var(--font-jetbrains), monospace"
                        >
                          {v}%
                        </text>
                      </g>
                    );
                  })}
                  {pckLinePath && (
                    <path
                      d={pckLinePath}
                      fill="none"
                      stroke="#34D399"
                      strokeWidth={2.5}
                      strokeLinecap="round"
                    />
                  )}
                  {[
                    { px: 5, val: m2.pck_5px_pct },
                    { px: 10, val: m2.pck_10px_pct },
                    { px: 20, val: m2.pck_20px_pct },
                  ].map((pt) => {
                    const cx = d3.scaleLinear().domain([4, 21]).range([36, 300])(pt.px);
                    const cy = d3.scaleLinear().domain([0, 100]).range([112, 16])(pt.val);
                    return (
                      <g key={pt.px}>
                        <circle
                          cx={cx}
                          cy={cy}
                          r={4.5}
                          fill="#34D399"
                          stroke="#060B10"
                          strokeWidth={1.5}
                        />
                        <text
                          x={cx}
                          y={cy - 8}
                          textAnchor="middle"
                          fill="#FFFFFF"
                          fontSize={9.5}
                          fontWeight={700}
                          fontFamily="var(--font-jetbrains), monospace"
                        >
                          {pt.val.toFixed(1)}%
                        </text>
                        <text
                          x={cx}
                          y={128}
                          textAnchor="middle"
                          fill="#94A3B8"
                          fontSize={9.5}
                          fontFamily="var(--font-jetbrains), monospace"
                        >
                          {pt.px}px
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>
          </section>
        )}
      </div>

      {/* ── Section 3: Model 3 17-Class Action Spotting Breakdown ── */}
      {m3 && (
        <section className="rounded-2xl obsidian-card p-5 space-y-5">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="space-y-1">
              <h2 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
                <Activity className="w-4 h-4 text-tv-mint shrink-0" />
                <span>{t("models.m3_title")}</span>
              </h2>
              <p className="text-xs text-slate-400">{t("models.m3_desc")}</p>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <div className="px-3 py-2 rounded-xl obsidian-well text-xs">
                <span className="text-slate-400 me-1.5">{t("models.m3_loose")}:</span>
                <BidiNum className="font-extrabold text-tv-mint">
                  {`${m3.test_metrics.loose_amap_pct.toFixed(2)}%`}
                </BidiNum>
              </div>
              <div className="px-3 py-2 rounded-xl obsidian-well text-xs">
                <span className="text-slate-400 me-1.5">{t("models.m3_tight")}:</span>
                <BidiNum className="font-extrabold text-tv-cyan">
                  {`${m3.test_metrics.tight_amap_pct.toFixed(2)}%`}
                </BidiNum>
              </div>
              <div className="px-3 py-2 rounded-xl obsidian-well text-xs">
                <span className="text-slate-400 me-1.5">{t("models.m3_map5")}:</span>
                <BidiNum className="font-extrabold text-white">
                  {`${m3.test_metrics.map_at_5s_pct.toFixed(2)}%`}
                </BidiNum>
              </div>
              <div className="px-3 py-2 rounded-xl obsidian-well text-xs">
                <span className="text-slate-400 me-1.5">{t("models.m3_params")}:</span>
                <BidiNum className="font-extrabold text-tv-amber">
                  {`${m3.params_millions}M`}
                </BidiNum>
              </div>
            </div>
          </div>

          <p className="text-xs font-mono text-slate-400 px-3 py-2 rounded-xl obsidian-well">
            <BidiNum>{m3.architecture}</BidiNum>
          </p>

          <D3ActionClassesChart
            perClassTightApPct={m3.test_metrics.per_class_tight_ap_pct}
          />
        </section>
      )}

      {/* ── Section 4: Model 4 Ablation & CPU INT8 ONNX Benchmarks ── */}
      {m4 && (
        <D3JerseyAndBenchmarksChart
          ablation={m4.parseq_ablation}
          benchmarks={suite.cpu_onnx_benchmarks}
        />
      )}
    </motion.div>
  );
}
