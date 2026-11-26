"use client";

import React, { useMemo } from "react";
import * as d3 from "d3";
import { Cpu, ShieldCheck } from "lucide-react";
import { useLang } from "@/context/LangContext";
import type { ModelsSuiteResponse } from "@/lib/types";
import { BidiNum } from "@/components/ui/BidiNum";

interface D3JerseyAndBenchmarksChartProps {
  ablation: NonNullable<
    ModelsSuiteResponse["metrics"]["model_4_jersey_recognition"]
  >["parseq_ablation"];
  benchmarks: ModelsSuiteResponse["cpu_onnx_benchmarks"];
}

export function D3JerseyAndBenchmarksChart({
  ablation,
  benchmarks,
}: D3JerseyAndBenchmarksChartProps) {
  const { t } = useLang();

  const ablationRows = useMemo(
    () => [
      {
        key: "overall",
        label: t("models.m4_overall_acc"),
        raw: ablation.raw_parseq_no_gate.overall_acc_pct,
        gated: ablation.full_gated_pipeline.overall_acc_pct,
      },
      {
        key: "legible",
        label: t("models.m4_legible_acc"),
        raw: ablation.raw_parseq_no_gate.legible_acc_pct,
        gated: ablation.full_gated_pipeline.legible_acc_pct,
      },
      {
        key: "reject",
        label: t("models.m4_reject_rate"),
        raw: ablation.raw_parseq_no_gate.illegible_rejection_pct,
        gated: ablation.full_gated_pipeline.illegible_rejection_pct,
      },
    ],
    [ablation, t],
  );

  const ablationScale = useMemo(
    () => d3.scaleLinear().domain([0, 80]).range([0, 100]),
    [],
  );

  const benchCards = useMemo(() => {
    const list: Array<{
      name: string;
      fp32Ms: number;
      fp32Fps: number;
      int8Ms: number;
      int8Fps: number;
      fp32Mb: number;
      int8Mb: number;
    }> = [];
    if (benchmarks.model_3_action_spotting_onnx) {
      const m3 = benchmarks.model_3_action_spotting_onnx;
      list.push({
        name: "Model 3 · Action Spotting",
        fp32Ms: m3.cpu_latency_ms,
        fp32Fps: m3.cpu_fps,
        int8Ms: m3.int8_cpu_benchmark.cpu_latency_ms,
        int8Fps: m3.int8_cpu_benchmark.cpu_fps,
        fp32Mb: m3.size_mb,
        int8Mb: m3.int8_cpu_benchmark.size_mb,
      });
    }
    if (benchmarks.model_4_jersey_legibility_onnx) {
      const m4 = benchmarks.model_4_jersey_legibility_onnx;
      list.push({
        name: "Model 4 · Jersey Gate",
        fp32Ms: m4.cpu_latency_ms,
        fp32Fps: m4.cpu_fps,
        int8Ms: m4.int8_cpu_benchmark.cpu_latency_ms,
        int8Fps: m4.int8_cpu_benchmark.cpu_fps,
        fp32Mb: m4.size_mb,
        int8Mb: m4.int8_cpu_benchmark.size_mb,
      });
    }
    return list;
  }, [benchmarks]);

  const maxLatency = useMemo(
    () => Math.max(25, d3.max(benchCards, (c) => c.fp32Ms) ?? 25),
    [benchCards],
  );

  const latencyScale = useMemo(
    () => d3.scaleLinear().domain([0, maxLatency]).range([0, 100]),
    [maxLatency],
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      {/* Panel A: Model 4 Gate + PARSeq Ablation */}
      <div className="rounded-2xl obsidian-card p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-tv-mint shrink-0" />
            <span>{t("models.m4_title")}</span>
          </h3>
          <div className="flex items-center gap-3 text-[11px] font-semibold">
            <span className="inline-flex items-center gap-1.5 text-slate-400">
              <span className="w-2.5 h-2.5 rounded-sm bg-slate-500" />
              <span>{t("models.m4_raw_parseq")}</span>
            </span>
            <span className="inline-flex items-center gap-1.5 text-tv-mint">
              <span className="w-2.5 h-2.5 rounded-sm bg-tv-mint" />
              <span>{t("models.m4_gated_pipe")}</span>
            </span>
          </div>
        </div>

        <p className="text-xs text-slate-400">{t("models.m4_desc")}</p>

        <div className="space-y-4 pt-1">
          {ablationRows.map((row) => {
            const delta = (row.gated - row.raw).toFixed(2);
            return (
              <div key={row.key} className="space-y-1.5 p-3 rounded-xl obsidian-well">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-white">{row.label}</span>
                  <BidiNum className="px-2 py-0.5 rounded bg-tv-mint/15 text-tv-mint font-bold text-[11px]">
                    {`+${delta}%`}
                  </BidiNum>
                </div>

                <div className="space-y-1.5">
                  <div className="flex items-center gap-2 text-[11px]">
                    <div dir="ltr" className="flex-1 h-2 rounded-full bg-white/[0.06] overflow-hidden">
                      <div
                        className="h-full rounded-full bg-slate-500"
                        style={{ width: `${Math.min(100, ablationScale(row.raw))}%` }}
                      />
                    </div>
                    <BidiNum className="w-14 text-end text-slate-400">
                      {`${row.raw.toFixed(2)}%`}
                    </BidiNum>
                  </div>

                  <div className="flex items-center gap-2 text-[11px]">
                    <div dir="ltr" className="flex-1 h-2.5 rounded-full bg-white/[0.06] overflow-hidden">
                      <div
                        className="h-full rounded-full bg-tv-mint"
                        style={{ width: `${Math.min(100, ablationScale(row.gated))}%` }}
                      />
                    </div>
                    <BidiNum className="w-14 text-end text-tv-mint font-bold">
                      {`${row.gated.toFixed(2)}%`}
                    </BidiNum>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Panel B: CPU ONNX FP32 vs INT8 Benchmarks */}
      <div className="rounded-2xl obsidian-card p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm sm:text-base font-bold text-white flex items-center gap-2">
            <Cpu className="w-4 h-4 text-tv-amber shrink-0" />
            <span>{t("models.bench_title")}</span>
          </h3>
          <div className="flex items-center gap-3 text-[11px] font-semibold">
            <span className="inline-flex items-center gap-1.5 text-tv-cyan">
              <span className="w-2.5 h-2.5 rounded-sm bg-tv-cyan/70" />
              <span>{t("models.fp32")}</span>
            </span>
            <span className="inline-flex items-center gap-1.5 text-tv-mint">
              <span className="w-2.5 h-2.5 rounded-sm bg-tv-mint" />
              <span>{t("models.int8")}</span>
            </span>
          </div>
        </div>

        <p className="text-xs text-slate-400">{t("models.bench_desc")}</p>

        <div className="space-y-4 pt-1">
          {benchCards.map((c) => (
            <div key={c.name} className="p-3.5 rounded-xl obsidian-well space-y-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-white">{c.name}</span>
                <BidiNum className="text-[11px] text-tv-mint font-bold">
                  {`${c.int8Fps} FPS INT8`}
                </BidiNum>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-12 text-slate-400 font-mono">FP32</span>
                  <div dir="ltr" className="flex-1 h-2 rounded-full bg-white/[0.06] overflow-hidden">
                    <div
                      className="h-full rounded-full bg-tv-cyan/75"
                      style={{ width: `${latencyScale(c.fp32Ms)}%` }}
                    />
                  </div>
                  <BidiNum className="w-28 text-end text-slate-300">
                    {`${c.fp32Ms}ms · ${c.fp32Mb}MB`}
                  </BidiNum>
                </div>

                <div className="flex items-center gap-2 text-[11px]">
                  <span className="w-12 text-tv-mint font-mono font-bold">INT8</span>
                  <div dir="ltr" className="flex-1 h-2.5 rounded-full bg-white/[0.06] overflow-hidden">
                    <div
                      className="h-full rounded-full bg-tv-mint"
                      style={{ width: `${latencyScale(c.int8Ms)}%` }}
                    />
                  </div>
                  <BidiNum className="w-28 text-end text-tv-mint font-bold">
                    {`${c.int8Ms}ms · ${c.int8Mb}MB`}
                  </BidiNum>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
