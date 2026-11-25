"use client";

import React, { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Upload,
  Cpu,
  Cloud,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ArrowRight,
  ArrowLeft,
  Sparkles,
  FileVideo,
  Ban,
  StopCircle,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import { useJobPolling } from "@/hooks/useJobPolling";
import { uploadMatch, cancelJob } from "@/lib/api";
import type { JobStage } from "@/lib/types";
import { BidiNum } from "@/components/ui/BidiNum";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onMatchCreated?: () => void;
}

export function UploadModal({ isOpen, onClose, onMatchCreated }: UploadModalProps) {
  const { t, isRTL } = useLang();
  const router = useRouter();
  const ArrowIcon = isRTL ? ArrowLeft : ArrowRight;

  // Form state (default CPU quick-scan = 3s)
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState<string>("");
  const [mode, setMode] = useState<"cpu_quick" | "remote_gpu">("cpu_quick");
  const [duration, setDuration] = useState<number>(3.0);
  const [remoteGpuUrl, setRemoteGpuUrl] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const {
    jobId,
    jobStatus,
    setJobStatus,
    isPolling,
    error: pollError,
    startPolling,
    cancelPolling,
    resetJob,
  } = useJobPolling();

  useEffect(() => {
    if (typeof window !== "undefined") {
      try {
        const savedUrl = localStorage.getItem("tv_remote_gpu_url");
        if (savedUrl) setRemoteGpuUrl(savedUrl);
      } catch {
        // Ignore storage errors
      }
    }
  }, []);

  const isTerminal =
    jobStatus?.status === "completed" ||
    jobStatus?.status === "failed" ||
    jobStatus?.status === "cancelled";

  const isJobRunning = (isSubmitting || isPolling) && !isTerminal;

  // Close UploadModal cleanly on Escape key when idle/terminal
  useEffect(() => {
    if (!isOpen || isJobRunning) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        resetJob();
        setFile(null);
        setTitle("");
        setIsSubmitting(false);
        setIsCancelling(false);
        setSubmitError(null);
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, isJobRunning, onClose, resetJob]);

  const handleClose = () => {
    if (isJobRunning) return;
    resetJob();
    setFile(null);
    setTitle("");
    setIsSubmitting(false);
    setIsCancelling(false);
    setSubmitError(null);
    onClose();
  };

  const handleResetToForm = () => {
    resetJob();
    setIsSubmitting(false);
    setIsCancelling(false);
    setSubmitError(null);
  };

  const handleCancelJob = async () => {
    const activeId = jobStatus?.job_id || jobId;
    if (!activeId || isCancelling) return;

    setIsCancelling(true);
    setSubmitError(null);
    try {
      const updated = await cancelJob(activeId);
      cancelPolling();
      setJobStatus(updated);
      setIsSubmitting(false);
    } catch (err: any) {
      setSubmitError(err?.message || t("upload.stage_failed"));
    } finally {
      setIsCancelling(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const validateAndSetFile = (f: File) => {
    const ext = f.name.toLowerCase();
    if (!ext.endsWith(".mp4") && !ext.endsWith(".mov") && !ext.endsWith(".mkv")) {
      setSubmitError(t("upload.err_ext"));
      return;
    }
    if (f.size > 250 * 1024 * 1024) {
      setSubmitError(t("upload.err_size"));
      return;
    }
    setSubmitError(null);
    setFile(f);
    if (!title) {
      const cleanName = f.name.replace(/\.[^/.]+$/, "").replace(/[-_]/g, " ");
      setTitle(cleanName.charAt(0).toUpperCase() + cleanName.slice(1));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setSubmitError(t("upload.err_no_file"));
      return;
    }

    if (mode === "remote_gpu") {
      if (!remoteGpuUrl.trim()) {
        setSubmitError(t("upload.err_no_gpu_url"));
        return;
      }
      try {
        localStorage.setItem("tv_remote_gpu_url", remoteGpuUrl.trim());
      } catch {
        // Ignore
      }
    }

    setIsSubmitting(true);
    setSubmitError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("title", title || file.name);
    formData.append("mode", mode);
    formData.append("max_seconds", duration.toString());
    if (mode === "remote_gpu") {
      formData.append("remote_gpu_url", remoteGpuUrl.trim());
    }

    try {
      const res = await uploadMatch(formData);
      startPolling(res.job_id);
    } catch (err: any) {
      setSubmitError(err?.message || t("upload.stage_failed"));
      setIsSubmitting(false);
    }
  };

  useEffect(() => {
    if (jobStatus?.status === "completed" && onMatchCreated) {
      onMatchCreated();
    }
  }, [jobStatus?.status, onMatchCreated]);

  const getStageStatus = (stage: "detecting" | "classifying" | "computing" | "encoding") => {
    if (!jobStatus) return "pending";
    const order: JobStage[] = [
      "queued",
      "detecting",
      "classifying",
      "computing",
      "encoding",
      "completed",
    ];
    const currentIdx = order.indexOf(jobStatus.status);
    const targetIdx = order.indexOf(stage);

    if (jobStatus.status === "failed" || jobStatus.status === "cancelled") return "stopped";
    if (currentIdx > targetIdx || jobStatus.status === "completed") return "done";
    if (currentIdx === targetIdx) return "active";
    return "pending";
  };

  const getLocalizedStageHeading = () => {
    if (!jobStatus) return t("upload.btn_uploading");
    switch (jobStatus.status) {
      case "queued":
        return t("upload.stage_queued");
      case "detecting":
        return t("upload.stage_detecting");
      case "classifying":
        return t("upload.stage_classifying");
      case "computing":
        return t("upload.stage_computing");
      case "encoding":
        return t("upload.stage_encoding");
      case "completed":
        return t("upload.stage_completed");
      case "failed":
        return t("upload.stage_failed");
      case "cancelled":
        return t("upload.stage_cancelled");
      default:
        return t("upload.btn_uploading");
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 overflow-y-auto">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={isJobRunning ? undefined : handleClose}
            className="fixed inset-0 bg-black/80 backdrop-blur-md"
          />

          {/* Modal Container */}
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 14 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 14 }}
            transition={{ type: "spring", duration: 0.35, bounce: 0 }}
            className="relative w-full max-w-2xl rounded-2xl obsidian-popover p-5 sm:p-7 shadow-2xl z-10 my-8 max-h-[92vh] overflow-y-auto"
          >
            {/* ── Modal Header ── */}
            <div className="flex items-start justify-between pb-4 border-b border-white/[0.08] mb-5">
              <div>
                <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-tv-cyan/10 border border-tv-cyan/25 text-xs font-semibold text-tv-cyan mb-2">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{t("upload.badge")}</span>
                </div>
                <h3 className="text-xl font-extrabold text-white tracking-tight">
                  {t("upload.title")}
                </h3>
              </div>

              {!isJobRunning && (
                <button
                  type="button"
                  onClick={handleClose}
                  className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors cursor-pointer"
                  aria-label={t("upload.close")}
                >
                  <X className="w-5 h-5" />
                </button>
              )}
            </div>

            {/* ── Active Processing View ── */}
            {isSubmitting || jobStatus ? (
              <div className="space-y-6 py-2">
                <div className="text-center space-y-1.5">
                  {jobStatus?.status === "completed" ? (
                    <div className="w-14 h-14 rounded-2xl bg-tv-mint/15 border border-tv-mint/30 flex items-center justify-center mx-auto mb-3 text-tv-mint">
                      <CheckCircle2 className="w-7 h-7" />
                    </div>
                  ) : jobStatus?.status === "failed" ? (
                    <div className="w-14 h-14 rounded-2xl bg-tv-coral/15 border border-tv-coral/30 flex items-center justify-center mx-auto mb-3 text-tv-coral">
                      <AlertCircle className="w-7 h-7" />
                    </div>
                  ) : jobStatus?.status === "cancelled" ? (
                    <div className="w-14 h-14 rounded-2xl bg-tv-amber/15 border border-tv-amber/30 flex items-center justify-center mx-auto mb-3 text-tv-amber">
                      <Ban className="w-7 h-7" />
                    </div>
                  ) : (
                    <div className="w-14 h-14 rounded-2xl bg-tv-cyan/15 border border-tv-cyan/30 flex items-center justify-center mx-auto mb-3 text-tv-cyan">
                      <Loader2 className="w-7 h-7 animate-spin" />
                    </div>
                  )}

                  <h4 className="text-lg font-bold text-white">
                    {getLocalizedStageHeading()}
                  </h4>

                  <p className="text-xs text-slate-400">
                    {jobStatus?.elapsed_s ? (
                      <>
                        <BidiNum className="text-slate-200 font-semibold">
                          {jobStatus.elapsed_s}s
                        </BidiNum>{" "}
                        <span>{t("upload.elapsed_suffix")}</span>
                      </>
                    ) : (
                      <span>{t("upload.initializing")}</span>
                    )}
                  </p>
                </div>

                {/* Progress Bar (Strict LTR Fill) */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400 font-medium">
                      {t("upload.progress_label")}
                    </span>
                    <BidiNum className="font-bold text-tv-cyan">
                      {(jobStatus?.progress_pct ?? 0).toFixed(0)}%
                    </BidiNum>
                  </div>
                  <div
                    dir="ltr"
                    className="h-2.5 w-full obsidian-well rounded-full overflow-hidden p-0.5"
                  >
                    <div
                      className={`h-full transition-all duration-500 rounded-full ${
                        jobStatus?.status === "failed"
                          ? "bg-tv-coral"
                          : jobStatus?.status === "cancelled"
                          ? "bg-tv-amber"
                          : jobStatus?.status === "completed"
                          ? "bg-tv-mint"
                          : "bg-gradient-to-r from-tv-cyan to-tv-mint"
                      }`}
                      style={{ width: `${Math.max(jobStatus?.progress_pct ?? 5, 5)}%` }}
                    />
                  </div>
                </div>

                {/* 4 Pipeline Stages Checklist */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-1">
                  {(
                    [
                      ["detecting", "upload.stage_detecting"],
                      ["classifying", "upload.stage_classifying"],
                      ["computing", "upload.stage_computing"],
                      ["encoding", "upload.stage_encoding"],
                    ] as const
                  ).map(([stageKey, labelKey]) => {
                    const st = getStageStatus(stageKey);
                    return (
                      <div
                        key={stageKey}
                        className={`p-3.5 rounded-xl border flex items-center gap-3 transition-colors ${
                          st === "done"
                            ? "bg-tv-mint/[0.06] border-tv-mint/25 text-tv-mint"
                            : st === "active"
                            ? "bg-tv-cyan/10 border-tv-cyan/35 text-white"
                            : "obsidian-well text-slate-500"
                        }`}
                      >
                        {st === "done" ? (
                          <CheckCircle2 className="w-4 h-4 shrink-0 text-tv-mint" />
                        ) : st === "active" ? (
                          <Loader2 className="w-4 h-4 shrink-0 text-tv-cyan animate-spin" />
                        ) : (
                          <div className="w-4 h-4 rounded-full border border-slate-700 shrink-0" />
                        )}
                        <span className="text-xs font-semibold">{t(labelKey)}</span>
                      </div>
                    );
                  })}
                </div>

                {/* Error message */}
                {(jobStatus?.error || submitError || pollError) && (
                  <div
                    dir="ltr"
                    className="p-3.5 rounded-xl bg-tv-coral/10 border border-tv-coral/25 text-tv-coral text-xs whitespace-pre-wrap max-h-36 overflow-y-auto font-mono"
                  >
                    {jobStatus?.error || submitError || pollError}
                  </div>
                )}

                {/* Bottom Action CTA */}
                <div className="pt-2 space-y-2.5">
                  {jobStatus?.status === "completed" && jobStatus.match_id ? (
                    <button
                      type="button"
                      onClick={() => {
                        handleClose();
                        router.push(`/match/${jobStatus.match_id}`);
                      }}
                      className="w-full py-3.5 px-5 rounded-xl font-bold text-sm bg-tv-cyan text-slate-950 hover:bg-tv-cyan/90 transition-all shadow-lg shadow-tv-cyan/20 flex items-center justify-center gap-2 cursor-pointer"
                    >
                      <span>{t("upload.open_warroom")}</span>
                      <ArrowIcon className="w-4 h-4" />
                    </button>
                  ) : jobStatus?.status === "failed" || jobStatus?.status === "cancelled" ? (
                    <div className="flex items-center gap-3">
                      <button
                        type="button"
                        onClick={handleResetToForm}
                        className="flex-1 py-3 px-4 rounded-xl font-bold text-xs bg-tv-cyan text-slate-950 hover:bg-tv-cyan/90 transition-colors cursor-pointer"
                      >
                        {t("upload.retry")}
                      </button>
                      <button
                        type="button"
                        onClick={handleClose}
                        className="py-3 px-5 rounded-xl font-semibold text-xs bg-white/[0.08] hover:bg-white/[0.14] text-white transition-colors cursor-pointer"
                      >
                        {t("upload.close")}
                      </button>
                    </div>
                  ) : (
                    <>
                      <button
                        type="button"
                        disabled={isCancelling || !(jobStatus?.job_id || jobId)}
                        onClick={handleCancelJob}
                        className="w-full py-3 px-4 rounded-xl font-bold text-xs bg-tv-coral/15 border border-tv-coral/40 text-tv-coral hover:bg-tv-coral hover:text-white disabled:opacity-50 transition-all flex items-center justify-center gap-2 cursor-pointer"
                      >
                        {isCancelling ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>{t("upload.btn_cancelling")}</span>
                          </>
                        ) : (
                          <>
                            <StopCircle className="w-4 h-4" />
                            <span>{t("upload.btn_cancel")}</span>
                          </>
                        )}
                      </button>
                      <p className="text-center text-xs text-slate-500">
                        {t("upload.background_note")}
                      </p>
                    </>
                  )}
                </div>
              </div>
            ) : (
              /* ── File Selection & Configuration Form ── */
              <form onSubmit={handleSubmit} className="space-y-5">
                {/* Drag & Drop File Area */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ${
                    isDragOver
                      ? "border-tv-cyan bg-tv-cyan/[0.06]"
                      : file
                      ? "border-tv-mint/40 bg-tv-mint/[0.03]"
                      : "border-white/15 obsidian-well hover:border-white/25"
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".mp4,.mov,.mkv"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        validateAndSetFile(e.target.files[0]);
                      }
                    }}
                    className="hidden"
                  />

                  {file ? (
                    <div className="flex items-center justify-center gap-3.5">
                      <div className="w-11 h-11 rounded-xl bg-tv-mint/15 border border-tv-mint/30 flex items-center justify-center text-tv-mint shrink-0">
                        <FileVideo className="w-5 h-5" />
                      </div>
                      <div className="text-start min-w-0">
                        <p className="text-sm font-bold text-white max-w-xs truncate">
                          {file.name}
                        </p>
                        <p className="text-xs text-slate-400 mt-0.5">
                          <BidiNum className="text-tv-mint font-semibold">
                            {(file.size / (1024 * 1024)).toFixed(1)} MB
                          </BidiNum>{" "}
                          · {t("upload.click_replace")}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="w-12 h-12 rounded-2xl bg-tv-cyan/10 border border-tv-cyan/25 flex items-center justify-center mx-auto text-tv-cyan">
                        <Upload className="w-5 h-5" />
                      </div>
                      <p className="text-sm font-bold text-white">
                        {t("upload.drag_drop_title")}
                      </p>
                      <p className="text-xs text-slate-400">
                        {t("upload.drag_drop_sub")}
                      </p>
                    </div>
                  )}
                </div>

                {/* Match Title Input */}
                <div className="space-y-1.5">
                  <label className="text-xs font-semibold text-slate-300 block">
                    {t("upload.title_input_label")}
                  </label>
                  <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder={t("upload.title_input_placeholder")}
                    className="w-full px-3.5 py-2.5 rounded-xl obsidian-well text-xs sm:text-sm text-white placeholder-slate-500 focus:outline-none focus:border-tv-cyan/60"
                  />
                </div>

                {/* Hybrid Mode Selector */}
                <div className="space-y-2">
                  <label className="text-xs font-semibold text-slate-300 block">
                    {t("upload.engine_mode_label")}
                  </label>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <button
                      type="button"
                      onClick={() => setMode("cpu_quick")}
                      className={`p-4 rounded-xl border text-start cursor-pointer transition-all ${
                        mode === "cpu_quick"
                          ? "bg-tv-cyan/10 border-tv-cyan/60 shadow-md shadow-tv-cyan/10"
                          : "obsidian-well hover:border-white/15"
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1.5">
                        <Cpu
                          className={`w-4 h-4 shrink-0 ${
                            mode === "cpu_quick" ? "text-tv-cyan" : "text-slate-400"
                          }`}
                        />
                        <span className="text-xs font-bold text-white">
                          {t("upload.mode_cpu")}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        {t("upload.mode_cpu_sub")}
                      </p>
                    </button>

                    <button
                      type="button"
                      onClick={() => setMode("remote_gpu")}
                      className={`p-4 rounded-xl border text-start cursor-pointer transition-all ${
                        mode === "remote_gpu"
                          ? "bg-tv-cyan/10 border-tv-cyan/60 shadow-md shadow-tv-cyan/10"
                          : "obsidian-well hover:border-white/15"
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1.5">
                        <Cloud
                          className={`w-4 h-4 shrink-0 ${
                            mode === "remote_gpu" ? "text-tv-cyan" : "text-slate-400"
                          }`}
                        />
                        <span className="text-xs font-bold text-white">
                          {t("upload.mode_gpu")}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        {t("upload.mode_gpu_sub")}
                      </p>
                    </button>
                  </div>
                </div>

                {/* Mode-Specific Sub-Options */}
                {mode === "cpu_quick" ? (
                  <div className="space-y-2.5 p-4 rounded-xl obsidian-well">
                    <label className="text-xs font-semibold text-slate-300 block">
                      {t("upload.duration_label")}
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      {[2.0, 3.0, 5.0].map((sec) => (
                        <button
                          key={`dur-${sec}`}
                          type="button"
                          onClick={() => setDuration(sec)}
                          className={`py-2 px-2.5 rounded-xl text-xs font-bold transition-colors cursor-pointer flex items-center justify-center gap-1.5 border ${
                            duration === sec
                              ? "editorial-tab-active border-tv-cyan/60 text-white shadow-sm"
                              : "bg-white/[0.04] text-slate-300 hover:text-white border-white/[0.06]"
                          }`}
                        >
                          <span className="inline-flex items-baseline gap-0.5">
                            <BidiNum>{sec}</BidiNum>
                            <span>{t("seconds")}</span>
                          </span>
                          {sec === 3.0 && (
                            <span className="text-[11px] text-tv-cyan">
                              ({t("upload.rec_short")})
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed pt-0.5">
                      {t("upload.cpu_hint")}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-1.5 p-4 rounded-xl obsidian-well">
                    <label className="text-xs font-semibold text-slate-300 block">
                      {t("upload.gpu_url_label")}
                    </label>
                    <input
                      dir="ltr"
                      type="url"
                      value={remoteGpuUrl}
                      onChange={(e) => setRemoteGpuUrl(e.target.value)}
                      placeholder={t("upload.gpu_url_placeholder")}
                      className="w-full px-3.5 py-2.5 rounded-xl bg-black/40 border border-white/10 text-xs font-mono text-white placeholder-slate-500 focus:outline-none focus:border-tv-cyan"
                    />
                  </div>
                )}

                {/* Error Banner */}
                {submitError && (
                  <div className="p-3 rounded-xl bg-tv-coral/10 border border-tv-coral/25 text-tv-coral text-xs">
                    {submitError}
                  </div>
                )}

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={!file}
                  className={`w-full py-3.5 px-5 rounded-xl font-bold text-sm transition-all shadow-lg ${
                    file
                      ? "bg-tv-cyan text-slate-950 hover:bg-tv-cyan/90 shadow-tv-cyan/20 active:scale-[0.99] cursor-pointer"
                      : "bg-white/[0.04] text-slate-600 cursor-not-allowed border border-white/[0.06]"
                  }`}
                >
                  {t("upload.btn_start")}
                </button>
              </form>
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
