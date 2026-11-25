"use client";

import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Maximize,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  Tv,
  Radar,
  Gauge,
  HelpCircle,
} from "lucide-react";
import { useLang } from "@/context/LangContext";
import { getVideoUrl } from "@/lib/api";
import { BidiNum } from "@/components/ui/BidiNum";

export type VideoSourceType = "broadcast" | "radar";

export interface VideoPlayerHandle {
  seekToTime: (timeSec: number) => void;
  seekToFrame: (frameIdx: number) => void;
  play: () => void;
  pause: () => void;
  getCurrentTime: () => number;
  getVideoElement: () => HTMLVideoElement | null;
}

interface VideoPlayerProps {
  matchId: string;
  fps: number;
  totalFrames: number;
  duration_s: number;
  onTimeUpdate?: (currentTime: number, currentFrame: number) => void;
  dockedBottom?: boolean;
}

export const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(
  function VideoPlayer(
    {
      matchId,
      fps = 25.0,
      totalFrames = 750,
      duration_s = 30.0,
      onTimeUpdate,
      dockedBottom = false,
    },
    ref
  ) {
    const { t, dir } = useLang();
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const shortcutsRef = useRef<HTMLDivElement | null>(null);

    const [sourceType, setSourceType] = useState<VideoSourceType>("broadcast");
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const [isMuted, setIsMuted] = useState<boolean>(true);
    const [currentTime, setCurrentTime] = useState<number>(0);
    const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
    const [isScrubbing, setIsScrubbing] = useState<boolean>(false);
    const [showShortcuts, setShowShortcuts] = useState<boolean>(false);

    const savedTimeBeforeSwitch = useRef<number>(0);
    const wasPlayingBeforeSwitch = useRef<boolean>(false);

    const currentFrame = Math.min(
      Math.max(1, totalFrames) - 1,
      Math.max(0, Math.floor(currentTime * fps))
    );

    useImperativeHandle(ref, () => ({
      seekToTime: (timeSec: number) => {
        if (videoRef.current) {
          const clamped = Math.max(0, Math.min(duration_s, timeSec));
          videoRef.current.currentTime = clamped;
          setCurrentTime(clamped);
          onTimeUpdate?.(clamped, Math.floor(clamped * fps));
        }
      },
      seekToFrame: (frameIdx: number) => {
        if (videoRef.current) {
          const timeSec = frameIdx / fps;
          const clamped = Math.max(0, Math.min(duration_s, timeSec));
          videoRef.current.currentTime = clamped;
          setCurrentTime(clamped);
          onTimeUpdate?.(clamped, frameIdx);
        }
      },
      play: () => {
        videoRef.current?.play().catch(() => {});
      },
      pause: () => {
        videoRef.current?.pause();
      },
      getCurrentTime: () => videoRef.current?.currentTime ?? 0,
      getVideoElement: () => videoRef.current,
    }));

    const handleSourceSwitch = (newSource: VideoSourceType) => {
      if (newSource === sourceType || !videoRef.current) return;
      savedTimeBeforeSwitch.current = videoRef.current.currentTime;
      wasPlayingBeforeSwitch.current = !videoRef.current.paused;
      setSourceType(newSource);
    };

    const handleLoadedMetadata = () => {
      if (!videoRef.current) return;
      if (savedTimeBeforeSwitch.current > 0) {
        videoRef.current.currentTime = savedTimeBeforeSwitch.current;
      }
      videoRef.current.playbackRate = playbackSpeed;
      if (wasPlayingBeforeSwitch.current) {
        videoRef.current.play().catch(() => {});
      }
    };

    const togglePlay = useCallback(() => {
      if (!videoRef.current) return;
      if (videoRef.current.paused) {
        videoRef.current.play().catch(() => {});
      } else {
        videoRef.current.pause();
      }
    }, []);

    const stepFrame = useCallback(
      (deltaFrames: number) => {
        if (!videoRef.current) return;
        videoRef.current.pause();
        const stepTime = 1 / fps;
        const newTime = Math.max(
          0,
          Math.min(duration_s, videoRef.current.currentTime + deltaFrames * stepTime)
        );
        videoRef.current.currentTime = newTime;
        setCurrentTime(newTime);
        onTimeUpdate?.(newTime, Math.floor(newTime * fps));
      },
      [fps, duration_s, onTimeUpdate]
    );

    const handleSpeedChange = (speed: number) => {
      setPlaybackSpeed(speed);
      if (videoRef.current) {
        videoRef.current.playbackRate = speed;
      }
    };

    const toggleMute = useCallback(() => {
      if (!videoRef.current) return;
      const nextMuted = !videoRef.current.muted;
      videoRef.current.muted = nextMuted;
      setIsMuted(nextMuted);
    }, []);

    const toggleFullscreen = () => {
      if (!containerRef.current) return;
      if (!document.fullscreenElement) {
        containerRef.current.requestFullscreen?.().catch(() => {});
      } else {
        document.exitFullscreen?.().catch(() => {});
      }
    };

    const handleTimeUpdate = () => {
      if (videoRef.current && !isScrubbing) {
        const tm = videoRef.current.currentTime;
        setCurrentTime(tm);
        onTimeUpdate?.(tm, Math.floor(tm * fps));
      }
    };

    const handleScrubberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const newTime = parseFloat(e.target.value);
      setCurrentTime(newTime);
      if (videoRef.current) {
        videoRef.current.currentTime = newTime;
        onTimeUpdate?.(newTime, Math.floor(newTime * fps));
      }
    };

    // Close shortcuts popover on outside click
    useEffect(() => {
      const handleOutside = (e: MouseEvent) => {
        if (shortcutsRef.current && !shortcutsRef.current.contains(e.target as Node)) {
          setShowShortcuts(false);
        }
      };
      document.addEventListener("mousedown", handleOutside);
      return () => document.removeEventListener("mousedown", handleOutside);
    }, []);

    // Analyst Keyboard Shortcuts
    useEffect(() => {
      const handleKeyDown = (e: KeyboardEvent) => {
        const activeTag = document.activeElement?.tagName.toLowerCase();
        if (activeTag === "input" || activeTag === "textarea" || activeTag === "select") {
          return;
        }

        if (e.code === "Space") {
          e.preventDefault();
          togglePlay();
        } else if (e.code === "ArrowLeft") {
          e.preventDefault();
          if (e.shiftKey) {
            if (videoRef.current) {
              const newTime = Math.max(0, videoRef.current.currentTime - 2.0);
              videoRef.current.currentTime = newTime;
              setCurrentTime(newTime);
              onTimeUpdate?.(newTime, Math.floor(newTime * fps));
            }
          } else {
            stepFrame(-1);
          }
        } else if (e.code === "ArrowRight") {
          e.preventDefault();
          if (e.shiftKey) {
            if (videoRef.current) {
              const newTime = Math.min(duration_s, videoRef.current.currentTime + 2.0);
              videoRef.current.currentTime = newTime;
              setCurrentTime(newTime);
              onTimeUpdate?.(newTime, Math.floor(newTime * fps));
            }
          } else {
            stepFrame(1);
          }
        } else if (e.code === "KeyM") {
          e.preventDefault();
          toggleMute();
        }
      };

      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }, [togglePlay, stepFrame, toggleMute, duration_s, fps, onTimeUpdate]);

    const formatTimecode = (sec: number) => {
      const m = Math.floor(sec / 60);
      const s = sec % 60;
      return `${m.toString().padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
    };

    const videoUrl = getVideoUrl(matchId, sourceType);
    const safeDuration = Math.max(0.1, duration_s || 30.0);
    const progressPct = Math.min(100, Math.max(0, (currentTime / safeDuration) * 100));

    return (
      <div
        ref={containerRef}
        className={`flex flex-col relative select-none bg-[#050912] ${
          dockedBottom
            ? "rounded-t-2xl overflow-visible"
            : "rounded-2xl obsidian-card overflow-hidden"
        }`}
      >
        {/* ── Compact Top Studio Bezel ── */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-2.5 sm:px-3 py-1.5 bg-[#0A101C]/95 border-b border-white/[0.07] rounded-t-2xl">
          {/* Calm Editorial Dual-Source Switcher */}
          <div className="inline-flex items-center p-0.5 rounded-xl obsidian-well max-w-full">
            <button
              type="button"
              onClick={() => handleSourceSwitch("broadcast")}
              className={`relative flex items-center gap-1.5 px-2 sm:px-2.5 py-1 rounded-lg text-[11px] sm:text-xs font-semibold whitespace-nowrap transition-colors cursor-pointer ${
                sourceType === "broadcast"
                  ? "text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {sourceType === "broadcast" && (
                <motion.span
                  layoutId="video-source-segmented"
                  transition={{ type: "spring", stiffness: 460, damping: 34 }}
                  className="absolute inset-0 rounded-lg editorial-tab-active"
                >
                  <span className="absolute bottom-0 inset-x-2 h-[2px] rounded-full bg-tv-cyan" />
                </motion.span>
              )}
              <Tv className="w-3.5 h-3.5 relative z-10 text-tv-cyan shrink-0" />
              <span className="relative z-10">{t("wr.broadcast")}</span>
            </button>

            <button
              type="button"
              onClick={() => handleSourceSwitch("radar")}
              className={`relative flex items-center gap-1.5 px-2 sm:px-2.5 py-1 rounded-lg text-[11px] sm:text-xs font-semibold whitespace-nowrap transition-colors cursor-pointer ${
                sourceType === "radar"
                  ? "text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {sourceType === "radar" && (
                <motion.span
                  layoutId="video-source-segmented"
                  transition={{ type: "spring", stiffness: 460, damping: 34 }}
                  className="absolute inset-0 rounded-lg editorial-tab-active"
                >
                  <span className="absolute bottom-0 inset-x-2 h-[2px] rounded-full bg-tv-cyan" />
                </motion.span>
              )}
              <Radar className="w-3.5 h-3.5 relative z-10 text-tv-cyan shrink-0" />
              <span className="relative z-10">{t("wr.radar")}</span>
            </button>
          </div>

          {/* Timecode, Frame Counter & [?] Shortcuts Popover */}
          <div className="flex items-center gap-1.5 ms-auto">
            <BidiNum className="px-2 py-0.5 rounded-lg obsidian-well text-[11px] text-slate-300">
              <span className="text-tv-cyan font-bold">
                {formatTimecode(currentTime)}
              </span>
              <span className="text-slate-600 mx-1">/</span>
              <span>{formatTimecode(duration_s)}</span>
            </BidiNum>

            <BidiNum className="hidden sm:inline-block px-2 py-0.5 rounded-lg obsidian-well text-[11px] text-tv-mint font-bold">
              F {currentFrame + 1}/{totalFrames}
            </BidiNum>

            {/* [?] Keyboard Shortcuts Popover Trigger */}
            <div ref={shortcutsRef} className="relative hidden sm:block">
              <button
                type="button"
                onClick={() => setShowShortcuts((prev) => !prev)}
                className={`p-1.5 rounded-lg border text-xs transition-colors cursor-pointer ${
                  showShortcuts
                    ? "bg-tv-cyan/15 border-tv-cyan/40 text-tv-cyan"
                    : "obsidian-well text-slate-400 hover:text-white"
                }`}
                title={t("wr.shortcuts_title")}
                aria-label={t("wr.shortcuts_title")}
              >
                <HelpCircle className="w-3.5 h-3.5" />
              </button>

              <AnimatePresence>
                {showShortcuts && (
                  <motion.div
                    dir={dir}
                    initial={{ opacity: 0, y: 6, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 6, scale: 0.97 }}
                    transition={{ duration: 0.15 }}
                    className="absolute top-full mt-2 end-0 w-64 rounded-xl obsidian-popover p-3 shadow-2xl z-50 space-y-2"
                  >
                    <div className="text-xs font-bold text-white pb-1.5 border-b border-white/[0.08]">
                      {t("wr.shortcuts_title")}
                    </div>
                    {[
                      ["Space", t("wr.sc_play")],
                      ["← / →", t("wr.sc_step")],
                      ["Shift + ←/→", t("wr.sc_jump")],
                      ["M", t("wr.sc_mute")],
                      ["Esc", t("wr.sc_deselect")],
                    ].map(([keyLabel, actionLabel]) => (
                      <div
                        key={keyLabel}
                        className="flex items-center justify-between gap-2 text-xs"
                      >
                        <span className="text-slate-300">{actionLabel}</span>
                        <BidiNum className="px-1.5 py-0.5 rounded bg-white/10 border border-white/10 text-[11px] font-semibold text-tv-cyan">
                          {keyLabel}
                        </BidiNum>
                      </div>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* ── Video Viewport (Constrained height on laptop so entire cockpit stays above the fold) ── */}
        <div className="relative aspect-video w-full xl:max-h-[calc(100vh-285px)] bg-black flex items-center justify-center overflow-hidden group">
          <video
            ref={videoRef}
            src={videoUrl}
            playsInline
            muted={isMuted}
            onTimeUpdate={handleTimeUpdate}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onLoadedMetadata={handleLoadedMetadata}
            onClick={togglePlay}
            className="w-full h-full object-contain cursor-pointer"
          />

          {!isPlaying && (
            <div
              onClick={togglePlay}
              className="absolute inset-0 flex items-center justify-center bg-black/25 cursor-pointer transition-opacity group-hover:bg-black/35"
            >
              <div className="w-14 h-14 rounded-full bg-tv-cyan/90 text-slate-950 flex items-center justify-center shadow-xl shadow-tv-cyan/30 transform scale-95 group-hover:scale-105 transition-transform">
                <Play className="w-6 h-6 fill-current translate-x-0.5" />
              </div>
            </div>
          )}
        </div>

        {/* ── Compact Analyst Playback Controls + Strictly LTR Range Scrubber ── */}
        <div className="px-3 py-2 bg-[#0A101C] border-t border-white/[0.07] flex flex-col gap-2">
          {/* CRITICAL RTL FIX: dir="ltr" ensures thumb & gradient move Left->Right identically in AR & EN */}
          <div dir="ltr" className="relative flex items-center w-full">
            <input
              dir="ltr"
              type="range"
              min={0}
              max={safeDuration}
              step={1 / fps}
              value={currentTime}
              onChange={handleScrubberChange}
              onMouseDown={() => setIsScrubbing(true)}
              onMouseUp={() => setIsScrubbing(false)}
              onTouchStart={() => setIsScrubbing(true)}
              onTouchEnd={() => setIsScrubbing(false)}
              aria-label="Video timeline scrubber"
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-tv-cyan focus:outline-none"
              style={{
                background: `linear-gradient(to right, #38BDF8 ${progressPct}%, #1E293B ${progressPct}%)`,
              }}
            />
          </div>

          {/* Transport & Speed Controls (Clean responsive wrapping so 375px never overflows) */}
          <div className="flex flex-wrap items-center justify-between gap-2">
            {/* Transport Cluster (LTR time direction so -1f is left, +1f is right) */}
            <div dir="ltr" className="flex items-center gap-1.5 shrink-0">
              <button
                type="button"
                onClick={togglePlay}
                className="p-1.5 rounded-lg editorial-tab-active text-tv-cyan hover:bg-white/[0.15] active:scale-95 transition-all cursor-pointer"
                aria-label={isPlaying ? t("wr.pause") : t("wr.play")}
                title={isPlaying ? t("wr.pause") : t("wr.play")}
              >
                {isPlaying ? (
                  <Pause className="w-4 h-4 fill-current" />
                ) : (
                  <Play className="w-4 h-4 fill-current translate-x-0.5" />
                )}
              </button>

              <button
                type="button"
                onClick={() => stepFrame(-1)}
                className="inline-flex items-center gap-0.5 px-2 py-1 rounded-lg obsidian-well text-xs font-mono text-slate-300 hover:text-white transition-colors cursor-pointer"
                title={t("wr.step_back")}
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                <span>-1f</span>
              </button>

              <button
                type="button"
                onClick={() => stepFrame(1)}
                className="inline-flex items-center gap-0.5 px-2 py-1 rounded-lg obsidian-well text-xs font-mono text-slate-300 hover:text-white transition-colors cursor-pointer"
                title={t("wr.step_fwd")}
              >
                <span>+1f</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>

              <button
                type="button"
                onClick={() => {
                  if (videoRef.current) {
                    videoRef.current.currentTime = 0;
                    setCurrentTime(0);
                    onTimeUpdate?.(0, 0);
                  }
                }}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.06] transition-colors cursor-pointer"
                title={t("wr.restart")}
                aria-label={t("wr.restart")}
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Speed, Mute & Fullscreen */}
            <div className="flex items-center gap-1.5 ms-auto shrink-0">
              <div
                dir="ltr"
                className="inline-flex items-center gap-0.5 p-0.5 rounded-lg obsidian-well"
              >
                <Gauge className="w-3 h-3 text-slate-500 ms-1.5 hidden sm:inline" />
                {[0.25, 0.5, 1.0, 1.5].map((speed) => (
                  <button
                    key={speed}
                    type="button"
                    onClick={() => handleSpeedChange(speed)}
                    className={`px-1.5 py-0.5 rounded text-[11px] font-mono transition-colors cursor-pointer ${
                      playbackSpeed === speed
                        ? "bg-tv-cyan/20 text-tv-cyan font-bold"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {speed}x
                  </button>
                ))}
              </div>

              <button
                type="button"
                onClick={toggleMute}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.06] transition-colors cursor-pointer"
                title={isMuted ? t("wr.unmute") : t("wr.mute")}
                aria-label={isMuted ? t("wr.unmute") : t("wr.mute")}
              >
                {isMuted ? (
                  <VolumeX className="w-4 h-4 text-tv-coral" />
                ) : (
                  <Volume2 className="w-4 h-4 text-slate-300" />
                )}
              </button>

              <button
                type="button"
                onClick={toggleFullscreen}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/[0.06] transition-colors cursor-pointer"
                title={t("wr.fullscreen")}
                aria-label={t("wr.fullscreen")}
              >
                <Maximize className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }
);
