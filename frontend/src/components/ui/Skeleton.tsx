import React from "react";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = "" }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse rounded-lg bg-white/5 border border-white/5 ${className}`}
    />
  );
}

export function KPIStripSkeleton() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      {[1, 2, 3, 4].map((i) => (
        <div
          key={i}
          className="p-5 rounded-xl border border-[rgba(148,163,184,0.12)] bg-tv-surface relative overflow-hidden"
        >
          <div className="flex items-center justify-between mb-3">
            <Skeleton className="h-4 w-28" />
            <Skeleton className="h-8 w-8 rounded-lg" />
          </div>
          <Skeleton className="h-8 w-24 mb-2" />
          <Skeleton className="h-3 w-36" />
        </div>
      ))}
    </div>
  );
}

export function MatchGridSkeleton() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="rounded-2xl border border-[rgba(148,163,184,0.12)] bg-tv-surface overflow-hidden p-5"
        >
          <Skeleton className="aspect-video w-full rounded-xl mb-4" />
          <Skeleton className="h-6 w-3/4 mb-2" />
          <Skeleton className="h-4 w-1/2 mb-4" />
          <Skeleton className="h-3 w-full rounded-full mb-4" />
          <div className="grid grid-cols-3 gap-3">
            <Skeleton className="h-10 rounded-lg" />
            <Skeleton className="h-10 rounded-lg" />
            <Skeleton className="h-10 rounded-lg" />
          </div>
        </div>
      ))}
    </div>
  );
}
