"use client";

import React, { type ReactNode } from "react";

interface BidiNumProps {
  children: ReactNode;
  className?: string;
  title?: string;
}

/**
 * Strictly isolates numbers, ratios, resolutions (1920×1080), units (37.3 km/h),
 * and IDs (#12) in LTR direction so Arabic RTL prose never flips digits or symbols.
 */
export function BidiNum({ children, className = "", title }: BidiNumProps) {
  return (
    <span
      dir="ltr"
      title={title}
      className={`bidi-num font-mono tabular-nums ${className}`}
    >
      {children}
    </span>
  );
}
