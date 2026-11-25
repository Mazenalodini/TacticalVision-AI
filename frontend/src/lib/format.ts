import type { Lang } from "./i18n";

/**
 * Arabic & English grammatical pluralization and tactical formatting helpers.
 * Guarantees 100% natural Arabic phrasing and zero leaked English identifiers.
 */

export function formatMatchCount(n: number, lang: Lang): string {
  if (lang === "ar") {
    if (n === 0) return "0 مباريات";
    if (n === 1) return "مباراة واحدة";
    if (n === 2) return "مباراتان";
    if (n >= 3 && n <= 10) return `${n} مباريات`;
    return `${n} مباراة`;
  }
  return n === 1 ? "1 Match" : `${n} Matches`;
}

export function formatEventCount(n: number, lang: Lang): string {
  if (lang === "ar") {
    if (n === 0) return "0 أحداث";
    if (n === 1) return "حدث واحد";
    if (n === 2) return "حدثان";
    if (n >= 3 && n <= 10) return `${n} أحداث`;
    return `${n} حدثاً`;
  }
  return n === 1 ? "1 Event" : `${n} Events`;
}

export function formatPlayerCount(n: number, lang: Lang): string {
  if (lang === "ar") {
    if (n === 0) return "0 لاعبين";
    if (n === 1) return "لاعب واحد";
    if (n === 2) return "لاعبان";
    if (n >= 3 && n <= 10) return `${n} لاعبين`;
    return `${n} لاعباً`;
  }
  return n === 1 ? "1 Player" : `${n} Players`;
}

export function formatSprintCount(n: number, lang: Lang): string {
  if (lang === "ar") {
    if (n === 0) return "0 انطلاقات";
    if (n === 1) return "انطلاقة واحدة";
    if (n === 2) return "انطلاقتان";
    if (n >= 3 && n <= 10) return `${n} انطلاقات`;
    return `${n} انطلاقة`;
  }
  return n === 1 ? "1 Sprint" : `${n} Sprints`;
}

export function formatLinkCount(n: number, lang: Lang): string {
  if (lang === "ar") {
    if (n === 0) return "0 روابط";
    if (n === 1) return "رابط واحد";
    if (n === 2) return "رابطان";
    if (n >= 3 && n <= 10) return `${n} روابط`;
    return `${n} رابطاً`;
  }
  return n === 1 ? "1 Link" : `${n} Links`;
}

export function formatTeamName(team: number, lang: Lang): string {
  if (lang === "ar") {
    return team === 0 ? "الفريق أ" : "الفريق ب";
  }
  return team === 0 ? "Team A" : "Team B";
}

export function formatTeamCode(team: number, lang: Lang): string {
  if (lang === "ar") {
    return team === 0 ? "أ" : "ب";
  }
  return team === 0 ? "A" : "B";
}

export function formatRole(role: string, lang: Lang, short = false): string {
  const isGK = role === "goalkeeper" || role === "gk";
  if (lang === "ar") {
    if (isGK) return short ? "حارس" : "حارس مرمى";
    return short ? "ميدان" : "لاعب ميدان";
  }
  if (isGK) return short ? "GK" : "Goalkeeper";
  return short ? "Outfield" : "Outfield Player";
}

export function formatResolution(resolution?: number[] | [number, number]): string {
  if (resolution && resolution.length >= 2 && resolution[0] && resolution[1]) {
    return `${resolution[0]}×${resolution[1]}`;
  }
  return "1920×1080";
}

export function formatResolutionBadge(resolution?: number[] | [number, number]): string {
  if (resolution && resolution.length >= 2 && resolution[1]) {
    return `${resolution[1]}p`;
  }
  return "1080p";
}

export function isCoreDemoMatch(id: string): boolean {
  return (
    id === "match_01_counter" ||
    id === "match_02_buildup" ||
    id === "match_03_press" ||
    id === "match_03_pressing"
  );
}

/**
 * Resolves a human-friendly localized match title and scenario tag,
 * hiding raw internal `user_<timestamp>_<hash>` strings.
 */
export function resolveMatchDisplay(
  match: { id: string; title?: string; is_user_uploaded?: boolean },
  t: (key: string) => string,
  lang: Lang
): { title: string; tag: string; badge: string } {
  const normalizedId =
    match.id === "match_03_pressing" ? "match_03_press" : match.id;

  const titleKey = `match.${normalizedId}`;
  const tagKey = `match.${normalizedId}_tag`;

  const translatedTitle = t(titleKey);
  const translatedTag = t(tagKey);

  if (translatedTitle !== titleKey) {
    return {
      title: translatedTitle,
      tag: translatedTag !== tagKey ? translatedTag : "",
      badge: lang === "ar" ? "مباراة نموذجية" : "Demo Match",
    };
  }

  // User uploaded match: clean up fallback auto-generated IDs if needed
  const rawTitle = (match.title || "").trim();
  const looksLikeRawId =
    !rawTitle ||
    /^user[\s_]\d+/i.test(rawTitle) ||
    rawTitle.toLowerCase() === "user uploaded match";

  const cleanTitle = looksLikeRawId
    ? lang === "ar"
      ? "تحليل تكتيكي مخصص"
      : "Custom Tactical Analysis"
    : lang === "ar" && /^match\s*01\s*counter$/i.test(rawTitle)
    ? "مقطع الهجمة المرتدة المرفوع"
    : lang === "ar" && /^match\s*02\s*buildup$/i.test(rawTitle)
    ? "مقطع البناء المنظم المرفوع"
    : lang === "ar" && /^match\s*03\s*press(ing)?$/i.test(rawTitle)
    ? "مقطع الضغط العالي المرفوع"
    : rawTitle;

  const defaultTag =
    lang === "ar"
      ? "مقطع مرفوع ومُحلل عبر محرك الرؤية الحاسوبية المزدوج"
      : "User-uploaded footage analyzed by the dual-YOLOv11 pipeline";

  return {
    title: cleanTitle,
    tag: defaultTag,
    badge: lang === "ar" ? "تحليل مخصص" : "Custom Clip",
  };
}
