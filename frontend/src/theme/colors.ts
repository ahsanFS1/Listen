// Listen — dark neon palette (matches the provided UI mockups)
export const colors = {
  bg: "#0B1020",
  bgElevated: "#111733",
  bgCard: "#151B36",
  bgSoft: "#1C2447",
  border: "#2A3358",
  borderSoft: "#1F2748",
  text: "#E6EAF7",
  textDim: "#8F9BC5",
  textMuted: "#5E6A95",
  accent: "#5EE6FF",
  accentSoft: "#3EC7E0",
  accentDeep: "#1E88A8",
  brandPurple: "#8A5CF6",
  brandPink: "#E06CF0",
  ok: "#4EE6A8",
  warn: "#FFC061",
  err: "#FF6B8A",
} as const;

export type Colors = typeof colors;
