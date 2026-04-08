export type Audience = "designer" | "developer";

const STORAGE_KEY = "helmlab-audience";
const DEFAULT_AUDIENCE: Audience = "developer";

/**
 * Read the current audience preference from localStorage.
 * Falls back to "developer" if nothing is stored or value is invalid.
 */
export function getAudience(): Audience {
  if (typeof window === "undefined") return DEFAULT_AUDIENCE;
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "designer" || stored === "developer") return stored;
  return DEFAULT_AUDIENCE;
}

/**
 * Persist audience preference and update the html element class.
 */
export function setAudience(audience: Audience): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, audience);
  document.documentElement.classList.remove("audience-designer", "audience-developer");
  document.documentElement.classList.add(`audience-${audience}`);
}
