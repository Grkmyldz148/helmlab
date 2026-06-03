/* global React */
const { useState, useEffect, useRef } = React;

// ============================================================
// Color.js logo component (UCS chromaticity tongue + RGB triangle)
// Compact version: tongue path + 3 primary dots
// ============================================================
const TONGUE_PATH = "M 256.804 -16.594 L 256.637 -16.593 L 256.418 -16.266 L 256.085 -16.263 L 255.699 -15.935 L 255.2 -15.932 L 254.535 -15.927 L 253.704 -15.921 L 252.207 -16.885 L 249.615 -19.128 L 246.12 -22.623 L 241.063 -27.948 L 234.69 -35.011 L 226.63 -43.68 L 216.06 -54.946 L 203.34 -68.779 L 187.72 -87.114 L 168.963 -111.946 L 144.076 -150.984 L 114.692 -204.374 L 82.816 -270.829 L 52.134 -342.686 L 28.159 -411.679 L 11.891 -469.843 L 3.473 -513.066 L 1.438 -543.162 L 4.643 -563.843 L 5.859 -567.036 L 7.213 -569.939 L 8.771 -572.575 L 10.438 -574.905 L 12.285 -576.969 L 14.22 -578.754 L 16.315 -580.309 L 18.482 -581.618 L 20.756 -582.743 L 23.116 -583.667 L 25.566 -584.422 L 28.05 -585.009 L 30.633 -585.478 L 33.285 -585.851 L 36.008 -586.142 L 50.059 -586.754 L 64.335 -586.523 L 79.224 -585.616 L 95.269 -584.113 L 112.701 -582.069 L 131.858 -579.553 L 153.101 -576.592 L 176.586 -573.182 L 202.596 -569.364 L 231.17 -565.107 L 262.343 -560.441 L 295.958 -555.411 L 331.527 -550.115 L 368.06 -544.626 L 403.475 -539.334 L 437.943 -534.187 L 469.171 -529.561 L 496.768 -525.419 L 520.257 -521.894 L 539.92 -518.977 L 556.503 -516.489 L 570.88 -514.332 L 583.036 -512.508 L 592.866 -511.07 L 600.496 -509.926 L 606.437 -509.035 L 610.879 -508.368 L 613.777 -507.933 L 616.162 -507.576 L 618.025 -507.296 L 619.897 -507.015 L 621.507 -506.774 L 622.584 -506.612 L 623.123 -506.531 L 623.393 -506.491 Z";

function ColorJsMark({ size = 24, mono = false, color = "currentColor", showTriangle = true, animate = false }) {
  const id = useRef(`cjs-${Math.random().toString(36).slice(2, 8)}`).current;
  return (
    <svg
      width={size}
      height={size * (615 / 630)}
      viewBox="0 -625 630 615"
      style={{ flexShrink: 0, overflow: "visible" }}
    >
      <defs>
        <linearGradient id={`${id}-tongue`} x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stopColor="oklch(0.75 0.18 250)" />
          <stop offset="50%" stopColor="oklch(0.78 0.2 140)" />
          <stop offset="100%" stopColor="oklch(0.7 0.22 30)" />
        </linearGradient>
      </defs>
      <path
        d={TONGUE_PATH}
        fill={mono ? color : `url(#${id}-tongue)`}
        opacity={mono ? 0.9 : 1}
      />
      {showTriangle && (
        <g style={{ transformOrigin: "315px -340px", animation: animate ? "cjs-spin 8s linear infinite" : "none" }}>
          <polygon
            points="496.35,-525.54 98.6,-577.67 175.44,-157.89"
            fill="none"
            stroke={mono ? color : "rgba(255,255,255,0.85)"}
            strokeWidth="14"
            strokeLinejoin="round"
          />
          <circle r="44" cx="496.35" cy="-525.54" fill={mono ? color : "hsl(0, 80%, 55%)"} />
          <circle r="44" cx="98.6" cy="-577.67" fill={mono ? color : "hsl(120, 55%, 50%)"} />
          <circle r="44" cx="175.44" cy="-157.89" fill={mono ? color : "hsl(220, 85%, 60%)"} />
        </g>
      )}
    </svg>
  );
}

const COLORJS_URL = "https://colorjs.io/docs/spaces.html";

// ============================================================
// Common badge wrapper — handles link + theme
// ============================================================
function BadgeLink({ href = COLORJS_URL, children, style }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener"
      style={{
        textDecoration: "none",
        color: "inherit",
        display: "inline-flex",
        ...style,
      }}
    >
      {children}
    </a>
  );
}

// ============================================================
// VARIANT 1 — Minimal pill (shields.io vibes, custom)
// ============================================================
function MinimalPill({ theme = "light", text = { kicker: "available in", brand: "Color.js" } }) {
  const dark = theme === "dark";
  return (
    <BadgeLink>
      <span
        style={{
          fontFamily: "'Inter', system-ui, sans-serif",
          fontSize: 13,
          fontWeight: 500,
          display: "inline-flex",
          alignItems: "center",
          height: 32,
          borderRadius: 999,
          overflow: "hidden",
          whiteSpace: "nowrap",
          border: dark ? "1px solid #2a2a2e" : "1px solid #e4e4e7",
          background: dark ? "#0e0e10" : "#ffffff",
          color: dark ? "#e4e4e7" : "#27272a",
          boxShadow: dark ? "0 1px 0 rgba(255,255,255,0.04) inset" : "0 1px 2px rgba(0,0,0,0.04)",
        }}
      >
        <span
          style={{
            padding: "0 12px",
            height: "100%",
            display: "inline-flex",
            alignItems: "center",
            background: dark ? "#18181b" : "#fafafa",
            color: dark ? "#a1a1aa" : "#71717a",
            borderRight: dark ? "1px solid #2a2a2e" : "1px solid #e4e4e7",
            fontSize: 12,
            whiteSpace: "nowrap",
            flexShrink: 0,
          }}
        >
          {text.kicker}
        </span>
        <span style={{ padding: "0 12px", display: "inline-flex", alignItems: "center", gap: 6, whiteSpace: "nowrap", flexShrink: 0 }}>
          <ColorJsMark size={16} />
          <span style={{ fontWeight: 600, letterSpacing: "-0.01em" }}>{text.brand}</span>
        </span>
      </span>
    </BadgeLink>
  );
}

// ============================================================
// VARIANT 2 — Terminal / monospace
// ============================================================
function TerminalBadge({ theme = "light", text = { line1: "import \"helmlab\"", line2: "// available in color.js" } }) {
  const dark = theme === "dark";
  return (
    <BadgeLink>
      <span
        style={{
          fontFamily: "'JetBrains Mono', ui-monospace, monospace",
          fontSize: 12,
          display: "inline-flex",
          flexDirection: "column",
          padding: "10px 14px",
          borderRadius: 8,
          background: dark ? "#0a0a0c" : "#f4f4f5",
          border: dark ? "1px solid #1f1f23" : "1px solid #e4e4e7",
          color: dark ? "#d4d4d8" : "#27272a",
          lineHeight: 1.5,
          minWidth: 240,
        }}
      >
        <span style={{ display: "inline-flex", gap: 6, alignItems: "center", marginBottom: 4 }}>
          <span style={{ width: 8, height: 8, borderRadius: 4, background: "#ef4444" }}></span>
          <span style={{ width: 8, height: 8, borderRadius: 4, background: "#eab308" }}></span>
          <span style={{ width: 8, height: 8, borderRadius: 4, background: "#22c55e" }}></span>
          <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.5 }}>colorjs.io</span>
        </span>
        <span>
          <span style={{ color: dark ? "#a78bfa" : "#7c3aed" }}>const</span>{" "}
          <span style={{ color: dark ? "#fbbf24" : "#b45309" }}>space</span>{" "}={" "}
          <span style={{ color: dark ? "#34d399" : "#047857" }}>"helmlab"</span>;
        </span>
        <span style={{ opacity: 0.55, marginTop: 2 }}>{text.line2}</span>
      </span>
    </BadgeLink>
  );
}

// ============================================================
// VARIANT 3 — Logo-forward
// ============================================================
function LogoForward({ theme = "light", text = { eyebrow: "shipped in", brand: "Color.js" } }) {
  const dark = theme === "dark";
  return (
    <BadgeLink>
      <span
        style={{
          fontFamily: "'Inter', system-ui, sans-serif",
          display: "inline-flex",
          alignItems: "center",
          gap: 12,
          padding: "10px 16px 10px 12px",
          borderRadius: 12,
          background: dark ? "linear-gradient(135deg, #0c0c0f 0%, #1a1a20 100%)" : "linear-gradient(135deg, #ffffff 0%, #f8f8fa 100%)",
          border: dark ? "1px solid #2a2a2e" : "1px solid #e4e4e7",
          boxShadow: dark ? "0 4px 12px rgba(0,0,0,0.4)" : "0 4px 12px rgba(0,0,0,0.06)",
          color: dark ? "#fafafa" : "#0a0a0c",
        }}
      >
        <span
          style={{
            width: 38,
            height: 38,
            borderRadius: 10,
            background: dark ? "#18181b" : "#0a0a0c",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <ColorJsMark size={26} />
        </span>
        <span style={{ display: "inline-flex", flexDirection: "column", lineHeight: 1.2 }}>
          <span style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.12em", opacity: 0.55, fontWeight: 500 }}>
            {text.eyebrow}
          </span>
          <span style={{ fontSize: 16, fontWeight: 600, letterSpacing: "-0.02em" }}>{text.brand}</span>
        </span>
      </span>
    </BadgeLink>
  );
}

// ============================================================
// VARIANT 4 — Hero / proud
// ============================================================
function HeroBadge({ theme = "light", text = { kicker: "helmlab is now", brand: "in Color.js", sub: "A new perceptual color space, available everywhere Color.js runs." } }) {
  const dark = theme === "dark";
  return (
    <BadgeLink style={{ width: "100%" }}>
      <span
        style={{
          fontFamily: "'Inter', system-ui, sans-serif",
          display: "flex",
          flexDirection: "column",
          padding: "28px 32px",
          borderRadius: 20,
          background: dark
            ? "radial-gradient(120% 100% at 0% 0%, oklch(0.32 0.08 280 / 0.5) 0%, #08080a 60%)"
            : "radial-gradient(120% 100% at 0% 0%, oklch(0.93 0.05 270) 0%, #ffffff 60%)",
          border: dark ? "1px solid #25252a" : "1px solid #e4e4e7",
          color: dark ? "#fafafa" : "#0a0a0c",
          width: 480,
          position: "relative",
          overflow: "hidden",
        }}
      >
        <span style={{ position: "absolute", top: -40, right: -40, opacity: dark ? 0.18 : 0.12 }}>
          <ColorJsMark size={220} />
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 12, opacity: 0.7, fontWeight: 500, letterSpacing: "0.02em" }}>
          <span style={{ width: 6, height: 6, borderRadius: 3, background: "oklch(0.7 0.22 30)" }}></span>
          {text.kicker}
        </span>
        <span style={{ fontSize: 36, fontWeight: 700, letterSpacing: "-0.03em", marginTop: 8, lineHeight: 1.05 }}>
          {text.brand}
        </span>
        <span style={{ fontSize: 14, opacity: 0.65, marginTop: 12, lineHeight: 1.5, maxWidth: 340 }}>
          {text.sub}
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6, marginTop: 20, fontSize: 13, fontWeight: 500, color: dark ? "oklch(0.85 0.12 270)" : "oklch(0.5 0.18 270)" }}>
          Read the docs
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M3 7h8M7.5 3.5L11 7l-3.5 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </span>
    </BadgeLink>
  );
}

// ============================================================
// VARIANT 5 — Animated chromaticity (gradient sweep)
// ============================================================
function AnimatedBadge({ theme = "light", text = { brand: "color.js", suffix: "supported" } }) {
  const dark = theme === "dark";
  return (
    <BadgeLink>
      <span
        className="cjs-animated"
        style={{
          fontFamily: "'Inter', system-ui, sans-serif",
          fontSize: 13,
          display: "inline-flex",
          alignItems: "center",
          gap: 10,
          padding: "8px 16px 8px 10px",
          borderRadius: 999,
          background: dark ? "#0a0a0c" : "#ffffff",
          border: "1px solid transparent",
          color: dark ? "#fafafa" : "#0a0a0c",
          position: "relative",
          backgroundImage: dark
            ? `linear-gradient(#0a0a0c, #0a0a0c), conic-gradient(from var(--cjs-angle, 0deg), oklch(0.7 0.22 30), oklch(0.78 0.2 140), oklch(0.75 0.22 250), oklch(0.7 0.22 320), oklch(0.7 0.22 30))`
            : `linear-gradient(#ffffff, #ffffff), conic-gradient(from var(--cjs-angle, 0deg), oklch(0.7 0.22 30), oklch(0.78 0.2 140), oklch(0.75 0.22 250), oklch(0.7 0.22 320), oklch(0.7 0.22 30))`,
          backgroundOrigin: "border-box",
          backgroundClip: "padding-box, border-box",
          fontWeight: 500,
        }}
      >
        <ColorJsMark size={20} animate />
        <span>
          <span style={{ fontWeight: 600 }}>{text.brand}</span>
          <span style={{ opacity: 0.55, marginLeft: 4 }}>{text.suffix}</span>
        </span>
      </span>
    </BadgeLink>
  );
}

// ============================================================
// VARIANT 6 — Shields.io style (README/GitHub)
// ============================================================
function ShieldsBadge({ theme = "light", text = { left: "color.js", right: "supported" } }) {
  return (
    <BadgeLink>
      <span
        style={{
          fontFamily: "'DejaVu Sans', Verdana, Geneva, sans-serif",
          fontSize: 11,
          display: "inline-flex",
          alignItems: "center",
          height: 20,
          borderRadius: 3,
          overflow: "hidden",
          boxShadow: "0 1px 0 rgba(0,0,0,0.05)",
          fontWeight: 400,
        }}
      >
        <span
          style={{
            background: "#555",
            color: "#fff",
            padding: "0 8px",
            height: "100%",
            display: "inline-flex",
            alignItems: "center",
            gap: 5,
            textShadow: "0 1px 0 rgba(0,0,0,0.25)",
          }}
        >
          <ColorJsMark size={12} />
          {text.left}
        </span>
        <span
          style={{
            background: "linear-gradient(180deg, oklch(0.65 0.2 280), oklch(0.58 0.22 280))",
            color: "#fff",
            padding: "0 8px",
            height: "100%",
            display: "inline-flex",
            alignItems: "center",
            textShadow: "0 1px 0 rgba(0,0,0,0.25)",
          }}
        >
          {text.right}
        </span>
      </span>
    </BadgeLink>
  );
}

// ============================================================
// VARIANT 7 — Inline footer mention
// ============================================================
function InlineFooter({ theme = "light", text = { prefix: "Color spaces by", brand: "Color.js" } }) {
  const dark = theme === "dark";
  return (
    <span
      style={{
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSize: 13,
        color: dark ? "#a1a1aa" : "#71717a",
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
      }}
    >
      {text.prefix}
      <BadgeLink
        style={{
          alignItems: "center",
          gap: 5,
          color: dark ? "#fafafa" : "#0a0a0c",
          fontWeight: 500,
          borderBottom: dark ? "1px solid #3f3f46" : "1px solid #d4d4d8",
          paddingBottom: 1,
        }}
      >
        <ColorJsMark size={14} />
        {text.brand}
      </BadgeLink>
    </span>
  );
}

// ============================================================
// VARIANT 8 — Floating sticky (corner)
// ============================================================
function FloatingSticky({ theme = "light", text = { tag: "New", line1: "helmlab is now in", brand: "Color.js" } }) {
  const dark = theme === "dark";
  const [dismissed, setDismissed] = useState(false);
  if (dismissed) {
    return (
      <button
        onClick={() => setDismissed(false)}
        style={{
          fontFamily: "'Inter', system-ui, sans-serif",
          fontSize: 12,
          padding: "6px 10px",
          borderRadius: 999,
          border: "1px solid",
          borderColor: dark ? "#2a2a2e" : "#e4e4e7",
          background: dark ? "#0a0a0c" : "#ffffff",
          color: dark ? "#a1a1aa" : "#71717a",
          cursor: "pointer",
        }}
      >
        Show badge
      </button>
    );
  }
  return (
    <span
      style={{
        fontFamily: "'Inter', system-ui, sans-serif",
        display: "inline-flex",
        alignItems: "center",
        gap: 12,
        padding: "12px 14px 12px 12px",
        borderRadius: 14,
        background: dark ? "#0e0e12" : "#ffffff",
        border: dark ? "1px solid #25252a" : "1px solid #e4e4e7",
        boxShadow: dark
          ? "0 12px 32px rgba(0,0,0,0.6), 0 2px 0 rgba(255,255,255,0.03) inset"
          : "0 12px 32px rgba(15,15,30,0.10), 0 1px 2px rgba(0,0,0,0.04)",
        color: dark ? "#fafafa" : "#0a0a0c",
        position: "relative",
        minWidth: 280,
      }}
    >
      <span
        style={{
          width: 40,
          height: 40,
          borderRadius: 10,
          background: dark ? "#18181b" : "#fafafa",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          border: dark ? "1px solid #25252a" : "1px solid #efefef",
        }}
      >
        <ColorJsMark size={26} />
      </span>
      <span style={{ display: "inline-flex", flexDirection: "column", lineHeight: 1.3, flex: 1 }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              fontSize: 9,
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              padding: "2px 6px",
              borderRadius: 4,
              background: "oklch(0.7 0.2 30)",
              color: "white",
            }}
          >
            {text.tag}
          </span>
          <span style={{ fontSize: 11, opacity: 0.6 }}>{text.line1}</span>
        </span>
        <BadgeLink style={{ fontSize: 15, fontWeight: 600, marginTop: 2, color: "inherit", letterSpacing: "-0.01em", alignItems: "center", gap: 4 }}>
          {text.brand}
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ opacity: 0.6 }}>
            <path d="M3 6h6M6 3l3 3-3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </BadgeLink>
      </span>
      <button
        onClick={() => setDismissed(true)}
        aria-label="Dismiss"
        style={{
          position: "absolute",
          top: 6,
          right: 6,
          width: 20,
          height: 20,
          borderRadius: 999,
          border: "none",
          background: "transparent",
          color: dark ? "#52525b" : "#a1a1aa",
          cursor: "pointer",
          fontSize: 14,
          lineHeight: 1,
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        ×
      </button>
    </span>
  );
}

// expose
Object.assign(window, {
  ColorJsMark,
  MinimalPill,
  TerminalBadge,
  LogoForward,
  HeroBadge,
  AnimatedBadge,
  ShieldsBadge,
  InlineFooter,
  FloatingSticky,
});
