import { useState } from "react";

type Audience = "designer" | "developer";

function getInitial(): Audience {
  if (typeof window === "undefined") return "developer";
  const s = localStorage.getItem("helmlab-audience");
  return s === "designer" ? "designer" : "developer";
}

export default function AudienceToggle() {
  const [audience, setAudience] = useState<Audience>(getInitial);

  function toggle() {
    const next: Audience = audience === "developer" ? "designer" : "developer";
    setAudience(next);
    const html = document.documentElement;
    html.classList.remove("audience-designer", "audience-developer");
    html.classList.add(`audience-${next}`);
    localStorage.setItem("helmlab-audience", next);
  }

  return (
    <button
      onClick={toggle}
      className="flex items-center rounded-full border border-white/[0.08] bg-white/[0.03] p-0.5 text-xs transition-colors"
      aria-label={`Switch to ${audience === "developer" ? "designer" : "developer"} view`}
    >
      <span
        className={`rounded-full px-2.5 py-1 transition-all duration-200 ${
          audience === "designer"
            ? "bg-[#f97316] text-white font-medium"
            : "text-[#52525b]"
        }`}
      >
        Designer
      </span>
      <span
        className={`rounded-full px-2.5 py-1 transition-all duration-200 ${
          audience === "developer"
            ? "bg-[#60a5fa] text-white font-medium"
            : "text-[#52525b]"
        }`}
      >
        Developer
      </span>
    </button>
  );
}
