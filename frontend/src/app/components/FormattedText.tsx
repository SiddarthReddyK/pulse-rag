import type { ReactNode } from "react";

/**
 * Renders the small subset of markdown the AI returns (bullet lines and
 * **bold**) as real HTML elements. It builds React nodes directly and never
 * uses dangerouslySetInnerHTML, so model output cannot inject markup.
 */

const BULLET = /^\s*(?:[-*•])\s+/;

function renderInline(text: string): ReactNode[] {
  return text
    .split(/(\*\*[^*]+\*\*)/g)
    .filter(Boolean)
    .map((part, i) =>
      part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
        <strong key={i} className="font-semibold">
          {part.slice(2, -2)}
        </strong>
      ) : (
        <span key={i}>{part}</span>
      ),
    );
}

export function FormattedText({ text }: { text: string }) {
  const lines = text.split("\n").filter((line) => line.trim() !== "");
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = () => {
    if (bullets.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="list-disc pl-5 space-y-2">
        {bullets.map((item, i) => (
          <li key={i}>{renderInline(item)}</li>
        ))}
      </ul>,
    );
    bullets = [];
  };

  for (const line of lines) {
    if (BULLET.test(line)) {
      bullets.push(line.replace(BULLET, ""));
    } else {
      flushBullets();
      blocks.push(<p key={`p-${blocks.length}`}>{renderInline(line)}</p>);
    }
  }
  flushBullets();

  return (
    <div className="text-[#0f172a] leading-relaxed space-y-3">{blocks}</div>
  );
}