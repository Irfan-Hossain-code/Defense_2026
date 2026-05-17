import { useEffect, useRef, useState } from 'react';

interface AgentSubtitlesProps {
  text: string;
}

// How long (ms) to keep the subtitle visible after it goes empty.
// Prevents rapid appear/disappear flickering between lines.
const HOLD_MS = 5000;

export function AgentSubtitles({ text }: AgentSubtitlesProps) {
  const [displayed, setDisplayed] = useState(text ?? '');
  const [visible,   setVisible]   = useState(false);
  const holdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const line = text?.trim() ?? '';

    if (line) {
      // New content — show immediately, cancel any pending hide
      if (holdTimer.current) clearTimeout(holdTimer.current);
      setDisplayed(line);
      setVisible(true);
    } else if (visible) {
      // Content cleared — hold for HOLD_MS then fade out
      holdTimer.current = setTimeout(() => setVisible(false), HOLD_MS);
    }

    return () => {
      if (holdTimer.current) clearTimeout(holdTimer.current);
    };
  }, [text]);

  return (
    <div
      className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 pointer-events-none w-full max-w-5xl px-8"
      style={{
        opacity:    visible ? 1 : 0,
        transition: 'opacity 0.6s ease',
      }}
    >
      <div className="backdrop-blur-sm bg-zinc-900/50 border border-white/20 rounded-lg px-6 py-3 text-center shadow-lg">
        <p className="text-xl text-white/95 tracking-wide font-light leading-snug">
          {displayed}
        </p>
      </div>
    </div>
  );
}
