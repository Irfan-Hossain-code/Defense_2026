interface AgentSubtitlesProps {
  text: string;
}

export function AgentSubtitles({ text }: AgentSubtitlesProps) {
  const line = text?.trim();
  if (!line) {
    return null;
  }

  return (
    <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-10 pointer-events-none w-full max-w-4xl px-6">
      <div className="backdrop-blur-sm bg-zinc-900/30 border border-white/12 rounded-sm px-3 py-1 text-center">
        <p className="text-[11px] text-white/90 tracking-wide font-light leading-tight">{line}</p>
      </div>
    </div>
  );
}
