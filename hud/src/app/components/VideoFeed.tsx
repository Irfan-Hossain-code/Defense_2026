import { useEffect, useState } from 'react';

const FRAME_BASE =
  (import.meta as unknown as { env?: { VITE_FRAME_URL?: string } }).env?.VITE_FRAME_URL ??
  '/frame.jpg';

/** Polls latest JPEG — works in Safari/Chrome (MJPEG in <img> often fails on Safari). */
export function VideoFeed() {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = () => {
      if (!alive) return;
      setSrc(`${FRAME_BASE}?t=${Date.now()}`);
    };
    tick();
    const id = window.setInterval(tick, 33);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  if (!src) {
    return <div className="absolute inset-0 bg-zinc-900 z-0" />;
  }

  return (
    <img
      src={src}
      alt="Camera feed"
      className="absolute inset-0 w-full h-full object-cover z-0 bg-zinc-900"
      draggable={false}
    />
  );
}
