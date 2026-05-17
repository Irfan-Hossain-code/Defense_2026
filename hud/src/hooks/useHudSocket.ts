import { useEffect, useRef, useState } from 'react';

export interface HudPayload {
  subtitle: string;
  sticky: {
    title: string;
    details: string[];
    priority: 'high' | 'medium' | 'low';
  };
  sensors: Array<{
    name: string;
    status: 'active' | 'standby' | 'warning' | 'offline';
    signal: number;
    selected?: boolean;
  }>;
  active_sensor?: string;
  team_requests: Array<{
    team: string;
    message: string;
    priority: 'high' | 'medium' | 'low';
  }>;
  dossier?: {
    active: boolean;
    suspect: string;
    probability_peak: number;
    description: string;
  };
}

const WS_URL =
  (import.meta as unknown as { env?: { VITE_WS_URL?: string } }).env?.VITE_WS_URL ??
  'ws://127.0.0.1:8765';

const DEFAULT: HudPayload = {
  subtitle: '',
  sticky: {
    title: 'TARGET UNKNOWN',
    details: ['Awaiting contact', 'No optical profile', 'Stand by'],
    priority: 'medium',
  },
  sensors: [
    { name: 'RGB',   status: 'active',  signal: 95, selected: true  },
    { name: 'RF',    status: 'standby', signal: 0,  selected: false },
    { name: 'DEPTH', status: 'offline', signal: 0,  selected: false },
    { name: 'INFRA', status: 'offline', signal: 0,  selected: false },
  ],
  team_requests: [],
};

// How often the HUD state actually updates (ms). 200 ms = max 5 repaints/sec.
// Prevents constant flickering when Python broadcasts every video frame.
const THROTTLE_MS = 200;

export function useHudSocket() {
  const [data,      setData]      = useState<HudPayload>(DEFAULT);
  const [connected, setConnected] = useState(false);

  // Throttle: hold latest payload and flush on interval
  const pendingRef  = useRef<HudPayload | null>(null);
  const timerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout>;

    const flush = () => {
      if (pendingRef.current) {
        setData(pendingRef.current);
        pendingRef.current = null;
      }
      timerRef.current = null;
    };

    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        retry = setTimeout(connect, 2000);
      };
      ws.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data as string) as HudPayload;
          pendingRef.current = parsed;
          // Schedule a flush if one isn't already pending
          if (!timerRef.current) {
            timerRef.current = setTimeout(flush, THROTTLE_MS);
          }
        } catch { /* ignore malformed frames */ }
      };
    };

    connect();
    return () => {
      clearTimeout(retry);
      if (timerRef.current) clearTimeout(timerRef.current);
      ws?.close();
    };
  }, []);

  return { data, connected };
}
