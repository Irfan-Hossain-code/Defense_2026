import { useEffect, useState } from 'react';

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
    { name: 'RGB', status: 'active', signal: 95, selected: true },
    { name: 'RF', status: 'standby', signal: 0, selected: false },
    { name: 'DEPTH', status: 'offline', signal: 0, selected: false },
    { name: 'INFRA', status: 'offline', signal: 0, selected: false },
  ],
  team_requests: [],
};

export function useHudSocket() {
  const [data, setData] = useState<HudPayload>(DEFAULT);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout>;

    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        retry = setTimeout(connect, 2000);
      };
      ws.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data as string) as HudPayload;
          setData(parsed);
        } catch {
          /* ignore */
        }
      };
    };

    connect();
    return () => {
      clearTimeout(retry);
      ws?.close();
    };
  }, []);

  return { data, connected };
}
