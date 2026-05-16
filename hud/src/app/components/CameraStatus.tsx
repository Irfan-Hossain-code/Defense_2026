import { Camera, Radio, Eye, Zap } from 'lucide-react';

interface CameraStatusProps {
  cameras: Array<{
    name: string;
    status: 'active' | 'standby' | 'warning' | 'offline';
    signal: number;
    selected?: boolean;
  }>;
}

const iconMap = { RF: Radio, DEPTH: Eye, RGB: Camera, INFRA: Zap };

export function CameraStatus({ cameras }: CameraStatusProps) {
  return (
    <div className="absolute top-4 left-4 z-10 pointer-events-auto">
      <div className="flex items-center gap-0 backdrop-blur-sm bg-zinc-900/25 border border-white/12 rounded px-1 py-0.5">
        {cameras.map((camera) => {
          const Icon = iconMap[camera.name as keyof typeof iconMap] || Camera;
          const isRgb = camera.name === 'RGB';
          const isRf = camera.name === 'RF';
          const isOn = isRgb || (isRf && camera.status === 'active');

          return (
            <div
              key={camera.name}
              className={`flex items-center gap-1 px-2 py-0.5 rounded transition-all border ${
                isOn
                  ? 'bg-white/20 text-white border-white/35'
                  : 'text-white/25 border-transparent'
              }`}
              title={
                isRgb
                  ? 'RGB camera online'
                  : isRf
                    ? isOn
                      ? 'RF tracking active'
                      : 'RF standby'
                    : `${camera.name} offline`
              }
            >
              <Icon className={`w-3 h-3 ${isOn ? '' : 'opacity-35'}`} />
              <span className="text-[9px] tracking-widest">{camera.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
