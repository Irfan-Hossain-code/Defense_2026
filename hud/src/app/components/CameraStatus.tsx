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
    <div className="absolute top-5 left-5 z-10 pointer-events-auto">
      <div className="flex items-center gap-1 backdrop-blur-sm bg-zinc-900/30 border border-white/15 rounded-md px-2 py-1">
        {cameras.map((camera) => {
          const Icon = iconMap[camera.name as keyof typeof iconMap] || Camera;
          const isRgb = camera.name === 'RGB';
          const isRf  = camera.name === 'RF';
          const isOn  = isRgb || (isRf && camera.status === 'active');

          return (
            <div
              key={camera.name}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md transition-all border ${
                isOn
                  ? 'bg-white/20 text-white border-white/35'
                  : 'text-white/30 border-transparent'
              }`}
              title={
                isRgb ? 'RGB camera online'
                : isRf  ? isOn ? 'RF tracking active' : 'RF standby'
                : `${camera.name} offline`
              }
            >
              <Icon className={`w-5 h-5 ${isOn ? '' : 'opacity-35'}`} />
              <span className="text-sm tracking-widest font-medium">{camera.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
