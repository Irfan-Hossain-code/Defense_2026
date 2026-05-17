import { motion } from 'motion/react';
import { CameraStatus } from './components/CameraStatus';
import { AgentSubtitles } from './components/AgentSubtitles';
import { StickyNotes } from './components/StickyNotes';
import { TeamRequests } from './components/TeamRequests';
import { useHudSocket } from '../hooks/useHudSocket';
import { VideoFeed } from './components/VideoFeed';

export default function App() {
  const { data, connected } = useHudSocket();

  const stickyNotes = [
    {
      id: 1,
      title: data.sticky.title,
      details: data.sticky.details,
      priority: data.sticky.priority,
    },
  ];

  return (
    <motion.div className="fixed inset-0 bg-black text-green-400 font-mono overflow-hidden">
      <VideoFeed />

      <motion.div className="absolute inset-0 z-10 pointer-events-none">
        <div
          className={`absolute top-4 left-1/2 -translate-x-1/2 text-sm tracking-[0.25em] z-20 px-4 py-1 rounded-md backdrop-blur-sm border font-medium ${
            connected
              ? 'text-white/90 bg-zinc-900/30 border-white/20'
              : 'text-white/50 bg-zinc-900/25 border-white/10'
          }`}
        >
          {connected ? 'LINK ACTIVE' : 'LINK DOWN'}
        </div>

        <CameraStatus cameras={data.sensors} />
        <StickyNotes notes={stickyNotes} />
        <TeamRequests requests={data.team_requests} />
        <AgentSubtitles text={data.subtitle} />
      </motion.div>

      <style>{`
        html, body, #root { margin: 0; height: 100%; background: #000; }
      `}</style>
    </motion.div>
  );
}
