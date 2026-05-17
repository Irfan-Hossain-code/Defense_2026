interface TeamRequest {
  id: number;
  team: string;
  message: string;
  priority: 'high' | 'medium' | 'low';
}

interface TeamRequestsProps {
  requests: TeamRequest[];
}

const priorityColor = {
  high:   'text-red-400',
  medium: 'text-yellow-400',
  low:    'text-white/70',
};

export function TeamRequests({ requests }: TeamRequestsProps) {
  return (
    <div className="absolute bottom-5 left-5 z-10 space-y-2 pointer-events-auto">
      {requests.map((request) => (
        <div
          key={request.id}
          className="backdrop-blur-sm bg-zinc-900/40 border border-white/15 rounded-lg px-4 py-2.5 max-w-[320px] shadow-md"
        >
          <p className={`text-sm tracking-wider font-semibold leading-none mb-1.5 ${priorityColor[request.priority]}`}>
            {request.team}
          </p>
          <p className="text-sm text-white/75 leading-snug">{request.message}</p>
        </div>
      ))}
    </div>
  );
}
