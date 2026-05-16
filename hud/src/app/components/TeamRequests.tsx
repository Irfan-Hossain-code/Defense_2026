interface TeamRequest {
  id: number;
  team: string;
  message: string;
  priority: 'high' | 'medium' | 'low';
}

interface TeamRequestsProps {
  requests: TeamRequest[];
}

export function TeamRequests({ requests }: TeamRequestsProps) {
  return (
    <div className="absolute bottom-3 left-3 z-10 space-y-1 pointer-events-auto">
      {requests.map((request) => (
        <div
          key={request.id}
          className="backdrop-blur-sm bg-zinc-900/30 border border-white/12 rounded px-2 py-1 max-w-[200px]"
        >
          <p className="text-[9px] tracking-wider text-white/80 font-medium leading-none mb-0.5">
            {request.team}
          </p>
          <p className="text-[9px] text-white/65 leading-tight">{request.message}</p>
        </div>
      ))}
    </div>
  );
}
