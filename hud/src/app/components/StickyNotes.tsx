interface Note {
  id: number;
  title: string;
  details: string[];
  priority: 'high' | 'medium' | 'low';
}

interface StickyNotesProps {
  notes: Note[];
}

const priorityBorder = {
  high:   'border-red-400/50',
  medium: 'border-yellow-400/40',
  low:    'border-white/15',
};

export function StickyNotes({ notes }: StickyNotesProps) {
  return (
    <div className="absolute top-5 right-5 z-10 pointer-events-auto space-y-2">
      {notes.map((note) => (
        <div
          key={note.id}
          className={`backdrop-blur-sm bg-zinc-900/40 border rounded-lg px-4 py-3 min-w-[320px] max-w-[400px] shadow-lg ${priorityBorder[note.priority]}`}
        >
          <div className="text-sm tracking-[0.15em] text-white/90 font-semibold mb-2 uppercase">
            {note.title}
          </div>
          <ul className="space-y-1">
            {note.details.map((detail, index) => (
              <li key={index} className="text-sm text-white/80 leading-snug">
                {detail}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
