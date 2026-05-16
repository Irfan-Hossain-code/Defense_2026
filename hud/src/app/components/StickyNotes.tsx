interface Note {
  id: number;
  title: string;
  details: string[];
  priority: 'high' | 'medium' | 'low';
}

interface StickyNotesProps {
  notes: Note[];
}

export function StickyNotes({ notes }: StickyNotesProps) {
  return (
    <div className="absolute top-4 right-4 z-10 pointer-events-auto">
      {notes.map((note) => (
        <div
          key={note.id}
          className="backdrop-blur-sm bg-zinc-900/30 border border-white/12 rounded px-2.5 py-2 min-w-[250px] max-w-[280px]"
        >
          <div className="text-[10px] tracking-[0.18em] text-white/80 font-medium mb-1 uppercase">
            {note.title}
          </div>
          <ul className="space-y-0.5">
            {note.details.map((detail, index) => (
              <li key={index} className="text-[10px] text-white/75 leading-tight">
                {detail}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
