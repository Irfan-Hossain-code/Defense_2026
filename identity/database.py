import os

_SUPPORTED_EXT = (".jpg", ".jpeg", ".png")


class FaceDatabase:
    """
    Loads face encodings once from a known_faces/ directory tree.

    Expected layout:
        known_faces/
            alice/
                alice1.jpg
            bob/
                bob_front.jpg
                bob_side.png
    """

    def __init__(self, known_faces_dir: str = "known_faces"):
        self.encodings: list = []
        self.names: list = []
        self._load(known_faces_dir)

    def _load(self, directory: str) -> None:
        if not os.path.isdir(directory):
            print(f"[identity] Directory '{directory}' not found — face ID disabled.")
            return

        try:
            import face_recognition
        except ImportError:
            print("[identity] face_recognition not installed — face ID disabled.")
            return

        count = 0
        for person_name in sorted(os.listdir(directory)):
            person_dir = os.path.join(directory, person_name)
            if not os.path.isdir(person_dir):
                continue
            for fname in sorted(os.listdir(person_dir)):
                if not fname.lower().endswith(_SUPPORTED_EXT):
                    continue
                fpath = os.path.join(person_dir, fname)
                try:
                    img = face_recognition.load_image_file(fpath)
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        self.encodings.append(encs[0])
                        self.names.append(person_name)
                        count += 1
                    else:
                        print(f"[identity] No face found in {fpath} — skipped.")
                except Exception as exc:
                    print(f"[identity] Failed to load {fpath}: {exc}")

        persons = len(set(self.names))
        print(f"[identity] Loaded {count} encoding(s) for {persons} person(s).")

    @property
    def is_empty(self) -> bool:
        return len(self.encodings) == 0
