"""Safe check — never import face_recognition unless models load (else it calls quit())."""


def dlib_face_recognition_ready() -> bool:
    try:
        import face_recognition_models  # noqa: F401
        import face_recognition  # noqa: F401
        return True
    except Exception:
        return False
