# Tactical AI + HUD Setup

## What was added

| Path | Role |
|------|------|
| `ai/` | JSON store, events, ConfidentialMind + Gemini, ElevenLabs TTS, PTT |
| `bridge/` | WebSocket server → React HUD |
| `hud/` | Figma tactical overlay (browser) |
| `data/` | `target_session.json` + PTT wav (gitignored) |

## Quick start (one machine)

### 1. Python

```bash
cd Defense_2026
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env — see checklist below
```

### 2. React HUD (second terminal)

```bash
cd hud
npm install
npm run dev
```

Open **http://localhost:5173** — fullscreen over the OpenCV window (same monitor).

### 3. Tracker + AI

```bash
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| **ESC** | Quit tracker |
| **T** | Push-to-talk: press once to **start** recording, again to **send** |

## Auto-narration (important events only)

- Target acquired (face ≥ 55%)
- Line of sight lost → RF
- Visual reacquired
- Optical profile on occlusion (vision model)
- Squad urgent (~every 55s mock: backup / under fire)

Otherwise the agent **waits** until you press **T**.

## Windows notes (for your teammate)

- Webcam index `0` is the same in `main.py`
- TTS uses `os.startfile()` for MP3 (default player)
- If `sounddevice` fails, install [PortAudio](https://www.portaudio.com/) or use WSL — or PTT falls back to silent
- `SpeechRecognition` needs internet for Google STT
- Serial RF: use `COM3` etc. when ESP32 is wired (existing port scripts in `port checking/`)

## macOS notes

- TTS uses `afplay`
- Camera permission: System Settings → Privacy → Camera

## `.env` checklist — **send these to your team lead**

Copy `.env.example` → `.env` and fill:

| Variable | Required for | Where to get |
|----------|----------------|--------------|
| `CONFIDENTIALMIND_BASE_URL` | Event lines + vision profile | Junction / ConfidentialMind handout |
| `CONFIDENTIALMIND_API_KEY` | Same | Same |
| `CM_MODEL_NARRATION` | e.g. `Gemma 4` | Run `list_models.py` on CM repo |
| `CM_MODEL_VISION` | e.g. `Qwen3-Omni-...` | Same |
| `GOOGLE_API_KEY` | Voice Q&A (Gemini Flash) | [trygcp.dev claim](https://trygcp.dev/claim/junction-defence-saturday) → GCP → API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Default OK |
| `ELEVENLABS_API_KEY` | Voice output | elevenlabs.io |
| `ELEVENLABS_VOICE_ID` | Voice output | ElevenLabs voice library |
| `WS_PORT` | Default `8765` | Optional |
| `LIVEKIT_*` | Optional later | livekit.io cloud |

**Without keys:** system still runs — HUD updates, fallback text in terminal, no TTS.

## LiveKit (optional phase 2)

LiveKit Cloud is **not required** for v1 — audio plays on the laptop speaker via ElevenLabs.

To route audio into the browser later, add `LIVEKIT_URL`, API key/secret, and we can enable `livekit` package publish.

## Verify WebSocket

With `python main.py` running:

```bash
# macOS
nc -z 127.0.0.1 8765

# or open HUD — top center should say LINK ACTIVE
```

## File map

```
main.py              ← tracker loop + tactical integration
ai/store.py          ← JSON truth
ai/events.py         ← when to speak
ai/narrator.py       ← background narration queue
ai/voice_ptt.py      ← T key Q&A
bridge/ws_server.py  ← React feed
hud/                 ← overlay UI
```
