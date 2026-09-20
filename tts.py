"""RFC Securities Robotic AI Assistant — text-to-speech engine.

Uses gTTS (Google Text-to-Speech) when available and returns an MP3 file that
the web player can stream or download. The player widget embeds the audio as
base64 so the robot can play, pause, resume, stop, mute, restart and change
speed entirely in the browser — plus generate a downloadable audio file.
"""
from __future__ import annotations

import base64
import io
import re
import time
import uuid

VOICE_MISSING_MSG = "Voice service is unavailable right now (no internet or gTTS)."


def synthesize(text: str, out_dir=None, *, lang: str = "en") -> dict:
    """Synthesise speech to an MP3 file and return {ok, path, error}."""
    from gtts import gTTS

    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean or len(clean) < 5:
        return {"ok": False, "path": None, "error": "Text is too short to speak."}

    import os
    from pathlib import Path
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(out_dir, f"rfc_speech_{uuid.uuid4().hex[:8]}.mp3")
    else:
        path = os.path.abspath(f"rfc_speech_{uuid.uuid4().hex[:8]}.mp3")

    try:
        tts = gTTS(text=clean, lang=lang)
        tts.save(path)
    except TypeError:
        tts = gTTS(text=clean, lang=lang)
        tts.save(path)
    except Exception as e:
        return {"ok": False, "path": None,
                "error": f"{VOICE_MISSING_MSG} ({e.__class__.__name__})"}
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"ok": False, "path": None, "error": VOICE_MISSING_MSG}
    return {"ok": True, "path": path, "error": ""}


def mp3_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ── the robotic web player ───────────────────────────────────────────────
def audio_player_html(mp3_path: str, *, autoplay: bool = False,
                      headline: str = "RFC Securities Robotic Assistant",
                      visible_text: str = "") -> str:
    """Return a self-contained robotic audio player with full transport
    controls (play / pause / resume / stop / mute / restart / speed) and a
    downloadable MP3 link."""
    b64 = mp3_base64(mp3_path)
    cid = uuid.uuid4().hex[:8]
    autoplay_js = "player.play();" if autoplay else ""
    visible = visible_text.replace("`", "").replace("\\", "").replace('"', "'")
    if len(visible) > 900:
        visible = visible[:897] + "..."
    return f"""
<div class="rfc-robot" id="rfc-robot-{cid}">
  <div class="rfc-robot-head">
    <span class="rfc-robot-face">\U0001F916</span>
    <div><div class="rfc-robot-title">{headline}</div>
    <div class="rfc-robot-sub">MODEL. PREDICT. OPTIMIZE.</div></div>
  </div>
  <audio id="audio-{cid}" preload="auto" src="data:audio/mpeg;base64,{b64}"></audio>
  <div class="rfc-now" id="now-{cid}">\u25B6 Standing by\u2026</div>
  <div class="rfc-buttons">
    <button class="rfc-btn" id="play-{cid}">\u25B6 Play</button>
    <button class="rfc-btn" id="pause-{cid}">\u23F8 Pause</button>
    <button class="rfc-btn" id="resume-{cid}">\u23ED Resume</button>
    <button class="rfc-btn" id="stop-{cid}">\u23F9 Stop</button>
    <button class="rfc-btn" id="mute-{cid}">\U0001F507 Mute</button>
    <button class="rfc-btn" id="restart-{cid}">\U0001F501 Restart</button>
    <select class="rfc-select" id="speed-{cid}" title="Playback speed">
      <option value="0.75">0.75\u00D7</option>
      <option value="1" selected>1\u00D7</option>
      <option value="1.25">1.25\u00D7</option>
      <option value="1.5">1.5\u00D7</option>
    </select>
  </div>
  <div class="rfc-text" id="text-{cid}">{visible}</div>
  <div class="rfc-foot">
    <span id="status-{cid}" class="rfc-status">offline</span>
    <a class="rfc-dl" download="rfc_securities_speech.mp3"
       href="data:audio/mpeg;base64,{b64}">\u2B07 Download audio (MP3)</a>
  </div>
</div>
<style>
#{cid}{{ display:none; }}
.rfc-robot{{background:linear-gradient(135deg,#0B3D2E,#157347);color:#fff;border:1px solid #C9A227;
  border-radius:14px;padding:1rem 1.1rem;margin:0.6rem 0;font-family:Segoe UI,Calibri,Arial,sans-serif;}}
.rfc-robot-head{{display:flex;align-items:center;gap:.7rem;margin-bottom:.5rem;}}
.rfc-robot-face{{font-size:2.1rem;line-height:1;}}
.rfc-robot-title{{font-weight:700;font-size:1.02rem;color:#F3E5B8;}}
.rfc-robot-sub{{font-size:.68rem;letter-spacing:.16em;color:#B8D8C8;}}
.rfc-now{{background:rgba(255,255,255,.08);border-radius:8px;padding:.35rem .6rem;
  font-size:.82rem;color:#fff;margin:.3rem 0;min-height:1.4rem;}}
.rfc-buttons{{display:flex;flex-wrap:wrap;gap:.35rem;margin:.5rem 0;}}
.rfc-btn{{background:#fff;color:#0B3D2E;border:none;border-radius:8px;padding:.3rem .65rem;
  font-size:.8rem;font-weight:600;cursor:pointer;}}
.rfc-btn:hover{{background:#F3E5B8;}}
.rfc-select{{background:#0B3D2E;color:#fff;border:1px solid #C9A227;border-radius:8px;padding:.3rem;}}
.rfc-text{{background:rgba(255,255,255,.06);border-left:3px solid #C9A227;padding:.5rem .7rem;
  border-radius:6px;font-size:.84rem;line-height:1.45;color:#F0F5F1;max-height:130px;overflow:auto;white-space:pre-wrap;}}
.rfc-foot{{display:flex;justify-content:space-between;align-items:center;margin-top:.45rem;}}
.rfc-status{{font-size:.72rem;color:#B8D8C8;}}
.rfc-dl{{color:#F3E5B8;font-size:.78rem;font-weight:600;text-decoration:none;}}
</style>
<script>
(function(){{
  const a=document.getElementById('audio-{cid}');
  const now=document.getElementById('now-{cid}');
  const status=document.getElementById('status-{cid}');
  const txt=document.getElementById('text-{cid}');
  let muted=false;
  function fmt(t){{const m=Math.floor(t/60),s=Math.floor(t%60);return m+':'+(s<10?'0':'')+s;}}
  a.addEventListener('timeupdate',function(){{now.textContent='\u25B6 Speaking \u00B7 '+fmt(a.currentTime)+' / '+fmt(a.duration||0);}});
  a.addEventListener('ended',function(){{now.textContent='\u23F9 Finished';status.textContent='finished';}});
  a.addEventListener('pause',function(){{status.textContent='paused';}});
  a.addEventListener('play',function(){{status.textContent='speaking';}});
  document.getElementById('play-{cid}').onclick=function(){{a.play();now.textContent='\u25B6 Playing \u00B7 '+fmt(a.currentTime||0);}};
  document.getElementById('pause-{cid}').onclick=function(){{a.pause();now.textContent='\u23F8 Paused \u00B7 '+fmt(a.currentTime||0);}};
  document.getElementById('resume-{cid}').onclick=function(){{a.play();}};
  document.getElementById('stop-{cid}').onclick=function(){{a.pause();a.currentTime=0;now.textContent='\u23F9 Stopped';}};
  document.getElementById('mute-{cid}').onclick=function(){{muted=!muted;a.muted=muted;this.textContent=muted?'\U0001F507 Unmute':'\U0001F507 Mute';}};
  document.getElementById('restart-{cid}').onclick=function(){{a.currentTime=0;a.play();}};
  document.getElementById('speed-{cid}').onchange=function(){{a.playbackRate=parseFloat(this.value);}};
  {autoplay_js}
}})();
</script>
"""


def how_speech_works_note() -> str:
    return ("The Robotic AI assistant converts the AI explanations into speech "
            "using on-the-fly text-to-speech. Use the player controls to "
            "play, pause, resume, stop, mute or restart, and download the "
            "generated MP3 for offline listening.")