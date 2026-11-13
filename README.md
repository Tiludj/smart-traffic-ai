# Smart Traffic AI - Hyderabad

Intelligent urban traffic management using **deep learning, machine learning, graph optimisation and agentic AI**.

## Run it (5 minutes)

```bash
pip install -r requirements.txt
pip install -r requirements-vision.txt   # only if you run vision.py / live.py locally
streamlit run app.py
```

The app works immediately with **no API key and no video** (simulated traffic + rule-based agent).
Then add the two upgrades below.

### Upgrade 1 - real camera counts (deep learning)

Get a 30-60 s Indian traffic video (YouTube "Hyderabad traffic junction CCTV", or record one), save it as `traffic.mp4`, then:

```bash
python vision.py --video traffic.mp4 --junction "Hitech City"
```

This creates `outputs/counts.json`, `outputs/annotated.mp4` and `outputs/frame.jpg`. Refresh the app: Hitech City now uses camera counts.

- Default model `yolo11n.pt` downloads automatically (COCO classes: car, motorcycle, bus, truck).
- Better for India: download UVH-26 weights (IISc, trained on Bengaluru CCTV, detects autos and two-wheelers) from
  https://huggingface.co/iisc-aim/UVH-26 and run `python vision.py --video traffic.mp4 --model <weights>.pt`
- Slow laptop? Add `--max-seconds 30 --stride 3`.

### Upgrade 2 - real LLM agent (agentic AI)

1. Get a free API key: Groq (https://console.groq.com) or Gemini (https://aistudio.google.com).
2. In the app sidebar, choose the provider and paste the key.
   Groq models that work (Nov 2026): `openai/gpt-oss-20b` for testing, `openai/gpt-oss-120b` for the demo.
   (`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` are no longer available on this account.)
   Save settings on Windows so they load automatically (open a new terminal afterwards):
   `setx LLM_PROVIDER "Groq (free tier)"`, `setx LLM_API_KEY "gsk_..."`, `setx LLM_MODEL "openai/gpt-oss-20b"`
3. Windows: run the app with `python -m streamlit run app.py` (plain `streamlit` is not on PATH with Microsoft Store Python).
   If `pip install` fails on torch (long path error), skip `ultralytics` locally and run `vision.py` on Google Colab.

If the LLM fails during the demo, the rule-based agent answers automatically - the demo never breaks.

### Upgrade 3 - multiple cameras and a LIVE camera

- Recorded videos: run `vision.py` once per junction (`--junction "Panjagutta"` etc.). Each gets `outputs/<junction>/`.
- Ask the agent about a junction that has a camera and its video pops up in the chat answer.
- Live feed (needs `ultralytics` installed on the laptop):
  1. Phone: install **IP Webcam** (Android), tap *Start server*, note the URL (e.g. `http://192.168.1.5:8080`).
     Phone and laptop must be on the same Wi-Fi or the phone's hotspot.
  2. `python live.py --source http://192.168.1.5:8080/video --junction "Narayanaguda"`
     (`--source 0` = laptop webcam; `--source traffic.mp4` = replay a file as if live)
  3. Keep it running in its own terminal; the dashboard refreshes that camera every second.
- Windows: if `pip install ultralytics` fails with a long-path error, enable long paths (admin PowerShell):
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`
  then open a new terminal and retry.

## Files

| File | What it does | Tech |
|---|---|---|
| `vision.py` | YOLO + ByteTrack vehicle counting on recorded video, PCU conversion | Deep learning |
| `live.py` | Same, on a live feed (phone / webcam / CCTV), updates the dashboard every second | Deep learning |
| `forecast.py` | Random Forest predicting congestion 30 min ahead | Machine learning |
| `core.py` | Junction graph, routing on predicted times, green corridor, Webster signal timing, queue simulation | Graph optimisation |
| `agent.py` | LLM agent with 4 tools + rule-based fallback | Agentic AI |
| `app.py` | Streamlit dashboard | UI |

## Demo script (3 minutes)

1. **Problem** (20 s): Hyderabad loses hours daily to congestion; signals run fixed timers; ambulances get stuck.
2. **Camera** (30 s): show the YOLO video - counts two-wheelers, cars, buses; converted to PCU.
3. **Forecast** (30 s): move the time slider from 14:00 to 09:30 - the map turns red *before* the jam (30-min forecast).
4. **Agent** (60 s): click "Why is Hitech City red?" - agent checks the forecast and re-times signals (wait time drops).
   Then click "Ambulance at Jubilee Hills" - purple green corridor appears, ETA almost halves.
5. **Results** (20 s): bar chart - smart signals cut average waiting time vs fixed timing at every junction.
6. **Future scope** (20 s): GNN forecasting on city-wide sensor data, RL signal control in SUMO, live CCTV feeds.

## Be honest with judges about

- Traffic demand for most junctions is **simulated** from a realistic daily profile; one junction uses real YOLO counts.
- The forecast model is trained on that simulated data; the same pipeline retrains on real junction data.
- Waiting-time results come from a queue simulation, not a field trial.
- Junction coordinates are approximate.

## Customise quickly

- Add/move junctions or roads: `JUNCTIONS` and `ROADS` at the top of `core.py`.
- Change busyness of a junction: third number in `JUNCTIONS`.
- Quick-prompt buttons: `QUICK_PROMPTS` in `app.py`.
