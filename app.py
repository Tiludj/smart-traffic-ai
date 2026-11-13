"""
app.py - Smart Traffic AI dashboard (Streamlit).

    streamlit run app.py
"""

import datetime
import json
import math
import os

import folium
import pandas as pd
import streamlit as st

import agent
import core
import forecast
import ui
import weather as wx

st.set_page_config(page_title="Smart Traffic AI - Hyderabad", page_icon="🚦", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(ui.PAGE_CSS, unsafe_allow_html=True)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PROVIDERS = {
    "No LLM (rule-based agent)": None,
    "Groq (free tier)": ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
    "Gemini (free tier)": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"),
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
}
QUICK_PROMPTS = {   # button label: (prompt sent to the agent)
    "🚑 Ambulance": "Ambulance from Gachibowli to Osmania Hospital, patient critical",
    "🔴 Hitech City": "Why is Hitech City congested and what are you doing about it?",
    "🧭 Best route": "What is the best route from Narayanaguda to Hitech City right now?",
    "🚦 Fix signals": "Optimise the signal timings at Panjagutta",
    "🌧️ Weather": "What is the weather now and how does it affect traffic?",
    "📍 Jam hotspots": "Which junctions will be heavily congested in the next 30 minutes?",
}
VEHICLE = {"ambulance": ("🚑", "Ambulance"), "fire": ("🚒", "Fire engine"), "police": ("🚓", "Police")}

ss = st.session_state
ss.setdefault("chat", [])        # [{"role", "content", "steps", "camera"}]
ss.setdefault("actions", [])     # map highlights from the last agent run


@st.cache_data(ttl=600, show_spinner=False)
def live_weather():
    try:
        return wx.fetch_live()
    except Exception as e:  # no internet / API down - app keeps working
        return {"error": f"{type(e).__name__}"}


@st.cache_resource
def get_model():
    return forecast.train()


def read_bytes(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


@st.fragment(run_every=1.0)
def live_frame(cam_dir, junction, where):
    """Refreshes every second on its own - shows the newest frame written by live.py."""
    img = read_bytes(os.path.join(cam_dir, "frame.jpg"))
    raw = read_bytes(os.path.join(cam_dir, "counts.json"))
    info = ""
    if raw:
        try:
            d = json.loads(raw)
            info = (f"LIVE {d.get('updated', '')} - last {int(d['duration_s'])} s: "
                    f"{sum(d['counts_in_clip'].values())} vehicles ({d.get('fps', '?')} fps)")
        except (ValueError, KeyError):
            pass
    if img:
        st.image(img, caption=info or f"LIVE - {junction}")
    else:
        st.caption("Waiting for the live feed...")


def show_camera(junction, cameras, where="panel"):
    cam = cameras[junction]
    if cam.get("live"):
        live_frame(cam["dir"], junction, where)
        return
    video, frame = os.path.join(cam["dir"], "annotated.mp4"), os.path.join(cam["dir"], "frame.jpg")
    if os.path.exists(video):
        # alt text differs per place, so the same video can appear in chat and panel
        st.video(video, loop=True, autoplay=True, muted=True, alt=f"{where} camera {junction}")
    elif os.path.exists(frame):
        st.image(frame)


def reset_view():
    ss.actions, ss.map_view = [], None
    ss.pop("cam_focus", None)


def on_search():
    j = core.match_junction(ss.get("search_q", ""))
    if j:
        ss.cam_focus = j
        ss.map_view = {"type": "junction", "junction": j}
        ss.actions = []
        ss.search_note = None
    elif ss.get("search_q"):
        ss.search_note = f"No junction called '{ss.search_q}'"
    ss.search_q = ""


# ----------------------------------------------------------------------------- top bar + toolbar
nav = st.container()          # filled after the numbers are computed (needs weather + time)
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))   # India Standard Time (no DST)


def ist_now():
    return datetime.datetime.now(IST)


ss.setdefault("live_mode", True)
ss.setdefault("hour_slider", 9.5)
ss.setdefault("day_sel", "Wednesday")
t0, t1, t2, t3, t4 = st.columns([1.05, 3.6, 1.5, 1.9, 1.2], vertical_alignment="bottom")
live = t0.toggle("🔴 Live", key="live_mode", help="On: use the real time in Hyderabad and refresh "
                 "automatically. Off: pick any time with the slider.")
if live:
    now = ist_now()
    # Live mode: the slider and the day follow the real clock (set before the widgets are drawn)
    ss.hour_slider = now.hour + now.minute / 60
    ss.day_sel = DAYS[now.weekday()]
    ss.live_bucket = f"{now.hour}:{now.minute // 5}"     # the map refreshes every 5 minutes
_h = ss.hour_slider
_hm = f"{int(_h):02d}:{int(round((_h % 1) * 60)) % 60:02d}"
hour = t1.slider(f"🕘 Time of day: {_hm}" + (" (live)" if live else " (simulation)"), 0.0, 23.99, step=0.25,
                 key="hour_slider", disabled=live, format=" ",
                 help="Turn Live off to move the slider and replay any time of day")
time_str = f"{int(hour):02d}:{int(round((hour % 1) * 60)) % 60:02d}"
day = t2.selectbox("📅 Day", DAYS, key="day_sel", disabled=live)
dow = DAYS.index(day)
weather_mode = t3.selectbox("🌦️ Weather", ["Live (Hyderabad now)", "Clear", "Light rain", "Heavy rain",
                                           "Thunderstorm"],
                            help="Live = real weather from Open-Meteo. Pick rain to show its effect on traffic.")
if t4.button("⟲ Full map", width="stretch", disabled=not (ss.get("map_view") or ss.actions),
             help="Show the whole city again (keeps the chat)"):
    reset_view()
    st.rerun()

# ----------------------------------------------------------------------------- sidebar: agent settings
def setting(name, default=""):
    """Read a setting from environment variables (laptop) or Streamlit secrets (cloud)."""
    if os.getenv(name):
        return os.getenv(name)
    try:
        return str(st.secrets.get(name, default))
    except Exception:          # no secrets file - normal on a laptop
        return default


with st.sidebar:
    st.markdown("### 🤖 AI agent settings")
    env_provider = setting("LLM_PROVIDER")
    provider = st.selectbox("LLM provider", list(PROVIDERS),
                            index=list(PROVIDERS).index(env_provider) if env_provider in PROVIDERS else 0)
    api_key, base_url, model_name = "", "", ""
    if PROVIDERS[provider]:
        default_url, default_model = PROVIDERS[provider]
        saved_key = setting("LLM_API_KEY")
        # The saved key is never put in the page, so visitors of a public site cannot see it
        typed_key = st.text_input("API key", type="password",
                                  placeholder="Saved key in use ✓" if saved_key else "Paste your API key")
        api_key = typed_key or saved_key
        model_name = st.text_input("Model", value=setting("LLM_MODEL", default_model))
        base_url = st.text_input("Base URL", value=default_url)
        if not api_key:
            st.info("No key yet - the rule-based agent will answer.")
    st.caption("Settings load from LLM_PROVIDER, LLM_API_KEY and LLM_MODEL "
               "(environment variables or Streamlit secrets).")

# ----------------------------------------------------------------------------- compute
model, accuracy = get_model()
cameras = core.load_cameras()
if weather_mode.startswith("Live"):
    weather = live_weather()
    if "error" in weather:
        weather = wx.make_weather("Clear", source=f"Live weather unavailable ({weather['error']}) - assuming clear")
else:
    weather = wx.make_weather(weather_mode, source="Simulated (sidebar)")
wf = weather["traffic_factor"]
state = core.current_state(hour, dow, cameras, wf)
pred = forecast.predict(model, state, hour, dow, wf)
core.set_edge_times({n: p["utilisation"] for n, p in pred.items()})
signals = {n: core.compare_signals(n, s["pcu_per_hour"]) for n, s in state.items()}
names = [n for n in core.JUNCTIONS if n not in core.HOSPITALS]
hot = [n for n in names if pred[n]["label"] == "High"]


@st.cache_data(show_spinner=False)
def trend(hour, dow, wf, cam_key):
    """KPI history for the sparklines: last 6 hours in 30-minute steps."""
    m, _ = get_model()
    cams = core.load_cameras()
    out = {"vehicles": [], "congestion": [], "jams": []}
    h = hour - 6
    while h <= hour + 1e-9:
        hh = h % 24                       # wraps past midnight in live mode
        st_h = core.current_state(hh, dow, cams, wf)
        pr = forecast.predict(m, st_h, hh, dow, wf)
        out["vehicles"].append(sum(sum(s["counts_per_hour"].values()) for s in st_h.values()))
        out["congestion"].append(sum(min(s["utilisation"], 1.2) for s in st_h.values()) / len(st_h) * 100)
        out["jams"].append(sum(1 for n in pr if pr[n]["label"] == "High" and not core.GRAPH.nodes[n]["hospital"]))
        h += 0.5
    return out


tr = trend(hour, dow, wf, tuple(sorted(cameras)))


@st.fragment(run_every=1.0)
def nav_clock(sim_time, sim_day, weather):
    """Ticks every second. In live mode it also refreshes the whole dashboard every 5 minutes."""
    now = ist_now()
    if ss.get("live_mode"):
        st.markdown(ui.nav_status(now.strftime("%a %I:%M:%S %p"), weather, live=True), unsafe_allow_html=True)
        if ss.get("live_bucket") != f"{now.hour}:{now.minute // 5}":
            st.rerun(scope="app")
    else:
        st.markdown(ui.nav_status(f"{sim_day[:3]} {sim_time}", weather, live=False), unsafe_allow_html=True)

# ----------------------------------------------------------------------------- top bar
with nav:
    b1, b2, b3, b4 = st.columns([2.2, 2.1, 3.0, 2.4], vertical_alignment="center")
    b1.markdown(ui.nav_brand(), unsafe_allow_html=True)
    b2.text_input("Search", key="search_q", placeholder="🔍  Search junction… e.g. Narayanaguda",
                  label_visibility="collapsed", on_change=on_search)
    page = b3.segmented_control("View", ["🗺️ Map", "📊 Analytics", "📹 Cameras"], default="🗺️ Map",
                                label_visibility="collapsed", key="page") or "🗺️ Map"
    with b4:
        nav_clock(time_str, day, weather)
if ss.get("search_note"):
    st.toast(ss.search_note, icon="🔍")
    ss.search_note = None


# ----------------------------------------------------------------------------- map
def path_km(path):
    return round(sum(core.GRAPH[a][b]["km"] for a, b in zip(path, path[1:])), 1)


def overlays():
    """Notifications, KPI cards and the focus card shown on top of the map."""
    view = ss.get("map_view") or {}
    cor = next((a for a in ss.actions if a["type"] == "corridor"), None)
    rte = next((a for a in ss.actions if a["type"] == "route"), None)
    sig = next((a for a in ss.actions if a["type"] == "signals"), None)

    notes = []
    if cor:
        icon, name = VEHICLE[cor.get("vehicle", "ambulance")]
        notes.append((icon, f"{name} corridor", f"{cor['path'][0]} → {cor['path'][-1]} · "
                      f"ETA {cor.get('eta')} min", "High"))
    for n in sorted(hot, key=lambda n: -pred[n]["probability"])[:2]:
        notes.append(("⚠️", f"Jam at {n}", f"expected in 30 min · "
                      f"{int(pred[n]['probability'] * 100)}% sure", "High" if n == hot[0] else "Medium"))
    if wf > 1:
        notes.append((weather["icon"], f"{weather['condition']} slowing traffic", weather["traffic_impact"], "Medium"))
    if rte:
        notes.append(("🧭", "Route updated", f"{rte['path'][0]} → {rte['path'][-1]} · {rte.get('minutes')} min", "Info"))
    if sig:
        c = signals[sig["junction"]]
        notes.append(("🚦", f"Signals retimed at {sig['junction']}",
                      f"wait {c['fixed_wait_s']} s → {c['smart_wait_s']} s", "Info"))
    if not notes:
        notes.append(("✅", "Traffic flowing normally", "no jams expected in 30 min", "Good"))

    def delta(series, good_when_up):
        if len(series) < 3 or not series[-3]:
            return "", True
        pct = (series[-1] - series[-3]) / series[-3] * 100
        return f"{'↑' if pct >= 0 else '↓'} {abs(pct):.0f}%", (pct >= 0) == good_when_up

    veh = tr["vehicles"][-1]
    d1, g1 = delta(tr["vehicles"], True)
    d2, g2 = delta(tr["congestion"], False)
    jams_delta = tr["jams"][-1] - (tr["jams"][-3] if len(tr["jams"]) >= 3 else tr["jams"][-1])
    total_pcu = sum(s["pcu_per_hour"] for s in state.values())
    fixed_w = sum(signals[n]["fixed_wait_s"] * state[n]["pcu_per_hour"] for n in state) / total_pcu
    smart_w = sum(signals[n]["smart_wait_s"] * state[n]["pcu_per_hour"] for n in state) / total_pcu
    cards = [
        ui.kpi_card("🚗", f"{veh / 1000:.1f}K", "Vehicles / hour", d1, g1, "#22c55e", tr["vehicles"], "v"),
        ui.kpi_card("🚧", f"{tr['congestion'][-1]:.0f}%", "Avg congestion", d2, g2, "#f59e0b", tr["congestion"], "c"),
        ui.kpi_card("⚠️", f"{len(hot)}", "Jams in next 30 min",
                    f"{'↑' if jams_delta >= 0 else '↓'} {abs(jams_delta)}" if jams_delta else "", jams_delta <= 0,
                    "#ef4444", tr["jams"], "j"),
        ui.kpi_card("🚑" if cor else "🚦", f"{1 if cor else 0}" if cor else f"{smart_w:.0f}s",
                    "Emergency corridors" if cor else "Avg signal wait (AI)",
                    f"saves {cor.get('saved')} min" if cor else f"↓ {(fixed_w - smart_w) / fixed_w * 100:.0f}%",
                    True, "#3b82f6", tr["vehicles"][-6:] if not cor else [0, 0, 0, 1, 1], "e"),
    ]

    if cor:
        icon, name = VEHICLE[cor.get("vehicle", "ambulance")]
        same = not cor.get("normal_path") or cor["normal_path"] == cor["path"]
        focus = ui.focus_card(icon, f"{name} green corridor", f"{cor.get('eta')} min",
                              f"• {path_km(cor['path'])} km · saves {cor.get('saved')} min",
                              [" → ".join(cor["path"]),
                               f"{cor.get('signals', 0)} signals held green · "
                               + ("same road as normal traffic" if same else "normal traffic sent another way")],
                              ui.PURPLE)
    elif rte:
        focus = ui.focus_card("🧭", "Recommended route", f"{rte.get('minutes')} min",
                              f"• {path_km(rte['path'])} km",
                              [" → ".join(rte["path"]),
                               f"saves {rte['saved']} min vs usual route" if rte.get("saved", 0) > 0.5
                               else "usual route is already the fastest"])
    elif view.get("type") == "junction" and view.get("junction") in state:
        j = view["junction"]
        c = signals[j]
        focus = ui.focus_card("📍", j, pred[j]["label"], f"in 30 min · now {state[j]['congestion']}",
                              [f"{state[j]['pcu_per_hour']} PCU/hour · source: {state[j]['source']}",
                               f"Signal wait {c['fixed_wait_s']} s → {c['smart_wait_s']} s with AI timing"],
                              ui.NEON[pred[j]["label"]])
    elif view.get("type") == "weather":
        focus = ui.focus_card(weather["icon"], "Weather impact", weather["condition"], weather["traffic_impact"],
                              [weather["source"]], "#60a5fa")
    else:
        focus = ui.focus_card("🛰️", "City overview", f"{len(names) - len(hot)}/{len(names)}",
                              "junctions clear for the next 30 min",
                              [f"Jams expected at: {', '.join(hot[:4])}{'…' if len(hot) > 4 else ''}" if hot
                               else "No jams expected", "Ask the agent or click a junction"], "#22c55e")
    return notes, cards, focus


def draw_map():
    lats = [v[0] for v in core.JUNCTIONS.values()]
    lons = [v[1] for v in core.JUNCTIONS.values()]
    px_per_deg_z0 = 256 / 360
    route_action = next((a for a in ss.actions if a["type"] in ("corridor", "route")), None)
    view = ss.get("map_view") or {}
    nodes = list(core.JUNCTIONS)
    if route_action:
        nodes = route_action["path"] + route_action.get("normal_path", []) + route_action.get("alt_path", [])
    elif view.get("type") == "junction" and view.get("junction") in core.JUNCTIONS:
        nodes = [view["junction"]] + list(core.GRAPH.neighbors(view["junction"]))
    plats = [core.JUNCTIONS[n][0] for n in nodes]
    plons = [core.JUNCTIONS[n][1] for n in nodes]
    lon_span = max(max(plons) - min(plons), 0.01)
    lat_span = max(max(plats) - min(plats), 0.01)
    # fit into the free middle of the map (overlay cards cover the corners)
    zoom = int(min(math.log2(520 / (lon_span * px_per_deg_z0)), math.log2(330 / (lat_span * 1.05 * px_per_deg_z0))))
    zoom = max(10, min(zoom, 15))
    # The overlay cards cover the right side and the bottom-left corner, so place what was asked
    # in the free area: shift the map centre ~145 px east and ~80 px south of the content centre.
    deg_per_px = 360 / (256 * 2 ** zoom)
    mid_lat, mid_lon = (max(plats) + min(plats)) / 2, (max(plons) + min(plons)) / 2
    center = [mid_lat - 80 * deg_per_px * math.cos(math.radians(mid_lat)), mid_lon + 145 * deg_per_px]
    # Free OpenStreetMap tiles (no API key); ui.MapStyle turns them into a dark map with a CSS filter
    fmap = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap",
                      zoom_control=False)
    fmap.add_child(ui.MapStyle())
    pos = {n: (v[0], v[1]) for n, v in core.JUNCTIONS.items()}

    # Only show what the last question was about (None = everything)
    visible_nodes, visible_edges = None, None
    if route_action:
        paths = [a[k] for a in ss.actions if a["type"] in ("route", "corridor")
                 for k in ("path", "normal_path", "alt_path") if a.get(k)]
        visible_nodes = {n for p in paths for n in p}
        visible_edges = {frozenset(e) for p in paths for e in zip(p, p[1:])}
    elif view.get("type") == "junction" and view.get("junction") in core.JUNCTIONS:
        j = view["junction"]
        visible_nodes = {j} | set(core.GRAPH.neighbors(j))
        visible_edges = {frozenset((j, n)) for n in core.GRAPH.neighbors(j)}
    elif view.get("type") == "hotspots":
        visible_nodes = set(hot)
        visible_edges = {frozenset((u, v)) for u, v in core.GRAPH.edges() if u in visible_nodes and v in visible_nodes}

    # Neon roads coloured by predicted congestion
    for u, v, d in core.GRAPH.edges(data=True):
        if visible_edges is not None and frozenset((u, v)) not in visible_edges:
            continue
        worst = max([pred[u]["label"], pred[v]["label"]], key=["Low", "Medium", "High"].index)
        ui.glow_line(fmap, [pos[u], pos[v]], ui.NEON[worst],
                     tooltip=f"{u} - {v}: {d['km']} km, ~{d['pred_min']:.0f} min predicted")

    # Agent highlights
    green_nodes, ends = [], []
    for a in ss.actions:
        if a["type"] == "route":
            folium.PolyLine([pos[n] for n in a["alt_path"]], color="#94a3b8", weight=3, dash_array="8",
                            tooltip="Usual route").add_to(fmap)
            ui.glow_line(fmap, [pos[n] for n in a["path"]], ui.BLUE, "Recommended route", core_weight=6)
        elif a["type"] == "corridor":
            vehicle = a.get("vehicle", "ambulance")
            if a.get("normal_path") and a["normal_path"] != a["path"]:
                folium.PolyLine([pos[n] for n in a["normal_path"]], color=ui.BLUE, weight=4, dash_array="10",
                                opacity=0.95, tooltip="Best route for normal traffic").add_to(fmap)
            ui.glow_line(fmap, [pos[n] for n in a["path"]], ui.PURPLE, f"Green corridor ({vehicle})", core_weight=7)
            folium.PolyLine([pos[n] for n in a["path"]], color="#f5e8ff", weight=2, opacity=0.9,
                            interactive=False).add_to(fmap)
            ends = [(a["path"][0], VEHICLE[vehicle][0]),
                    (a["path"][-1], "🏥" if core.GRAPH.nodes[a["path"][-1]]["hospital"] else "🚨")]
            green_nodes = a["path"][1:-1]

    # Junctions
    for n, (lat, lon, _, hosp) in core.JUNCTIONS.items():
        if visible_nodes is not None and n not in visible_nodes:
            continue
        s, p = state[n], pred[n]
        popup = (f"<b>{n}</b><br>Now: {s['congestion']} ({s['pcu_per_hour']} PCU/h)<br>"
                 f"In 30 min: {p['label']} ({int(p['probability'] * 100)}%)<br>Source: {s['source']}")
        if hosp:
            folium.CircleMarker([lat, lon], radius=8, color="#3b82f6", weight=2, fill=True, fill_color="#0b1220",
                                fill_opacity=1, tooltip=n, popup=popup).add_to(fmap)
            ui.pin(fmap, [lat, lon], "H", "#3b82f6")
            if visible_nodes is not None:          # city view is crowded: name on hover only
                ui.label(fmap, [lat, lon], n, "#93c5fd")
            continue
        col = ui.NEON[p["label"]]
        folium.CircleMarker([lat, lon], radius=13, color=col, weight=0, fill=True, fill_color=col,
                            fill_opacity=0.18, interactive=False).add_to(fmap)
        folium.CircleMarker([lat, lon], radius=7, color="#e5edf8", weight=1.5, fill=True, fill_color=col,
                            fill_opacity=1, tooltip=f"{n}: {p['label']} in 30 min", popup=popup).add_to(fmap)
        if n in cameras:
            ui.pin(fmap, [lat, lon], "📷", "#10b981")
        ui.label(fmap, [lat, lon], n)
    for node in green_nodes:
        folium.CircleMarker(pos[node], radius=9, color="#ffffff", weight=2.5, fill=True, fill_color="#22c55e",
                            fill_opacity=1, tooltip=f"{node}: signal held GREEN").add_to(fmap)
    for node, emoji in ends:
        folium.Marker(pos[node], interactive=False, icon=folium.DivIcon(icon_size=(0, 0), html=(
            f"<div style='font-size:30px;transform:translate(-15px,-52px);"
            f"filter:drop-shadow(0 0 8px rgba(168,85,247,.9))'>{emoji}</div>"))).add_to(fmap)
        ui.pulse_ring(fmap, pos[node], ui.PURPLE)

    if view.get("type") == "hotspots":
        for n in hot:
            ui.pulse_ring(fmap, pos[n], ui.NEON["High"], 40)
    if view.get("type") == "junction" and view.get("junction") in pos:
        ui.pulse_ring(fmap, pos[view["junction"]], ui.CYAN, 46)

    notes, cards, focus = overlays()
    fmap.add_child(ui.HtmlControl(ui.legend_pills(), "topleft"))
    fmap.add_child(ui.HtmlControl(ui.notifications_panel(notes), "topright"))
    fmap.add_child(ui.HtmlControl(ui.kpi_grid(cards), "bottomleft"))
    fmap.add_child(ui.HtmlControl(focus, "bottomright"))
    return fmap


def run_prompt(prompt):
    history = [{"role": m["role"], "content": m["content"]} for m in ss.chat]
    with st.spinner("Agent is thinking..."):
        reply, steps, actions, used = agent.run_agent(
            prompt, state, pred, history, api_key, base_url, model_name, f"{day} {time_str}", weather)
    candidates = agent.find_junctions(prompt)
    for a in actions:
        candidates += [a.get("junction")] + a.get("path", [])
    cam_j = next((j for j in candidates if j in cameras), None)
    ss.chat.append({"role": "user", "content": prompt})
    ss.chat.append({"role": "assistant", "content": reply, "steps": steps, "camera": cam_j})
    if cam_j:
        ss.cam_focus = cam_j
    tools_used = {s["tool"] for s in steps}
    asked_j = next(iter(agent.find_junctions(prompt)), None)
    acted_j = next((a.get("junction") for a in actions if a.get("junction")), None)
    if any(a["type"] in ("corridor", "route") for a in actions):
        ss.map_view = {"type": "path"}
    elif acted_j or asked_j:
        ss.map_view = {"type": "junction", "junction": acted_j or asked_j}
    elif "get_traffic_status" in tools_used:
        ss.map_view = {"type": "hotspots"}
    elif "get_weather" in tools_used:
        ss.map_view = {"type": "weather"}
    else:
        ss.map_view = None
    ss.actions = actions
    st.rerun()


if page.endswith("Map"):
    map_col, side_col = st.columns([7.3, 2.7], gap="medium")
    with map_col:
        fmap = draw_map()
        try:
            from streamlit_folium import st_folium   # clickable map
            clicked = st_folium(fmap, height=620, use_container_width=True, key="city_map",
                                returned_objects=["last_object_clicked_tooltip"])
            tip = (clicked or {}).get("last_object_clicked_tooltip")
            if tip and tip != ss.get("last_tip"):
                ss.last_tip = tip
                name = tip.split(":")[0].strip()
                if name in core.JUNCTIONS:       # a junction was clicked (not a road)
                    ss.cam_focus = name
                    ss.map_view = {"type": "junction", "junction": name}
                    ss.actions = []
                    st.rerun()
        except ImportError:                      # streamlit-folium not installed: static map
            st.iframe(fmap.get_root().render(), height=620)

    with side_col:
        with st.container(border=True):
            mode = "LLM · " + model_name if api_key else "Rule-based"
            h1, h2 = st.columns([3, 1.3], vertical_alignment="center")
            h1.markdown(f"<div class='panel-title'>🤖 AI Traffic Agent</div>"
                        f"<div class='panel-sub'>{mode} · acts and explains</div>", unsafe_allow_html=True)
            if h2.button("🧹 Clear", key="clear_chat", width="stretch", help="Clear the chat and the map"):
                ss.chat = []
                reset_view()
                st.rerun()
            prompt = None
            qc = st.columns(2)
            for i, (lbl, text) in enumerate(QUICK_PROMPTS.items()):
                if qc[i % 2].button(lbl, width="stretch", key=f"q{i}", help=text):
                    prompt = text
            box = st.container(height=300, border=False)
            with box:
                if not ss.chat:
                    st.caption("Try: *Fire engine from Paradise to Koti* · *How is Narayanaguda?* · "
                               "*Is rain slowing traffic?*")
                for i, msg in enumerate(ss.chat):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"].replace("\n", "  \n"))
                        if msg.get("steps"):
                            with st.expander(f"Agent actions ({len(msg['steps'])} tool calls)"):
                                for s in msg["steps"]:
                                    st.markdown(f"**{s['tool']}**  `{s['args']}`")
                                    st.json(s["result"], expanded=False)
            typed = st.chat_input("Ask the traffic agent…")
            prompt = typed or prompt
            if prompt:
                run_prompt(prompt)

        focus_j = ss.get("cam_focus")
        with st.container(border=True):
            if focus_j in cameras:
                st.markdown(f"<div class='panel-title'>📹 Camera · {focus_j}</div>"
                            f"<div class='panel-sub'>YOLO11 · UVH-26 Indian vehicle model</div>",
                            unsafe_allow_html=True)
                show_camera(focus_j, cameras, where="side")
                counts = cameras[focus_j]["counts_in_clip"]
                st.caption(" · ".join(f"{k.replace('_', ' ').title()} {v}" for k, v in counts.items()))
            else:
                st.markdown("<div class='panel-title'>📹 Camera feeds</div>", unsafe_allow_html=True)
                st.caption((f"No camera at {focus_j}. " if focus_j else "")
                           + (f"Cameras at: {', '.join(cameras)}. Click one on the map (green 📷 pin)."
                              if cameras else "No camera data yet - run vision.py."))

    with st.expander("🗺️ Map legend"):
        st.markdown(ui.legend_html(), unsafe_allow_html=True)

# ----------------------------------------------------------------------------- analytics
elif page.endswith("Analytics"):
    a1, a2 = st.columns([1, 1.25], gap="medium")
    with a1:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>🚦 Signal optimisation</div>"
                        "<div class='panel-sub'>Webster's method vs today's fixed 120 s timer</div>",
                        unsafe_allow_html=True)
            default = ss.get("cam_focus") or (hot[0] if hot else "Hitech City")
            j = st.selectbox("Junction", names, index=names.index(default) if default in names else 0)
            c = signals[j]
            st.dataframe(pd.DataFrame({"Flow PCU/h": c["flows"], "Fixed green (s)": c["fixed"]["green_s"],
                                       "AI green (s)": c["smart"]["green_s"]}), width="stretch")
            s1, s2, s3 = st.columns(3)
            s1.metric("Cycle", f"{c['smart']['cycle_s']:.0f} s", f"was {c['fixed']['cycle_s']} s", delta_color="off")
            s2.metric("Avg wait", f"{c['smart_wait_s']} s",
                      f"{-(c['fixed_wait_s'] - c['smart_wait_s']) / c['fixed_wait_s'] * 100:.0f}%",
                      delta_color="inverse")
            s3.metric("In 30 min", pred[j]["label"], f"{int(pred[j]['probability'] * 100)}% sure", delta_color="off")
        with st.container(border=True):
            st.markdown("<div class='panel-title'>🧠 How it works</div>", unsafe_allow_html=True)
            st.markdown(f"""
| Layer | Technology | What it does |
|---|---|---|
| See | **Deep learning** · YOLO11 + ByteTrack | Counts two-wheelers, autos, cars, buses, trucks; converts to PCU |
| Predict | **Machine learning** · Random Forest ({accuracy}%) | Congestion 30 min ahead from time, day, traffic, weather |
| Decide | **Graph optimisation** · Dijkstra + Webster | Routes on predicted times, green corridors, signal timing |
| Act | **Agentic AI** · LLM + 5 tools | Understands requests, acts, explains; officer stays in control |
""")
    with a2:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>⏱️ Fixed vs AI signal timing</div>"
                        "<div class='panel-sub'>Average wait per vehicle, 1-hour queue simulation</div>",
                        unsafe_allow_html=True)
            chart = pd.DataFrame({"Fixed timing": {n: signals[n]["fixed_wait_s"] for n in names},
                                  "AI timing": {n: signals[n]["smart_wait_s"] for n in names}})
            st.bar_chart(chart, stack=False, horizontal=True, y_label="Avg wait per vehicle (s)", x_label="",
                         height=560, color=["#ef4444", "#22c55e"])
        with st.container(border=True):
            st.markdown("<div class='panel-title'>📋 Live data and 30-min forecast</div>", unsafe_allow_html=True)
            rows = [{"Junction": n, "Now": state[n]["congestion"], "PCU/hour": state[n]["pcu_per_hour"],
                     "In 30 min": pred[n]["label"], "Confidence": f"{int(pred[n]['probability'] * 100)}%",
                     "Source": state[n]["source"]} for n in names]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=300)

# ----------------------------------------------------------------------------- cameras
else:
    if not cameras:
        st.info("No camera data yet. Run: `python vision.py --video traffic.mp4 --junction \"Hitech City\"`")
    else:
        cam_names = list(cameras)
        focus = ss.get("cam_focus")
        c1, c2 = st.columns([1.6, 1], gap="medium")
        with c1:
            with st.container(border=True):
                cam_j = st.selectbox(f"Camera ({len(cam_names)} connected)", cam_names,
                                     index=cam_names.index(focus) if focus in cameras else 0)
                show_camera(cam_j, cameras)
        with c2:
            cam = cameras[cam_j]
            with st.container(border=True):
                st.markdown(f"<div class='panel-title'>🚗 Vehicles at {cam_j}</div>"
                            f"<div class='panel-sub'>{'LIVE feed' if cam.get('live') else str(cam['duration_s']) + ' s clip'}"
                            f" · model {cam['model']} · PCU = IRC:106</div>", unsafe_allow_html=True)
                df = pd.DataFrame({"In clip": cam["counts_in_clip"], "Per hour": cam["counts_per_hour"]})
                df.index = [i.replace("_", " ").title() for i in df.index]
                st.dataframe(df, width="stretch")
                st.bar_chart(pd.Series(cam["counts_in_clip"]).rename(lambda k: k.replace("_", " ").title()),
                             height=220, color="#22d3ee")
                st.caption("We only store vehicle counts - no faces, no number plates.")

# ----------------------------------------------------------------------------- footer
now = ist_now().strftime("%I:%M %p")
st.markdown(f"<div class='footer'><span>📍 <b>Hyderabad, Telangana</b></span><span>|</span>"
            f"<span>{len(names)} junctions · {len(core.HOSPITALS)} hospitals · {len(cameras)} camera"
            f"{'s' if len(cameras) != 1 else ''}</span><span>|</span><span>Weather: {weather['source']}</span>"
            f"<span>|</span><span>Last updated: <b>{now}</b></span></div>", unsafe_allow_html=True)
