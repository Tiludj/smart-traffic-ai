"""
ui.py - Look and feel: page CSS, map overlays (legend, notifications, KPI cards, route card),
sparklines and small HTML helpers. Only presentation lives here.
"""

import folium
from branca.element import MacroElement, Template

# Neon palette on a dark map
NEON = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
BLUE, PURPLE, CYAN = "#3b82f6", "#a855f7", "#22d3ee"
PANEL = "rgba(13,20,36,.88)"
BORDER = "rgba(148,163,184,.18)"

PAGE_CSS = """
<style>
.stApp {background: radial-gradient(1200px 650px at 15% -10%, #15244a 0%, #0b1220 55%);}
.block-container {padding-top: 1.1rem; padding-bottom: .6rem; max-width: 100%;}
[data-testid="stHeader"] {background: transparent; pointer-events: none;}
[data-testid="stToolbar"] {pointer-events: none;}
[data-testid="stHeader"] button, [data-testid="stHeader"] a {pointer-events: auto;}
[data-testid="stAppDeployButton"], #MainMenu {display: none;}
section[data-testid="stSidebar"] {background: #0d1526; border-right: 1px solid rgba(148,163,184,.12);}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(15,23,42,.78); border-radius: 18px; border: 1px solid rgba(148,163,184,.16) !important;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);}
.stButton > button {border-radius: 999px; background: #13203b; border: 1px solid rgba(148,163,184,.22);
  color: #dbe6f7; font-size: .85rem; padding: .35rem .9rem;}
.stButton > button:hover {border-color: #3b82f6; color: #fff; background: #172a50;}
[data-testid="stChatMessage"] {background: rgba(19,32,59,.65); border-radius: 14px; padding: .6rem .8rem;}
[data-testid="stChatInput"] textarea {font-size: .92rem;}
[data-testid="stTextInput"] input {border-radius: 999px; background: #101a30;}
[data-testid="stExpander"] {border-radius: 14px; border-color: rgba(148,163,184,.16);}
h1, h2, h3 {letter-spacing: -.01em;}
.nav-brand {display:flex; align-items:center; gap:.75rem;}
.nav-brand .t1 {font-size:1.35rem; font-weight:700; color:#f1f5ff; line-height:1.1}
.nav-brand .t2 {font-size:.78rem; color:#93a4c3}
.nav-status {display:flex; justify-content:flex-end; align-items:center; gap:.6rem; height:42px; white-space:nowrap;
  font-size:.85rem; color:#cbd7ea}
.live-dot {display:inline-block; width:9px; height:9px; border-radius:50%; background:#22c55e;
  box-shadow:0 0 0 0 rgba(34,197,94,.7); animation:pulse 1.6s infinite; margin-right:6px}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(34,197,94,.7)} 70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}
  100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
.chip {white-space:nowrap; padding:.25rem .65rem; border-radius:999px; background:#13203b; border:1px solid rgba(148,163,184,.2)}
.panel-title {font-size:1.05rem; font-weight:700; color:#f1f5ff; margin:0 0 .15rem}
.panel-sub {font-size:.78rem; color:#93a4c3; margin-bottom:.4rem}
.footer {display:flex; gap:1.4rem; align-items:center; color:#8fa0bd; font-size:.8rem; padding:.3rem .2rem}
.footer b {color:#cbd7ea; font-weight:600}
</style>
"""

LOGO_SVG = """<svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#60a5fa"/>
<stop offset="1" stop-color="#2563eb"/></linearGradient></defs>
<rect x="2" y="2" width="36" height="36" rx="11" fill="url(#lg)"/>
<path d="M14 33 L18 7 M26 33 L22 7" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/>
<path d="M20 11v3 M20 18v3 M20 25v4" stroke="#bfdbfe" stroke-width="2" stroke-linecap="round"/>
<circle cx="31" cy="10" r="3.2" fill="#22c55e" stroke="#0b1220" stroke-width="1.5"/></svg>"""


def nav_brand():
    return (f"<div class='nav-brand'>{LOGO_SVG}<div><div class='t1'>Smart Traffic AI</div>"
            f"<div class='t2'>AI traffic control · Hyderabad</div></div></div>")


def nav_status(clock, weather, live=True):
    temp = f"{weather['temp_c']:.0f}°C " if weather.get("temp_c") is not None else ""
    badge = ("<span style='color:#22c55e;font-weight:600'><span class='live-dot'></span>Live</span>" if live else
             "<span style='color:#f59e0b;font-weight:600'>⏸ Simulation</span>")
    return (f"<div class='nav-status'><span class='chip'>{weather['icon']} {temp}{weather['condition']}</span>"
            f"<span class='chip'>🕘 {clock}</span>{badge}</div>")


# ----------------------------------------------------------------------------- map overlays
class HtmlControl(MacroElement):
    """A block of HTML pinned to a corner of the Leaflet map (legend, cards, panels)."""
    _template = Template("""
    {% macro script(this, kwargs) %}
    (function () {
      var c = L.control({position: {{ this.position|tojson }}});
      c.onAdd = function () {
        var d = L.DomUtil.create('div', 'stai-ctl');
        d.innerHTML = {{ this.html|tojson }};
        L.DomEvent.disableClickPropagation(d);
        L.DomEvent.disableScrollPropagation(d);
        return d;
      };
      c.addTo({{ this._parent.get_name() }});
    })();
    {% endmacro %}
    """)

    def __init__(self, html, position="topleft"):
        super().__init__()
        self._name = "HtmlControl"
        self.html = html
        self.position = position


class MapStyle(MacroElement):
    """Dark styling inside the map frame + a zoom control at the bottom right."""
    _template = Template("""
    {% macro script(this, kwargs) %}
    (function () {
      var s = document.createElement('style');
      s.innerHTML = {{ this.css|tojson }};
      document.head.appendChild(s);
      L.control.zoom({position: 'bottomright'}).addTo({{ this._parent.get_name() }});
    })();
    {% endmacro %}
    """)
    css = """
    .leaflet-container {background:#0b1220; font-family: 'Source Sans Pro', 'Segoe UI', sans-serif;}
    .leaflet-tile-pane {filter: invert(100%) hue-rotate(195deg) brightness(.9) contrast(1.05) saturate(.5);}
    .leaflet-control-attribution {background:rgba(11,18,32,.7)!important; color:#7d8db0!important; font-size:10px!important}
    .leaflet-control-attribution a {color:#93c5fd!important}
    .leaflet-bar a {background:rgba(13,20,36,.92); color:#e5edf8; border-color:rgba(148,163,184,.2)}
    .leaflet-bar a:hover {background:#1d2b4d; color:#fff}
    .leaflet-bar {border:1px solid rgba(148,163,184,.25)!important; border-radius:12px!important; overflow:hidden}
    .leaflet-tooltip {background:rgba(13,20,36,.95); color:#e5edf8; border:1px solid rgba(148,163,184,.3);
      border-radius:8px; box-shadow:0 6px 18px rgba(0,0,0,.4)}
    .leaflet-tooltip:before {display:none}
    .leaflet-popup-content-wrapper, .leaflet-popup-tip {background:#0f182c; color:#e5edf8}
    .stai-ctl {margin:12px!important}
    @keyframes ring {0%{transform:scale(.6);opacity:.9} 100%{transform:scale(1.8);opacity:0}}
    .ring {position:absolute; border-radius:50%; animation:ring 1.6s ease-out infinite}
    """

    def __init__(self):
        super().__init__()
        self._name = "MapStyle"


def card_css(width=None, pad="12px 14px"):
    w = f"width:{width}px;" if width else ""
    return (f"{w}box-sizing:border-box;background:{PANEL};border:1px solid {BORDER};border-radius:16px;padding:{pad};"
            f"color:#e5edf8;font-size:12.5px;box-shadow:0 12px 30px rgba(0,0,0,.45);"
            f"backdrop-filter:blur(6px);")


def legend_pills():
    pill = lambda c, t: (f"<span style='display:inline-flex;align-items:center;gap:6px;margin-right:12px'>"
                         f"<span style='width:10px;height:10px;border-radius:50%;background:{c};"
                         f"box-shadow:0 0 8px {c}'></span>{t}</span>")
    return (f"<div style='{card_css(pad='9px 14px')}border-radius:999px;font-size:12.5px'>"
            + pill(NEON["Low"], "Free flow") + pill(NEON["Medium"], "Moderate") + pill(NEON["High"], "Heavy")
            + pill(PURPLE, "Green corridor").replace("margin-right:12px", "margin-right:0") + "</div>")


BADGE = {"High": ("#ef4444", "rgba(239,68,68,.16)"), "Medium": ("#f59e0b", "rgba(245,158,11,.16)"),
         "Info": ("#60a5fa", "rgba(59,130,246,.18)"), "Good": ("#22c55e", "rgba(34,197,94,.16)")}


def notifications_panel(items):
    """items: [(icon, title, subtitle, level)]"""
    rows = ""
    for icon, title, sub, level in items[:4]:
        fg, bg = BADGE[level]
        rows += (f"<div style='display:flex;align-items:center;gap:10px;padding:9px 0;"
                 f"border-top:1px solid rgba(148,163,184,.12)'>"
                 f"<div style='width:32px;height:32px;border-radius:10px;background:{bg};display:flex;"
                 f"align-items:center;justify-content:center;font-size:16px'>{icon}</div>"
                 f"<div style='flex:1;min-width:0'><div style='font-weight:600;font-size:13px;"
                 f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{title}</div>"
                 f"<div style='color:#8fa0bd;font-size:11.5px;white-space:nowrap;overflow:hidden;"
                 f"text-overflow:ellipsis'>{sub}</div></div>"
                 f"<span style='color:{fg};background:{bg};border:1px solid {fg}55;border-radius:999px;"
                 f"padding:2px 9px;font-size:11px;font-weight:600'>{level}</span></div>")
    return (f"<div style='{card_css(286)}'><div style='display:flex;justify-content:space-between;"
            f"align-items:center;margin-bottom:4px'><span style='font-size:15px;font-weight:700'>"
            f"Notifications</span><span style='color:#60a5fa;font-size:12px'>{len(items)} active</span></div>"
            f"{rows}</div>")


def sparkline(values, color, uid, w=190, h=36):
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    pts = [(i * w / max(len(values) - 1, 1), h - 3 - (v - lo) / span * (h - 8)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} " + line + f" {w},{h}"
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'><defs><linearGradient id='g{uid}' "
            f"x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='{color}' stop-opacity='.35'/>"
            f"<stop offset='1' stop-color='{color}' stop-opacity='0'/></linearGradient></defs>"
            f"<polygon points='{area}' fill='url(#g{uid})'/><polyline points='{line}' fill='none' "
            f"stroke='{color}' stroke-width='2' stroke-linejoin='round'/></svg>")


def kpi_card(icon, value, label, delta, delta_good, color, spark, uid):
    dcol = "#22c55e" if delta_good else "#ef4444"
    delta_html = (f"<span style='color:{dcol};font-size:11.5px;font-weight:600'>{delta}</span>" if delta else "")
    return (f"<div style='{card_css(196, '10px 12px 5px')}'>"
            f"<div style='display:flex;align-items:flex-start;gap:10px'>"
            f"<div style='font-size:22px;line-height:1;filter:drop-shadow(0 0 6px {color}88)'>{icon}</div>"
            f"<div style='flex:1'><div style='display:flex;justify-content:space-between;align-items:baseline'>"
            f"<span style='font-size:21px;font-weight:700;line-height:1.1'>{value}</span>{delta_html}</div>"
            f"<div style='color:#9fb0cc;font-size:11.5px;line-height:1.25'>{label}</div></div></div>"
            f"<div style='margin-top:4px'>{sparkline(spark, color, uid, 170, 24)}</div></div>")


def kpi_grid(cards):
    return ("<div style='display:grid;grid-template-columns:196px 196px;gap:8px'>"
            + "".join(cards) + "</div>")


def focus_card(icon, title, big, small, lines, accent=BLUE):
    body = "".join(f"<div style='color:#9fb0cc;font-size:11.5px;margin-top:3px'>{ln}</div>" for ln in lines)
    return (f"<div style='{card_css(292)}'><div style='display:flex;gap:12px;align-items:flex-start'>"
            f"<div style='font-size:24px;filter:drop-shadow(0 0 6px {accent})'>{icon}</div>"
            f"<div style='flex:1;min-width:0'><div style='font-weight:700;font-size:13.5px'>{title}</div>"
            f"<div style='margin-top:2px'><span style='color:{accent};font-size:21px;font-weight:700'>{big}</span>"
            f"<span style='color:#9fb0cc;font-size:12.5px'>&nbsp; {small}</span></div>{body}</div></div></div>")


# ----------------------------------------------------------------------------- map drawing helpers
def glow_line(fmap, points, color, tooltip=None, core_weight=3.5, dash=None):
    """Neon road: two soft glow passes under a bright core line."""
    folium.PolyLine(points, color=color, weight=core_weight * 5, opacity=0.10, interactive=False).add_to(fmap)
    folium.PolyLine(points, color=color, weight=core_weight * 2.4, opacity=0.25, interactive=False).add_to(fmap)
    folium.PolyLine(points, color=color, weight=core_weight, opacity=0.95, dash_array=dash,
                    tooltip=tooltip).add_to(fmap)


def pin(fmap, point, symbol, color, label=None):
    """Map pin (teardrop) with a symbol - cameras, hospitals."""
    html = (f"<div style='transform:translate(-12px,-33px)'><div style='width:24px;height:24px;"
            f"border-radius:50% 50% 50% 0;transform:rotate(-45deg);background:{color};"
            f"box-shadow:0 0 12px {color}aa;border:2px solid rgba(255,255,255,.85);display:flex;"
            f"align-items:center;justify-content:center'><span style='transform:rotate(45deg);"
            f"font-size:11px;color:#fff;font-weight:800'>{symbol}</span></div></div>")
    folium.Marker(point, interactive=False, icon=folium.DivIcon(icon_size=(0, 0), html=html)).add_to(fmap)


def label(fmap, point, text, color="#e5edf8", dx=11, dy=-8, size=11):
    html = (f"<div style='font-size:{size}px;font-weight:600;color:{color};white-space:nowrap;"
            f"text-shadow:0 0 3px #0b1220,0 0 6px #0b1220;transform:translate({dx}px,{dy}px)'>{text}</div>")
    folium.Marker(point, interactive=False, icon=folium.DivIcon(icon_size=(0, 0), html=html)).add_to(fmap)


def pulse_ring(fmap, point, color, size=44):
    html = (f"<div style='position:relative;transform:translate(-{size // 2}px,-{size // 2}px)'>"
            f"<div class='ring' style='width:{size}px;height:{size}px;border:3px solid {color}'></div></div>")
    folium.Marker(point, interactive=False, icon=folium.DivIcon(icon_size=(0, 0), html=html)).add_to(fmap)


def legend_html():
    def chip(sym, text):
        return f"<span style='margin-right:16px;white-space:nowrap'>{sym}&nbsp; {text}</span>"
    dot = lambda c: (f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
                     f"background:{c};box-shadow:0 0 8px {c};vertical-align:middle'></span>")
    bar = lambda c, dash=False: (f"<span style='display:inline-block;width:28px;vertical-align:middle;"
                                 + (f"border-top:3px dashed {c}" if dash else
                                    f"height:5px;border-radius:3px;background:{c};box-shadow:0 0 8px {c}")
                                 + "'></span>")
    return ("<div style='font-size:13px;line-height:2.2'>"
            "<b>Congestion in 30 min:</b> " + chip(dot(NEON["Low"]), "Low") + chip(dot(NEON["Medium"]), "Medium")
            + chip(dot(NEON["High"]), "High")
            + "<br><b>Routes:</b> " + chip(bar(BLUE), "Recommended route") + chip(bar("#94a3b8", True), "Usual route")
            + chip(bar(BLUE, True), "Normal-traffic route (in emergencies)") + chip(bar(PURPLE), "Green corridor")
            + "<br><b>Markers:</b> " + chip(dot("#22c55e"), "Signal held green")
            + chip("<b style='color:#10b981'>📷</b>", "Camera installed") + chip("<b style='color:#3b82f6'>H</b>", "Hospital")
            + chip("🚑 🚒 🚓", "Emergency vehicle") + chip("🏥 🚨", "Destination")
            + "<br><span style='color:#8fa0bd'>Click any junction for its camera, signal plan and forecast.</span></div>")
