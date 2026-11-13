"""
agent.py - Agentic AI: an LLM traffic-control agent that decides which tool to use,
calls it, and explains its decision in plain language.

Works with any OpenAI-compatible API (Groq, Gemini, OpenAI, ...).
If no API key is set, or the API fails, a rule-based fallback agent runs the
same tools - so the demo never dies on stage.
"""

import json
import re

import core

SYSTEM_PROMPT = """You are the AI Traffic Control Agent for a smart-city traffic
management centre in Hyderabad, India. You manage these junctions: {junctions}.
Hospitals: {hospitals}. Current simulated time: {time_str}.
Weather now: {weather}. Rain makes roads slower and more congested - mention it when relevant.

You have tools to check traffic (live + 30-minute ML forecast), find the best route,
create an ambulance green corridor, and optimise signal timings.

Rules:
- ALWAYS call a tool to get facts before answering. Never invent numbers.
- For ANY emergency vehicle (ambulance, fire engine, police, VIP convoy) call create_green_corridor
  immediately - do not check status first. Set vehicle to ambulance / fire / police.
  Ambulance with no hospital named: leave destination empty (= nearest hospital).
  Fire engine / police: destination = the incident location the user mentions.
- If a place is not in the junction list, say so and suggest the nearest listed junction.
- When the corridor path differs from normal_traffic_path, explain why: normal cars must avoid jams,
  but the emergency vehicle gets green signals and right of way, so the shorter road is fastest for it.
- Keep answers short (3-6 lines): what you found, what action you took, and why.
- Mention concrete numbers (minutes saved, waiting time reduction, congestion level).
- If an action affects many vehicles, end with: "Officer approval recommended."
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_traffic_status",
        "description": "Live traffic and 30-minute congestion forecast. Leave junction empty for all junctions.",
        "parameters": {"type": "object", "properties": {
            "junction": {"type": "string", "description": "Junction name, or empty for all"}}}}},
    {"type": "function", "function": {
        "name": "find_best_route",
        "description": "Fastest route between two junctions using PREDICTED travel times (avoids congestion forming in the next 30 min).",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string"}, "destination": {"type": "string"}},
            "required": ["source", "destination"]}}},
    {"type": "function", "function": {
        "name": "create_green_corridor",
        "description": "Emergency vehicle green corridor: fastest route avoiding predicted congestion, and turns every signal on the way green just before the vehicle arrives. Draws the route on the map.",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string", "description": "Junction where the emergency vehicle is now"},
            "destination": {"type": "string", "description": "Junction/hospital to reach; empty = nearest hospital"},
            "vehicle": {"type": "string", "enum": ["ambulance", "fire", "police"],
                        "description": "Type of emergency vehicle"}},
            "required": ["source"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current Hyderabad weather, rain forecast for the next hours, and its effect on traffic.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "optimize_signals",
        "description": "Recalculate signal timings for a junction (Webster's method) and compare waiting time against the current fixed timing.",
        "parameters": {"type": "object", "properties": {
            "junction": {"type": "string"}}, "required": ["junction"]}}},
]


class TrafficTools:
    """Wraps core.py functions. `ctx` holds the current state + forecast from the app."""

    def __init__(self, state, forecast, weather=None):
        self.state = state
        self.forecast = forecast
        self.weather = weather or {}
        self.actions = []  # what the map should highlight afterwards

    def weather_summary(self):
        w = self.weather
        if not w:
            return "unknown"
        temp = f", {w['temp_c']} C" if w.get("temp_c") is not None else ""
        return f"{w['condition']}{temp} ({w['source']}; traffic impact: {w['traffic_impact']})"

    def get_weather(self):
        w = dict(self.weather) if self.weather else {"condition": "unknown"}
        w.pop("icon", None)
        return w

    def _need(self, name):
        j = core.match_junction(name)
        if not j:
            raise ValueError(f"Unknown junction '{name}'. Known: {', '.join(core.JUNCTIONS)}")
        return j

    def get_traffic_status(self, junction=""):
        names = [self._need(junction)] if junction else list(self.state)
        out = {}
        for n in names:
            s, f = self.state[n], self.forecast[n]
            out[n] = {"now": s["congestion"], "pcu_per_hour": s["pcu_per_hour"],
                      "in_30_min": f["label"], "confidence": f["probability"],
                      "data_source": s["source"]}
        if junction:
            self.actions.append({"type": "focus", "junction": names[0]})
        return out

    def find_best_route(self, source, destination):
        r = core.best_route(self._need(source), self._need(destination))
        self.actions.append({"type": "route", "path": r["recommended_path"],
                             "alt_path": r["usual_path"], "minutes": r["recommended_minutes"],
                             "usual_minutes": r["usual_minutes"], "saved": r["minutes_saved"]})
        return r

    def create_green_corridor(self, source, destination="", vehicle="ambulance"):
        generic = {"", "hospital", "nearest hospital", "nearest", "any", "any hospital", "the hospital"}
        dest = None if (destination or "").lower().strip() in generic else self._need(destination)
        r = core.green_corridor(self._need(source), dest)
        vehicle = (vehicle or "ambulance").lower()
        vehicle = "fire" if "fire" in vehicle else "police" if "police" in vehicle else "ambulance"
        self.actions.append({"type": "corridor", "path": r["path"], "vehicle": vehicle,
                             "normal_path": r["normal_traffic_path"],
                             "eta": r["eta_with_corridor_min"], "eta_normal": r["eta_without_corridor_min"],
                             "saved": r["minutes_saved"], "signals": len(r["signal_schedule"])})
        return r

    def optimize_signals(self, junction):
        j = self._need(junction)
        r = core.compare_signals(j, self.state[j]["pcu_per_hour"])
        reduction = 100 * (r["fixed_wait_s"] - r["smart_wait_s"]) / max(r["fixed_wait_s"], 1e-9)
        self.actions.append({"type": "signals", "junction": j})
        return {"junction": j, "new_cycle_s": r["smart"]["cycle_s"],
                "new_green_s": r["smart"]["green_s"],
                "avg_wait_fixed_s": r["fixed_wait_s"], "avg_wait_smart_s": r["smart_wait_s"],
                "wait_reduction_pct": round(reduction, 1)}

    def call(self, name, args):
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_"):
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as e:  # return errors to the LLM so it can recover
            return {"error": str(e)}


# ----------------------------------------------------------------------------
# LLM agent
# ----------------------------------------------------------------------------

def run_llm_agent(message, tools, history, api_key, base_url, model, time_str):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=30)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
        junctions=", ".join(core.JUNCTIONS), hospitals=", ".join(core.HOSPITALS), time_str=time_str,
        weather=tools.weather_summary())}]
    messages += history[-6:]  # short memory of the conversation
    messages.append({"role": "user", "content": message})
    steps = []

    for _ in range(5):  # max 5 reasoning/tool rounds
        resp = client.chat.completions.create(model=model, messages=messages,
                                              tools=TOOLS, tool_choice="auto", temperature=0.2)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "(no answer)", steps
        # model_dump keeps provider extras (e.g. Gemini thought signatures) intact
        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = tools.call(tc.function.name, args)
            steps.append({"tool": tc.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)})
    return "I ran several checks - see the actions on the map.", steps


# ----------------------------------------------------------------------------
# Rule-based fallback agent (no API key needed)
# ----------------------------------------------------------------------------

def find_junctions(text):
    """Junction names mentioned in the text, in the order they appear."""
    t = text.lower()
    found = []
    keys = {n.lower(): n for n in core.JUNCTIONS}
    keys.update({"hitec": "Hitech City", "cyber": "Hitech City", "aig": "AIG Hospital",
                 "nims": "NIMS Hospital", "gandhi": "Gandhi Hospital", "osmania": "Osmania Hospital",
                 "jubilee": "Jubilee Hills", "gachi": "Gachibowli", "kphb": "Kukatpally",
                 "jntu": "Kukatpally", "lb nagar": "LB Nagar", "dsnr": "Dilsukhnagar",
                 "banjara": "Banjara Hills", "narayanguda": "Narayanaguda"})
    for key, name in keys.items():
        i = t.find(key)
        if i >= 0 and name not in [f[1] for f in found]:
            found.append((i, name))
    return [n for _, n in sorted(found)]


def run_fallback_agent(message, tools):
    t = message.lower()
    js = find_junctions(message)
    steps = []

    def use(name, **args):
        r = tools.call(name, args)
        steps.append({"tool": name, "args": args, "result": r})
        return r

    if re.search(r"weather|rain|raining|forecast|temperature|storm|flood", t) and not js:
        w = use("get_weather")
        nxt = "; ".join(f"{h['time']} {h['condition']} ({h['rain_chance_pct']}% rain)"
                        for h in w.get("next_hours", [])[:4])
        temp = f", {w['temp_c']} C" if w.get("temp_c") is not None else ""
        return (f"Weather in Hyderabad: {w.get('condition')}{temp} (source: {w.get('source', '?')}).\n"
                f"Traffic impact: {w.get('traffic_impact', 'unknown')} - the forecast and travel times "
                f"already include it." + (f"\nNext hours: {nxt}" if nxt else "")), steps

    route_words = re.search(r"route|go from|reach|travel|way to|divert", t)
    emergency = re.search(r"ambulance|emergency|fire|accident|police|patient|critical|injur", t)
    if emergency or ("hospital" in t and not route_words):
        vehicle = "fire" if "fire" in t else "police" if "police" in t else "ambulance"
        src = next((j for j in js if j not in core.HOSPITALS), "Jubilee Hills")
        rest = [j for j in js if j != src]
        # "fire engine from Paradise to Koti" -> second place is the destination
        dest = rest[0] if rest and (vehicle != "ambulance" or rest[0] in core.HOSPITALS) else ""
        r = use("create_green_corridor", source=src, destination=dest, vehicle=vehicle)
        if "error" in r:
            return r["error"], steps
        sched = ", ".join(f"{s['junction']} at +{s['turn_green_at_min']} min" for s in r["signal_schedule"]) or "none needed"
        diff = r["path"] != r["normal_traffic_path"]
        note = (f"Normal traffic should take {' -> '.join(r['normal_traffic_path'])} to avoid jams, but with "
                f"signals held green the {vehicle} takes the shorter road straight through.\n") if diff else ""
        return (note + f"Emergency ({vehicle}) - green corridor created: {' -> '.join(r['path'])}.\n"
                f"Signals turning green: {sched}.\n"
                f"ETA {r['eta_with_corridor_min']} min instead of {r['eta_without_corridor_min']} min "
                f"(saves {r['minutes_saved']} min). Cross traffic is held briefly at each junction."), steps

    if re.search(r"route|go from|reach|travel|way to|divert", t) and len(js) >= 2:
        r = use("find_best_route", source=js[0], destination=js[1])
        if r["minutes_saved"] > 0.5:
            return (f"Usual route {' -> '.join(r['usual_path'])} will be congested in 30 min "
                    f"({r['usual_minutes']} min).\nRecommended: {' -> '.join(r['recommended_path'])} "
                    f"({r['recommended_minutes']} min, saves {r['minutes_saved']} min).\n"
                    f"Diversion pushed to signboards / navigation apps."), steps
        return (f"The usual route {' -> '.join(r['usual_path'])} is already the fastest "
                f"({r['recommended_minutes']} min). No diversion needed."), steps

    if re.search(r"signal|timing|green time|optimi", t) and js:
        r = use("optimize_signals", junction=js[0])
        greens = ", ".join(f"{a} {g}s" for a, g in r["new_green_s"].items())
        return (f"New plan for {r['junction']}: cycle {r['new_cycle_s']} s, greens {greens}.\n"
                f"Average wait drops from {r['avg_wait_fixed_s']} s to {r['avg_wait_smart_s']} s "
                f"({r['wait_reduction_pct']}% less). Officer approval recommended."), steps

    # Default: status report
    r = use("get_traffic_status", junction=js[0] if js else "")
    if js:
        s = r[js[0]]
        reply = (f"{js[0]} is {s['now']} now ({s['pcu_per_hour']} PCU/hour, source: {s['data_source']}). "
                 f"Forecast for 30 min: {s['in_30_min']} ({int(s['confidence'] * 100)}% confidence).")
        if s["in_30_min"] == "High":
            o = use("optimize_signals", junction=js[0])
            reply += (f"\nI re-timed its signals: average wait {o['avg_wait_fixed_s']} s -> "
                      f"{o['avg_wait_smart_s']} s ({o['wait_reduction_pct']}% less).")
        return reply, steps
    hot = [n for n, s in r.items() if s["in_30_min"] == "High"]
    return (f"Junctions expected to be heavily congested in 30 min: {', '.join(hot) or 'none'}.\n"
            f"Ask me to optimise signals there, plan a route, or create an ambulance corridor."), steps


def run_agent(message, state, forecast, history, api_key="", base_url="", model="", time_str="",
              weather=None):
    """Returns (reply_text, steps, actions, mode)."""
    tools = TrafficTools(state, forecast, weather)
    if api_key:
        try:
            reply, steps = run_llm_agent(message, tools, history, api_key, base_url, model, time_str)
            return reply, steps, tools.actions, "LLM"
        except Exception as e:
            tools.actions.clear()
            reply, steps = run_fallback_agent(message, tools)
            note = f"{type(e).__name__}: {str(e)[:160]}"
            return reply + f"\n\n(LLM unavailable - {note}. Used rule-based agent.)", steps, tools.actions, "fallback"
    reply, steps = run_fallback_agent(message, tools)
    return reply, steps, tools.actions, "rule-based"
