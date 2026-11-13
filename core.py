"""
core.py - Road graph, live traffic state, signal timing, routing and simulation.

Everything here is plain Python (NetworkX + NumPy) so it is easy to explain to judges.
Junction coordinates are approximate - adjust them in Google Maps if you want them exact.
"""

import json
import math
import os
import random

import networkx as nx
import numpy as np

# ----------------------------------------------------------------------------
# 1. The road network (graph)
# ----------------------------------------------------------------------------

# name: (lat, lon, busyness weight 0-1.2, is_hospital)
# Coordinates are approximate - check in Google Maps and adjust if needed.
JUNCTIONS = {
    # West - Cyberabad IT corridor
    "Gachibowli":       (17.4401, 78.3489, 0.95, False),
    "Kondapur":         (17.4615, 78.3636, 0.85, False),
    "Kothaguda":        (17.4583, 78.3712, 0.80, False),
    "Hitech City":      (17.4504, 78.3810, 1.10, False),
    "Mindspace":        (17.4416, 78.3802, 0.90, False),
    "Madhapur":         (17.4486, 78.3908, 0.95, False),
    "Kukatpally":       (17.4849, 78.4138, 1.00, False),
    "Jubilee Hills":    (17.4295, 78.4123, 0.85, False),
    "Banjara Hills":    (17.4190, 78.4480, 0.85, False),
    "Tolichowki":       (17.3990, 78.4130, 0.85, False),
    "Mehdipatnam":      (17.3960, 78.4400, 1.00, False),
    # Central
    "Ameerpet":         (17.4375, 78.4483, 1.00, False),
    "Panjagutta":       (17.4265, 78.4510, 1.05, False),
    "Begumpet":         (17.4440, 78.4620, 0.80, False),
    "Khairatabad":      (17.4116, 78.4609, 0.85, False),
    "Lakdikapul":       (17.4040, 78.4650, 0.80, False),
    "Narayanaguda":     (17.3990, 78.4870, 0.90, False),
    "Abids":            (17.3920, 78.4760, 0.90, False),
    "Koti":             (17.3850, 78.4860, 1.00, False),
    "Charminar":        (17.3616, 78.4747, 0.90, False),
    # North-east / east
    "Paradise":         (17.4436, 78.4870, 0.85, False),
    "Secunderabad":     (17.4337, 78.5016, 0.95, False),
    "Tarnaka":          (17.4270, 78.5390, 0.80, False),
    "Uppal":            (17.4018, 78.5602, 0.85, False),
    "Dilsukhnagar":     (17.3687, 78.5247, 0.95, False),
    "LB Nagar":         (17.3474, 78.5513, 1.00, False),
    # Hospitals
    "AIG Hospital":     (17.4430, 78.3640, 0.50, True),
    "NIMS Hospital":    (17.4196, 78.4523, 0.55, True),
    "Gandhi Hospital":  (17.4243, 78.5034, 0.55, True),
    "Osmania Hospital": (17.3727, 78.4747, 0.55, True),
}

ROADS = [
    # IT corridor
    ("Gachibowli", "AIG Hospital"), ("Gachibowli", "Kondapur"), ("Gachibowli", "Mindspace"),
    ("AIG Hospital", "Kondapur"), ("AIG Hospital", "Mindspace"), ("Kondapur", "Kothaguda"),
    ("Kothaguda", "Hitech City"), ("Kothaguda", "Madhapur"), ("Hitech City", "Mindspace"),
    ("Hitech City", "Madhapur"), ("Mindspace", "Jubilee Hills"), ("Madhapur", "Jubilee Hills"),
    ("Hitech City", "Kukatpally"), ("Gachibowli", "Tolichowki"),
    # West <-> centre
    ("Jubilee Hills", "Banjara Hills"), ("Jubilee Hills", "Tolichowki"),
    ("Tolichowki", "Mehdipatnam"), ("Banjara Hills", "Mehdipatnam"),
    ("Banjara Hills", "Panjagutta"), ("Kukatpally", "Ameerpet"),
    # Centre
    ("Panjagutta", "Ameerpet"), ("Panjagutta", "NIMS Hospital"), ("NIMS Hospital", "Khairatabad"),
    ("Panjagutta", "Khairatabad"), ("Ameerpet", "Begumpet"), ("Khairatabad", "Lakdikapul"),
    ("Mehdipatnam", "Lakdikapul"), ("Lakdikapul", "Abids"), ("Lakdikapul", "Narayanaguda"),
    ("Narayanaguda", "Abids"), ("Narayanaguda", "Koti"), ("Abids", "Koti"),
    ("Abids", "Osmania Hospital"), ("Koti", "Osmania Hospital"), ("Osmania Hospital", "Charminar"),
    ("Abids", "Charminar"),
    # North-east / east
    ("Begumpet", "Paradise"), ("Paradise", "Secunderabad"), ("Secunderabad", "Gandhi Hospital"),
    ("Gandhi Hospital", "Narayanaguda"), ("Secunderabad", "Narayanaguda"),
    ("Secunderabad", "Tarnaka"), ("Narayanaguda", "Tarnaka"), ("Tarnaka", "Uppal"),
    ("Uppal", "LB Nagar"), ("LB Nagar", "Dilsukhnagar"), ("Dilsukhnagar", "Koti"),
    ("Charminar", "Dilsukhnagar"),
]

FREE_SPEED_KMPH = 35          # typical free-flow urban speed
JUNCTION_CAPACITY_PCU = 3300  # PCU/hour a junction handles at utilisation 1.0
SATURATION_FLOW = 4500        # PCU/hour per approach while green
LOST_TIME_PER_PHASE = 4       # seconds (amber + all-red)

# Passenger Car Units - approximate IRC:106 values
PCU = {"two_wheeler": 0.5, "auto": 1.2, "car": 1.0, "bus": 2.2, "truck": 2.2}
# Typical Indian urban mix (share of vehicle COUNT)
VEHICLE_MIX = {"two_wheeler": 0.55, "car": 0.25, "auto": 0.12, "bus": 0.04, "truck": 0.04}


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def build_graph():
    g = nx.Graph()
    for name, (lat, lon, weight, hosp) in JUNCTIONS.items():
        g.add_node(name, lat=lat, lon=lon, weight=weight, hospital=hosp)
    for u, v in ROADS:
        a, b = JUNCTIONS[u][:2], JUNCTIONS[v][:2]
        km = round(haversine_km(a, b) * 1.3, 2)  # 1.3 = roads are not straight lines
        g.add_edge(u, v, km=km, free_min=km / FREE_SPEED_KMPH * 60)
    return g


GRAPH = build_graph()
HOSPITALS = [n for n, d in GRAPH.nodes(data=True) if d["hospital"]]


def match_junction(text):
    """Fuzzy-match free text (from the user or the LLM) to a junction name."""
    if not text:
        return None
    t = text.lower().strip()
    for name in JUNCTIONS:
        if name.lower() == t:
            return name
    for name in JUNCTIONS:
        if name.lower() in t or t in name.lower():
            return name
    aliases = {"hitec": "Hitech City", "cyber": "Hitech City", "aig": "AIG Hospital",
               "nims": "NIMS Hospital", "gandhi": "Gandhi Hospital", "osmania": "Osmania Hospital",
               "jubilee": "Jubilee Hills", "checkpost": "Jubilee Hills", "gachi": "Gachibowli",
               "raheja": "Mindspace", "kphb": "Kukatpally", "jntu": "Kukatpally",
               "secunderabad": "Secunderabad", "secbad": "Secunderabad", "lbnagar": "LB Nagar",
               "l.b. nagar": "LB Nagar", "l b nagar": "LB Nagar", "dsnr": "Dilsukhnagar",
               "banjara": "Banjara Hills", "mehdipatnam": "Mehdipatnam", "narayanguda": "Narayanaguda"}
    for key, name in aliases.items():
        if key in t:
            return name
    return None


# ----------------------------------------------------------------------------
# 2. Live traffic state (simulated, with real YOLO counts plugged in if present)
# ----------------------------------------------------------------------------

def daily_profile(hour):
    """Traffic demand shape over a day: morning peak ~9:30, evening peak ~18:30."""
    morning = math.exp(-((hour - 9.5) ** 2) / (2 * 1.3 ** 2))
    evening = math.exp(-((hour - 18.5) ** 2) / (2 * 1.6 ** 2))
    base = 0.25 if 6 <= hour <= 23 else 0.08
    return min(1.0, base + 0.75 * morning + 0.8 * evening)


def utilisation(junction, hour, day_of_week, noise_seed=None):
    weight = JUNCTIONS[junction][2]
    weekend = 0.75 if day_of_week >= 5 else 1.0
    rng = random.Random(noise_seed) if noise_seed is not None else random
    noise = rng.uniform(-0.06, 0.06)
    return max(0.05, weight * weekend * daily_profile(hour) + noise)


def congestion_label(util):
    if util < 0.6:
        return "Low"
    if util < 0.85:
        return "Medium"
    return "High"


def counts_from_pcu(pcu_per_hour):
    """Split a PCU/hour total into vehicle counts per class using the typical mix."""
    avg_pcu = sum(VEHICLE_MIX[k] * PCU[k] for k in VEHICLE_MIX)
    total_vehicles = pcu_per_hour / avg_pcu
    return {k: int(round(total_vehicles * share)) for k, share in VEHICLE_MIX.items()}


def pcu_from_counts(counts):
    return sum(PCU.get(k, 1.0) * v for k, v in counts.items())


def load_cameras(root="outputs"):
    """All camera results produced by vision.py: {junction: data}.
    Reads outputs/<junction>/counts.json (one folder per camera) and the older
    single-camera layout outputs/counts.json."""
    cameras = {}
    if not os.path.isdir(root):
        return cameras
    folders = [root] + [os.path.join(root, d) for d in sorted(os.listdir(root))
                        if os.path.isdir(os.path.join(root, d))]
    for folder in folders:
        path = os.path.join(folder, "counts.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        name = match_junction(data.get("junction", ""))
        if name and name not in cameras:
            data["junction"] = name
            data["dir"] = folder
            cameras[name] = data
    return cameras


def current_state(hour, day_of_week, cameras=None, weather_factor=1.0):
    """Current traffic at every junction: counts, PCU/hour, utilisation, label.
    weather_factor > 1 when it rains (roads slower and more congested)."""
    state = {}
    for name in JUNCTIONS:
        seed = f"{name}-{round(hour * 4)}-{day_of_week}"  # stable between reruns
        util = utilisation(name, hour, day_of_week, noise_seed=seed) * weather_factor
        pcu = util * JUNCTION_CAPACITY_PCU
        counts = counts_from_pcu(pcu)
        source = "simulated"
        if cameras and name in cameras:
            counts = cameras[name]["counts_per_hour"]
            # A short clip can over/under-state an hourly rate - keep it in a sane range
            util = min(max(pcu_from_counts(counts) / JUNCTION_CAPACITY_PCU, 0.05), 1.25)
            pcu = util * JUNCTION_CAPACITY_PCU
            source = "camera (YOLO)"
        state[name] = {
            "counts_per_hour": counts,
            "pcu_per_hour": round(pcu),
            "utilisation": round(util, 2),
            "congestion": congestion_label(util),
            "source": source,
        }
    return state


# ----------------------------------------------------------------------------
# 3. Signal timing (Webster's method) and before/after simulation
# ----------------------------------------------------------------------------

def approach_flows(junction, pcu_per_hour):
    """Split junction demand over 4 approaches (N, E, S, W) - uneven, like real roads."""
    rng = random.Random(junction)
    shares = np.array([rng.uniform(0.15, 0.45) for _ in range(4)])
    shares = shares / shares.sum()
    return dict(zip(["North", "East", "South", "West"], (shares * pcu_per_hour).round()))


def fixed_plan():
    """What most junctions run today: same green for everyone, 120 s cycle."""
    cycle = 120
    green = (cycle - 4 * LOST_TIME_PER_PHASE) / 4
    return {"cycle_s": cycle, "green_s": {a: green for a in ["North", "East", "South", "West"]}}


def webster_plan(flows):
    """Webster's optimal cycle; green split proportional to each approach's flow ratio."""
    y = {a: f / SATURATION_FLOW for a, f in flows.items()}
    Y = min(sum(y.values()), 0.92)
    L = 4 * LOST_TIME_PER_PHASE
    cycle = (1.5 * L + 5) / (1 - Y)
    cycle = float(np.clip(cycle, 60, 150))
    total_y = sum(y.values()) or 1
    green = {a: round((cycle - L) * y[a] / total_y, 1) for a in flows}
    # Every approach gets at least 10 s of green (pedestrian safety)
    for a in green:
        green[a] = max(green[a], 10.0)
    cycle = round(sum(green.values()) + L, 1)
    return {"cycle_s": cycle, "green_s": green}


def simulate_delay(flows, plan, minutes=60):
    """Fluid queue simulation, 1 s steps. Returns average wait per vehicle (seconds)."""
    approaches = list(flows)
    arrive = np.array([flows[a] / 3600 for a in approaches])
    serve = SATURATION_FLOW / 3600
    greens = [plan["green_s"][a] for a in approaches]
    # Build the phase schedule: approach i is green for greens[i], then lost time
    schedule = []
    for i, g in enumerate(greens):
        schedule += [i] * int(round(g)) + [-1] * LOST_TIME_PER_PHASE
    cycle = len(schedule)
    queue = np.zeros(len(approaches))
    queue_seconds = 0.0
    for t in range(minutes * 60):
        queue += arrive
        active = schedule[t % cycle]
        if active >= 0:
            queue[active] = max(0.0, queue[active] - serve)
        queue_seconds += queue.sum()
    total_arrivals = arrive.sum() * minutes * 60
    return queue_seconds / max(total_arrivals, 1e-9)


def compare_signals(junction, pcu_per_hour):
    flows = approach_flows(junction, pcu_per_hour)
    fixed, smart = fixed_plan(), webster_plan(flows)
    return {
        "flows": flows,
        "fixed": fixed,
        "smart": smart,
        "fixed_wait_s": round(simulate_delay(flows, fixed), 1),
        "smart_wait_s": round(simulate_delay(flows, smart), 1),
    }


# ----------------------------------------------------------------------------
# 4. Routing on PREDICTED travel times
# ----------------------------------------------------------------------------

def travel_factor(util):
    """BPR-style slowdown: almost free below 60% utilisation, ~2x at 100%."""
    return 1 + 1.0 * util ** 4


EMERGENCY_DELAY_SHARE = 0.2  # with signals held green and traffic pulling aside, an emergency
                             # vehicle suffers only ~20% of the congestion delay normal cars do


def set_edge_times(predicted_util):
    """predicted_util: {junction: utilisation 30 min ahead}.
    pred_min = travel time for normal traffic (full congestion delay)
    em_min   = travel time for an emergency vehicle on a green corridor"""
    for u, v, d in GRAPH.edges(data=True):
        worst = max(predicted_util.get(u, 0.5), predicted_util.get(v, 0.5))
        d["pred_min"] = d["free_min"] * travel_factor(worst)
        d["em_min"] = d["free_min"] * (1 + EMERGENCY_DELAY_SHARE * (travel_factor(worst) - 1))


def path_minutes(path, attr):
    return sum(GRAPH[a][b][attr] for a, b in zip(path, path[1:]))


def road_view(*keep):
    """Graph without hospital nodes (traffic shouldn't be routed through a hospital),
    except the ones we start or end at."""
    return nx.subgraph_view(GRAPH, filter_node=lambda n: not GRAPH.nodes[n]["hospital"] or n in keep)


def best_route(source, destination):
    g = road_view(source, destination)
    fastest = nx.shortest_path(g, source, destination, weight="pred_min")
    shortest = nx.shortest_path(g, source, destination, weight="km")
    return {
        "recommended_path": fastest,
        "recommended_minutes": round(path_minutes(fastest, "pred_min"), 1),
        "usual_path": shortest,
        "usual_minutes": round(path_minutes(shortest, "pred_min"), 1),
        "minutes_saved": round(path_minutes(shortest, "pred_min") - path_minutes(fastest, "pred_min"), 1),
    }


def green_corridor(source, destination=None):
    """Emergency route. Normal traffic avoids jams; an emergency vehicle gets every signal
    green and cars pull aside, so congestion hurts it far less - its best route is usually
    the SHORTEST one, straight through busy junctions that normal traffic should avoid."""
    if destination is None:
        destination = min(HOSPITALS, key=lambda h: nx.shortest_path_length(
            road_view(source, h), source, h, weight="em_min"))
    g = road_view(source, destination)
    path = nx.shortest_path(g, source, destination, weight="em_min")
    normal_path = nx.shortest_path(g, source, destination, weight="pred_min")
    eta_normal = path_minutes(normal_path, "pred_min")   # best a normal car could do
    eta_corridor = path_minutes(path, "em_min")
    schedule, t = [], 0.0
    for a, b in zip(path, path[1:]):
        t += GRAPH[a][b]["em_min"]
        if b != destination:
            schedule.append({"junction": b, "turn_green_at_min": round(max(t - 0.5, 0), 1)})
    return {
        "path": path,
        "normal_traffic_path": normal_path,
        "destination": destination,
        "eta_without_corridor_min": round(eta_normal, 1),
        "eta_with_corridor_min": round(eta_corridor, 1),
        "minutes_saved": round(eta_normal - eta_corridor, 1),
        "distance_km": round(sum(GRAPH[a][b]["km"] for a, b in zip(path, path[1:])), 1),
        "signal_schedule": schedule,
    }
