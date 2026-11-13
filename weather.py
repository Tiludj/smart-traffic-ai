"""
weather.py - Live Hyderabad weather from Open-Meteo (free, no API key) and its effect on traffic.

Rain slows Indian city traffic a lot (waterlogging, slower speeds, more two-wheelers stopping),
so the app scales congestion by a weather factor.
"""

import json
import urllib.request

HYDERABAD = (17.385, 78.4867)

# condition -> how much busier/slower roads get (1.0 = no effect)
TRAFFIC_FACTOR = {
    "Clear": 1.00, "Partly cloudy": 1.00, "Cloudy": 1.00, "Fog": 1.10,
    "Drizzle": 1.08, "Light rain": 1.15, "Rain": 1.25, "Heavy rain": 1.40, "Thunderstorm": 1.40,
}
ICON = {"Clear": "☀️", "Partly cloudy": "⛅", "Cloudy": "☁️", "Fog": "🌫️", "Drizzle": "🌦️",
        "Light rain": "🌦️", "Rain": "🌧️", "Heavy rain": "🌧️", "Thunderstorm": "⛈️"}


def condition_from_code(code):
    """WMO weather code (used by Open-Meteo) -> simple condition."""
    if code == 0:
        return "Clear"
    if code in (1, 2):
        return "Partly cloudy"
    if code == 3:
        return "Cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57):
        return "Drizzle"
    if code in (61, 80):
        return "Light rain"
    if code in (63, 66, 81):
        return "Rain"
    if code in (65, 67, 82):
        return "Heavy rain"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return "Cloudy"


def fetch_live(timeout=6):
    """Current weather + next 6 hours of rain chance. Raises on network errors."""
    lat, lon = HYDERABAD
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           "&current=temperature_2m,precipitation,weather_code,wind_speed_10m"
           "&hourly=precipitation_probability,weather_code,temperature_2m"
           "&forecast_hours=6&timezone=Asia%2FKolkata")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = json.load(r)
    cur = data["current"]
    cond = condition_from_code(int(cur["weather_code"]))
    hourly = data.get("hourly", {})
    next_hours = [
        {"time": t[-5:], "rain_chance_pct": p, "condition": condition_from_code(int(c)), "temp_c": tc}
        for t, p, c, tc in zip(hourly.get("time", []), hourly.get("precipitation_probability", []),
                               hourly.get("weather_code", []), hourly.get("temperature_2m", []))
    ]
    return make_weather(cond, temp_c=cur.get("temperature_2m"), rain_mm=cur.get("precipitation"),
                        wind_kmh=cur.get("wind_speed_10m"), next_hours=next_hours,
                        source="Live - Open-Meteo")


def make_weather(condition, temp_c=None, rain_mm=None, wind_kmh=None, next_hours=None,
                 source="Simulated"):
    factor = TRAFFIC_FACTOR.get(condition, 1.0)
    return {
        "condition": condition,
        "icon": ICON.get(condition, ""),
        "temp_c": temp_c,
        "rain_mm": rain_mm,
        "wind_kmh": wind_kmh,
        "next_hours": next_hours or [],
        "traffic_factor": factor,
        "traffic_impact": f"+{round((factor - 1) * 100)}% congestion" if factor > 1 else "no effect",
        "source": source,
    }
