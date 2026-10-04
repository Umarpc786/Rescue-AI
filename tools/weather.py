import requests

def get_weather_data(lat: float, lon: float) -> dict:
    """Fetch live weather context using Open-Meteo free API."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json().get("current_weather", {})
            return {
                "temperature_c": data.get("temperature"),
                "windspeed_kmh": data.get("windspeed"),
                "weather_code": data.get("weathercode"),
                "status": "Success"
            }
        return {"status": "Error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "Error", "message": str(e)}
