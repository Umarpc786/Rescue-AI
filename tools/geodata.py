import requests

def get_location_data(location_name: str) -> dict:
    """Geocode a location using OpenStreetMap Nominatim API."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": location_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "RescueAI-Emergency-System/1.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200 and len(response.json()) > 0:
            place = response.json()[0]
            return {
                "display_name": place.get("display_name"),
                "latitude": float(place.get("lat")),
                "longitude": float(place.get("lon")),
                "type": place.get("type", "unknown"),
                "status": "Success"
            }
        return {"status": "Not Found", "display_name": location_name, "latitude": 29.3956, "longitude": 71.6836}
    except Exception as e:
        return {"status": "Error", "message": str(e), "latitude": 29.3956, "longitude": 71.6836}
