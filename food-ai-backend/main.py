import os
from groq import Groq
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import asyncio 

from schemas import DeciderRequest, DeciderResponse

load_dotenv()

app = FastAPI(title="AI Food Decider Core")

# Enable CORS so your future Next.js frontend can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# async def get_weather(lat: float, lon: float) -> str:
#     """Fetches the current weather condition to feed into the AI mood generator."""
#     url = f"https://openweathermap.org{lat}&lon={lon}&appid={os.getenv('OPENWEATHER_API_KEY')}&units=metric"
#     async with httpx.AsyncClient() as client:
#         try:
#             response = await client.get(url)
#             if response.status_code == 200:
#                 data = response.json()
#                 return f"{data['weather'][0]['main']} ({data['main']['temp']}°C)"
#             return "Clear (Unknown Temp)"
#         except Exception:
#             return "Unknown Weather"

async def get_weather(lat: float, lon: float) -> str:
    """Fetches weather condition and verifies local .env token loading."""
    # Force a local reload of the environment file to prevent caching issues
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # Pre-flight check inside the function to warn you if python is blind to the key
    if not api_key:
        print("⚠️ BACKEND ERROR: OPENWEATHER_API_KEY is completely missing from your environment variables!")
        return "Key Missing From .env"
        
    url = f"https://api.openweathermap.org/data/4.0/onecall/current?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                # Accessing the first dictionary inside the weather list array correctly
                condition = data["data"][0]["weather"][0]["main"]
                temp = data['data'][0]['temp']
                print(f"✅ Weather API Connected successfully! Current Weather: {condition} ({temp}°C)")
                return f"{condition} ({temp}°C)"
            
            # Print the exact rejection status code from OpenWeatherMap
            print(f"\n⚠️ Weather API Server Response Error Code: {response.status_code}")
            print(f"⚠️ Response content payload: {response.text}")
            return "Clear (Authentication Processing)"
            
        except Exception as e:
            print(f"⚠️ Network connection wrapper failed: {e}")
            return "Unknown Weather"


async def get_nearby_restaurants(lat: float, lon: float) -> list:
    """Queries Google Places for open restaurants within a 3km radius."""
    # Note: TextSearch or NearbySearch can be used. This uses Google's standard Place Search format.
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": 3000,
        "type": "restaurant",
        "opennow": "true",
        "key": os.getenv("GOOGLE_PLACES_API_KEY")
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            if response.status_code == 200:
                results = response.json().get("results", [])
                # Truncate data to save LLM tokens and remove fluff
                cleaned_places = []
                for place in results[:15]:  # Take top 15 matches to filter
                    cleaned_places.append({
                        "name": place.get("name"),
                        "address": place.get("vicinity"),
                        "rating": place.get("rating", 0.0),
                        "price_level": place.get("price_level", 2),
                        "types": place.get("types", [])
                    })
                return cleaned_places
            return []
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Google Places error: {str(e)}")

@app.post("/api/decide", response_model=DeciderResponse)
async def decide_food(payload: DeciderRequest):
    # 1. Gather context data concurrently
    weather = await get_weather(payload.latitude, payload.longitude)
    restaurants = await get_nearby_restaurants(payload.latitude, payload.longitude)
    
    if not restaurants:
        raise HTTPException(status_code=404, detail="No open restaurants found nearby via Google Maps.")

    # return {
    #     "weather_condition": weather,
    #     "restaurants": restaurants
    # }
    # 2. Build the system prompt to guide the AI decision engine
    system_prompt = (
        "You are an elite local culinary AI concierge designed to fight decision fatigue. "
        "Your job is to analyze the user's current physical environment (weather), their mood, "
        "and a raw list of open nearby restaurants, then select the absolute best 3 matches."
    )

    user_content = f"""
    Current Weather: {weather}
    User Mood/Craving: {payload.mood_input}
    Target Budget Level: {payload.budget}
    
    Available Restaurants Near User:
    {restaurants}
    
    Filter and rank these options. Select the top 3 best fits.
    """

    # 3. Call OpenAI using structured output format to enforce our schema
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # High-performance, smart reasoning model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}  # Forces a clean JSON mapping response
        )
        
        return completion.choices[0].message.content
        # completion = openai_client.beta.chat.completions.parse(
        #     model="gpt-4o-mini",  # Highly cost-effective and perfectly supports structured output parsing
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_content}
        #     ],
        #     response_format=DeciderResponse,
        # )
        
        # The parsed object natively adheres completely to our DeciderResponse Pydantic Model
        # ai_decision = completion.choices[0].message.parsed
        # return ai_decision

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Decision Engine failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # 2. Define an internal test function to isolate your API connectivity checks
    async def run_preflight_checks():
        print("\n--- Running Core AI Preflight Checks ---")
        try:
            # Example coordinates for Vancouver, BC
            test_weather = await get_weather(49.2827, -123.1207)
            print(f"✅ Weather API Connected successfully!")
            print(f"   Current Test Weather Data: {test_weather}")
        except Exception as e:
            print(f"❌ Weather API Check Failed: {e}")
        print("---------------------------------------\n")

    # 3. Use asyncio to cleanly execute, await, and print the result
    # asyncio.run(run_preflight_checks())

    # 4. Start the blocking web server after your tests finish executing
    print("Starting FastAPI Application Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
