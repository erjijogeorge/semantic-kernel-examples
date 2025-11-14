"""
WeatherPlugin - Demonstrates API-like function calls.

This plugin simulates weather API calls to show how AI can interact with external services.
In a real application, these would make actual HTTP requests to weather APIs.
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated
import random
from datetime import datetime


class WeatherPlugin:
    """A plugin that provides weather information (simulated)."""
    
    # Simulated weather data
    _weather_data = {
        "new york": {"temp": 72, "condition": "Partly Cloudy", "humidity": 65},
        "london": {"temp": 59, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp": 68, "condition": "Sunny", "humidity": 55},
        "paris": {"temp": 64, "condition": "Cloudy", "humidity": 70},
        "sydney": {"temp": 77, "condition": "Sunny", "humidity": 60},
        "mumbai": {"temp": 86, "condition": "Humid", "humidity": 85},
        "dubai": {"temp": 95, "condition": "Hot and Sunny", "humidity": 45},
    }
    
    @kernel_function(
        name="get_current_weather",
        description="Gets the current weather for a specified city. Returns temperature, condition, and humidity."
    )
    def get_current_weather(
        self,
        city: Annotated[str, "The city name to get weather for"],
    ) -> Annotated[str, "Weather information including temperature, condition, and humidity"]:
        """Get current weather for a city."""
        city_lower = city.lower()
        
        if city_lower in self._weather_data:
            weather = self._weather_data[city_lower]
            result = (
                f"Weather in {city.title()}:\n"
                f"Temperature: {weather['temp']}°F\n"
                f"Condition: {weather['condition']}\n"
                f"Humidity: {weather['humidity']}%"
            )
        else:
            # Simulate random weather for unknown cities
            temp = random.randint(50, 90)
            conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
            condition = random.choice(conditions)
            humidity = random.randint(40, 90)
            result = (
                f"Weather in {city.title()}:\n"
                f"Temperature: {temp}°F\n"
                f"Condition: {condition}\n"
                f"Humidity: {humidity}%"
            )
        
        print(f"[WeatherPlugin.get_current_weather] Retrieved weather for {city}")
        return result
    
    @kernel_function(
        name="get_forecast",
        description="Gets a 3-day weather forecast for a specified city."
    )
    def get_forecast(
        self,
        city: Annotated[str, "The city name to get forecast for"],
    ) -> Annotated[str, "3-day weather forecast"]:
        """Get weather forecast."""
        print(f"[WeatherPlugin.get_forecast] Getting forecast for {city}")
        
        # Simulate 3-day forecast
        forecast = f"3-Day Forecast for {city.title()}:\n"
        conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Stormy"]
        
        for day in range(1, 4):
            temp_high = random.randint(65, 85)
            temp_low = random.randint(50, 65)
            condition = random.choice(conditions)
            forecast += f"Day {day}: High {temp_high}°F, Low {temp_low}°F - {condition}\n"
        
        return forecast
    
    @kernel_function(
        name="compare_weather",
        description="Compares the weather between two cities."
    )
    def compare_weather(
        self,
        city1: Annotated[str, "The first city"],
        city2: Annotated[str, "The second city"],
    ) -> Annotated[str, "Comparison of weather between two cities"]:
        """Compare weather between two cities."""
        print(f"[WeatherPlugin.compare_weather] Comparing {city1} and {city2}")
        
        weather1 = self.get_current_weather(city1)
        weather2 = self.get_current_weather(city2)
        
        result = f"Weather Comparison:\n\n{weather1}\n\nvs\n\n{weather2}"
        return result
    
    @kernel_function(
        name="should_bring_umbrella",
        description="Determines if you should bring an umbrella based on the weather in a city."
    )
    def should_bring_umbrella(
        self,
        city: Annotated[str, "The city to check weather for"],
    ) -> Annotated[str, "Recommendation on whether to bring an umbrella"]:
        """Check if umbrella is needed."""
        print(f"[WeatherPlugin.should_bring_umbrella] Checking for {city}")
        
        city_lower = city.lower()
        if city_lower in self._weather_data:
            condition = self._weather_data[city_lower]["condition"].lower()
            if "rain" in condition or "storm" in condition:
                return f"Yes, bring an umbrella! It's {self._weather_data[city_lower]['condition']} in {city.title()}."
            else:
                return f"No umbrella needed. It's {self._weather_data[city_lower]['condition']} in {city.title()}."
        else:
            return f"Weather data not available for {city}, but it's always good to be prepared!"

