
'''
Create a basic map with Leaflet using Folium.
Add a Choropleth Map layer.
Add a Heatmap.
Place Markers on the map.

'''
import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import requests
import os
import json

# URL for the Natural Earth GeoJSON file hosted on GitHub
url = "https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/ne_110m_admin_0_countries.geojson"
geojson_path = "ne_110m_admin_0_countries.geojson"

# Download and save the GeoJSON file if it doesn't exist
if not os.path.exists(geojson_path):
    try:
        print("Downloading Natural Earth GeoJSON data...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Save the GeoJSON file
        with open(geojson_path, "wb") as file:
            file.write(response.content)
        print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
else:
    print("GeoJSON file already exists. Skipping download.")

# Load the GeoJSON data
if os.path.exists(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    # Extract country names and population estimates into a DataFrame
    country_data = []
    for feature in geojson_data["features"]:
        country_name = feature["properties"]["NAME"]
        pop_est = feature["properties"].get("POP_EST", 0)  # Default to 0 if population estimate is missing
        country_data.append({"Country": country_name, "Population": pop_est})
    
    # Create a DataFrame
    country_df = pd.DataFrame(country_data)

    # Create a map centered around a specific location
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=2, tiles="CartoDB positron")

    # Sample data for markers
    data = pd.DataFrame({
        'name': ['Place A', 'Place B', 'Place C','Bangalore'],
        'lat': [37.7749, 37.7849, 37.7649,12.9716],
        'lon': [-122.4194, -122.4094, -122.4294,77.5946]
    })

    # Add markers
    for _, row in data.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['name'],
            icon=folium.Icon(color='blue')
        ).add_to(m)

    # Add Marker Cluster
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in data.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['name']
        ).add_to(marker_cluster)

    # Sample data for heatmap
    heat_data = [
        [37.7749, -122.4194, 0.9],
        [37.7849, -122.4094, 0.7],
        [37.7649, -122.4294, 0.6],
        [35.7648, -128.4294,0.8]
    ]

    # Add heatmap layer
    HeatMap(heat_data).add_to(m)

    # Choropleth map: shading countries by population estimate
    folium.Choropleth(
        geo_data=geojson_data,
        name="choropleth",
        data=country_df,
        columns=["Country", "Population"],
        key_on="feature.properties.NAME",
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Population Estimate",
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map to HTML file
    m.save("map_example.html")
    print("Map created and saved as map_example.html.")
else:
    print("GeoJSON file could not be loaded. Please check the download step.")
