from .config import *

from . import access
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
import osmnx as ox
import matplotlib.pyplot as plt
import seaborn as sns

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

def buildings_with_address_in_1km_and_plot(name, latitude, longitude):
    box_width = 0.02/2.2 # About 2.2 km
    box_height = 0.02/2.2
    north = latitude + box_height/2
    south = latitude - box_width/2
    west = longitude - box_width/2
    east = longitude + box_width/2

    tags = {
        "building": True,
        "addr:housenumber": True,
        "addr:street": True,
        "addr:postcode": True

    }
    graph = ox.graph_from_bbox(north, south, east, west)
    nodes, edges = ox.graph_to_gdfs(graph)
    area = ox.geocode_to_gdf(name)
    pois = ox.geometries_from_bbox(north, south, east, west, tags)

    buildings_with_address = pois.dropna(subset=['addr:housenumber', 'addr:street', 'addr:postcode'])
    buildings_with_address = buildings_with_address.to_crs("EPSG:27700")

    buildings_with_address['area_sqm'] = buildings_with_address['geometry'].apply(
        lambda geom: geom.area if isinstance(geom, (Polygon, MultiPolygon)) else 0
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the area footprint
    area.plot(ax=ax, facecolor="white")

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    # Define boundaries (adjust to your area's coordinates)
    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot buildings with addresses in green
    buildings_with_address = pois.dropna(subset=['addr:housenumber', 'addr:street', 'addr:postcode'])
    buildings_with_address.plot(ax=ax, color="black", alpha=1, markersize=10, label="With Address")

    # Plot buildings without addresses in red
    pois.plot(ax=ax, color="blue", alpha=0.5, markersize=10, label="Without Address")

    # Add a legend to distinguish between buildings with and without addresses
    ax.legend()

    # Final layout adjustments
    plt.tight_layout()
    plt.show()

    return buildings_with_address

def match_price_paid_with_buildings(conn, latitude, longitude, buildings_with_address):

    lat_min, lat_max = latitude - 0.009, latitude + 0.009
    lon_min, lon_max = longitude - 0.015, longitude + 0.015

    cur = conn.cursor()
    print('Selecting joined data')
    cur.execute("SELECT * FROM pp_data AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode WHERE latitude BETWEEN " + str(lat_min) + " AND " + str(lat_max) + " AND longitude BETWEEN " + str(lon_min) + " AND " + str(lon_max) + " AND date_of_transfer >= '2020-01-01';")
    rows = cur.fetchall()
    print('Finished selecting', rows)
    # Replace placeholders with actual lat/lon values for the SQL execution
    
    # Step 2: Convert result into a pandas DataFrame and remove duplicates
    df = pd.DataFrame(rows)
    df = df.loc[:, ~df.columns.duplicated()]
    buildings_with_address['normalized_address'] =  buildings_with_address['addr:street'].str.strip().str.lower() + ' ' + \
                               buildings_with_address['addr:postcode'].str.strip().str.lower()

    df['normalized_address'] =  df['street'].str.strip().str.lower() + ' ' + \
                                df['postcode'].str.strip().str.lower()

    merged_data = pd.merge(
        df,
        buildings_with_address,
        how='inner',  
        left_on=['normalized_address'],  
        right_on=['normalized_address']  
    )

    # Now check if primary or secondary addressable object matches the house number
    def match_with_house_number(row):
        """Check if either primary or secondary addressable object matches addr:housenumber."""
        primary_match = row['primary_addressable_object_name'] == row['addr:housenumber']
        secondary_match = row['secondary_addressable_object_name'] == row['addr:housenumber'] if pd.notnull(row['secondary_addressable_object_name']) else False
        return primary_match or secondary_match

    # Apply the matching function to the merged data
    merged_data['address_match'] = merged_data.apply(match_with_house_number, axis=1)
    return merged_data


def plot_price_to_area(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='area_sqm', y='price')

    # Set the title and labels
    plt.title('Price vs Area of Properties', fontsize=16)
    plt.xlabel('Area (m²)', fontsize=12)
    plt.ylabel('Price (£)', fontsize=12)

    # Optimize layout and display the plot
    plt.tight_layout()
    plt.show()

    # Calculate the correlation coefficient between 'price' and 'area_sqm'
    correlation = data['price'].corr(data['area_sqm'])
    print(f"Correlation between Price and Area: {correlation:.2f}")


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
