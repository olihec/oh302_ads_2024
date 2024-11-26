from .config import *

from . import access
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
import osmnx as ox
import matplotlib.pyplot as plt
import seaborn as sns
import math

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
    cur.execute("SELECT * FROM prices_coordinates_data WHERE latitude BETWEEN " + str(lat_min) + " AND " + str(lat_max) + " AND longitude BETWEEN " + str(lon_min) + " AND " + str(lon_max) + " AND date_of_transfer >= '2020-01-01';")
    rows = cur.fetchall()
    print('Finished selecting', rows)
    # Replace placeholders with actual lat/lon values for the SQL execution
    columns = [desc[0] for desc in cur.description]
    # Step 2: Convert result into a pandas DataFrame and remove duplicates
    df = pd.DataFrame(rows, columns=columns)
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

def get_osm_features_from_location_within_1km(center_lat, center_lon, tags):
    dist_per_degree_lat = 111.1  # Approximate km per degree latitude
    area_km2 = 4
    side_length = math.sqrt(area_km2)

    lat_offset = side_length / (2 * dist_per_degree_lat)
    lon_offset = side_length / (2 * (dist_per_degree_lat * math.cos(math.radians(center_lat))))

    vertices = [(center_lon - lon_offset, center_lat - lat_offset),
                (center_lon + lon_offset, center_lat - lat_offset),
                (center_lon + lon_offset, center_lat + lat_offset),
                (center_lon - lon_offset, center_lat + lat_offset)]

    square_polygon = Polygon(vertices)

    osm_features = ox.features_from_polygon(square_polygon, tags)



    return osm_features

def get_osm_features_from_codes(connection, oa_codes, tags):

    oa_codes_str = ','.join([f"'{code}'" for code in oa_codes])  # Format codes for SQL query
    query = f"SELECT `OA21CD`, `LAT`, `LONG`, `Shape__Area` FROM oa_geographies_data WHERE `OA21CD` IN ({oa_codes_str})"
    oa_data = pd.read_sql(query, connection)

    osm_features_df = pd.DataFrame()

    dist_per_degree_lat = 111.1  # Approximate km per degree latitude

    for _, row in oa_data.iterrows():
        oa_code = row['OA21CD']
        center_lat = row['LAT']
        center_lon = row['LONG']
        area_m2 = row['Shape__Area']

        area_km2 = area_m2 / 1000000
        side_length = math.sqrt(area_km2)

        lat_offset = side_length / (2 * dist_per_degree_lat)
        lon_offset = side_length / (2 * (dist_per_degree_lat * math.cos(math.radians(center_lat))))

        vertices = [(center_lon - lon_offset, center_lat - lat_offset),
                    (center_lon + lon_offset, center_lat - lat_offset),
                    (center_lon + lon_offset, center_lat + lat_offset),
                    (center_lon - lon_offset, center_lat + lat_offset)]

        square_polygon = Polygon(vertices)
        try:

            new_osm_features = ox.features_from_polygon(square_polygon, tags)
            new_osm_features['OA21CD'] = oa_code

            osm_features_df = pd.concat([osm_features_df, new_osm_features], ignore_index=True)
        except ox._errors.InsufficientResponseError:
            print(f"Warning: No OSM features found for OA code '{oa_code}'. Skipping.")
    
    return osm_features_df

def get_osm_features_in_1km(connection, oa_codes, tags):
    oa_codes_str = ','.join([f"'{code}'" for code in oa_codes])  # Format codes for SQL query
    query = f"SELECT `OA21CD`, `LAT`, `LONG`, `Shape__Area` FROM oa_geographies_data WHERE `OA21CD` IN ({oa_codes_str})"
    oa_data = pd.read_sql(query, connection)

    osm_features_df = pd.DataFrame()

    dist_per_degree_lat = 111.1  # Approximate km per degree latitude

    for _, row in oa_data.iterrows():
        oa_code = row['OA21CD']
        center_lat = row['LAT']
        center_lon = row['LONG']

        area_km2 = 4 #4km to get sided of 2 km each, do edge is 1 km away from center
        side_length = math.sqrt(area_km2)

        lat_offset = side_length / (2 * dist_per_degree_lat)
        lon_offset = side_length / (2 * (dist_per_degree_lat * math.cos(math.radians(center_lat))))

        vertices = [(center_lon - lon_offset, center_lat - lat_offset),
                    (center_lon + lon_offset, center_lat - lat_offset),
                    (center_lon + lon_offset, center_lat + lat_offset),
                    (center_lon - lon_offset, center_lat + lat_offset)]

        square_polygon = Polygon(vertices)
        try:

            new_osm_features = ox.features_from_polygon(square_polygon, tags)
            new_osm_features['OA21CD'] = oa_code

            osm_features_df = pd.concat([osm_features_df, new_osm_features], ignore_index=True)
        except ox._errors.InsufficientResponseError:
            print(f"Warning: No OSM features found for OA code '{oa_code}'. Skipping.")

    return osm_features_df

def count_osm_features_by_oa(osm_features, tags):

    all_counts = []
    for oa_code in osm_features['OA21CD'].unique():
        oa_features = osm_features[osm_features['OA21CD'] == oa_code]
        tag_counts = {'OA21CD': oa_code}
        for tag in tags:
            if tags[tag] == True:
                if tag in oa_features.columns:
                    tag_counts[tag] = oa_features[tag].notnull().sum()
                else:
                    tag_counts[tag] = 0
            else:
                specific_tags = tags[tag]
                for specific_tag in specific_tags:
                    if tag in oa_features.columns:
                        tag_counts[specific_tag] = oa_features[oa_features[tag] == specific_tag].shape[0]
                    else:
                        tag_counts[specific_tag] = 0
        all_counts.append(tag_counts)
    return pd.DataFrame(all_counts).set_index('OA21CD')

def get_columns_with_most_values(df, n):
    non_nan_counts = df.count() #move this to fynesse

    # Sort columns by non-NaN counts in descending order
    sorted_columns = non_nan_counts.sort_values(ascending=False)

    # Select the top columns
    return sorted_columns.head(n).index
  
def plot_value_counts(df, columns):
    for column in columns:
        value_counts = df[column].value_counts()

        
        # Create the plot
        plt.figure(figsize=(10, 6))  
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Number of Different {column.capitalize()} Values')
        plt.xlabel(column.capitalize())
        plt.ylabel('Count')
        plt.xticks(rotation=90) 
        plt.tight_layout()
        plt.show()

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
