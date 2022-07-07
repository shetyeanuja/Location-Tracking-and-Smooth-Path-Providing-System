import os, sys
from google.colab import drive
import geopandas as gpd
import plotly
import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import geopy
import scipy.signal
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import time
import geopy
from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from pprint import pprint
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from gtts import gTTS 
from IPython.display import Audio
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, url_for, request, redirect
from branca.element import Figure
from opencage.geocoder import OpenCageGeocode
from opencage.geocoder import InvalidInputError, RateLimitExceededError, UnknownError
import requests,json
import geopy.distance

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
# ritik opencage - c8dd5bcbf8fc4a32a1e4d658123f54fc

#Weather and temperature conditions of source and destination using openweathermap api
def weather(city):

    api_key = "be2c186e8f4283d29b95c5175dacbddc"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
  
    complete_url = base_url + "appid=" + api_key + "&q=" + city
    response = requests.get(complete_url)
    x = response.json()
  
    if x["cod"] != "404":
        y = x["main"]
        current_temperature = y["temp"]
        print(current_temperature)
        
        z = x["weather"]
        weather_description = z[0]["description"]
    
        return current_temperature, weather_description
    
    else:
        print("Done")



def distanceMatrix(origin, destination):

    token = "2OX4HduEBS5wOMli4cMYUkzpYwFAR"
    base_url = "https://api.distancematrix.ai/maps/api/distancematrix/json?"
  
    complete_url = base_url + "origins=" + str(origin[0])+","+str(origin[1]) + "&destinations=" + str(destination[0])+","+str(destination[1])+"&key="+token
    response = requests.get(complete_url)
    x = response.json()
    print(x)
    if x["rows"] != "":
        y = x["rows"]
        
        print(y)
        
    
        return y[0]['elements'][0]['duration']['text']
    else:
        print("Done")

   


#Running the flask App
app=Flask(__name__)
run_with_ngrok(app)
@app.route("/", methods=['GET','POST'])


def index():
  # try:
    locn=[]
    time=0
    div=None
    travel_time=0
    approx_travel_time=0
    loader=False
    if(request.method=='POST'):
      loader=True
      source_locn=request.form['source']
      dest_locn=request.form['destination']
      locn.append(source_locn)
      locn.append(dest_locn)
      print(locn)
        
      #Using opencage api for converting physical address into coordinates
      key = '049e6fb09005450884e33ab899d4ecaa'
      geocoder = OpenCageGeocode(key)
      query1 = source_locn
      query2 = dest_locn
      source_results = geocoder.geocode(query1)
      dest_results = geocoder.geocode(query2)
      try:
        origin_lat,origin_long = source_results[0]['geometry']['lat'], source_results[0]['geometry']['lng']
        dest_lat,dest_long = dest_results[0]['geometry']['lat'], dest_results[0]['geometry']['lng']
      except:
        return render_template('index.html', location=locn, div_placeholder=div, travel_time=travel_time, loader=loader, error=True)
      print("Source lat: "+str(origin_lat)+" Source long: "+str(origin_long))
      print("Dest lat: "+str(dest_lat)+" Dest long: "+str(dest_long))
      
      #Graph view of the map
      ox.config(log_console=True, use_cache=True)
      #G = ox.graph_from_bbox(origin_lat,dest_lat,origin_long,dest_long, retain_all=False, truncate_by_edge=True, simplify=False, network_type='drive')
      
      coords_1 = (origin_long, origin_lat)
      coords_2 = (dest_long, dest_lat)
      distance_s_to_d = geopy.distance.geodesic(coords_1, coords_2).km
      print(distance_s_to_d)
      point = origin_lat,origin_long
      dist = int(distance_s_to_d * 1000)
      G = ox.graph_from_point(point,dist=dist,network_type='drive')
      print("Graph view of map loaded")
      #ox.plot_graph(G_proj)
      ox.plot_graph(G)
      G = ox.utils_graph.get_largest_component(G)

      #Getting the nearest node from source and destination from the graph by Euclidean algorithm
      origin_point = (origin_lat, origin_long) 
      destination_point = (dest_lat, dest_long)
      # origin_node,dd1 = ox.get_nearest_node(G, origin_point, method='euclidean',return_dist=True)
      # destination_node,dd2 = ox.get_nearest_node(G, destination_point, method='euclidean',return_dist=True)
      origin_node = ox.distance.nearest_nodes(G, origin_long,origin_lat)
      destination_node = ox.distance.nearest_nodes(G, dest_long,dest_lat)
      print("Source node: "+str(origin_node))
      print("Destination node: "+str(destination_node))

      #All the nodes between source node and destination node such that the distance is shortest by Dijkstra's algorithm
      try:
        route = nx.shortest_path(G, origin_node, destination_node, weight='length')
      except Exception:
        route = None
      

      #Getting all the routes from source to destination and storing their coordinates
      try:
        routes = ox.k_shortest_paths(G, origin_node, destination_node, k=5, weight='length')
        routes = list(routes)
      except:
        print("Alternative routes obtained")
      fig, ax = ox.plot_graph_routes(G, list(routes),  route_colors='r', route_linewidth=2, node_size=0)
      lola1 = []
      lola2 = []
      for i in range(len(routes)):
        longi = [] 
        lat = []  
        rr = list(routes[i])
        for j in rr:
          point = G.nodes[j]
          longi.append(point['x'])
          lat.append(point['y'])
        lola1.append(longi)
        lola2.append(lat)
      
      print("All the nodes from source to destination: ")
      print(route)
      #Graph view by connecting all the nodes of optimal route
      print("the shortest route")
      fig, ax = ox.plot_graph_route(G, route)
      

      #getting the latitude & longitude of all the nodes of optimal route for kalman filter estimation
      initial_longi = [] 
      initial_lat = []  
      for i in route:
          point = G.nodes[i]
          initial_longi.append(point['x'])
          initial_lat.append(point['y'])

      #Modifying the route according to actual geometry on map
      def node_list_to_path(G, node_list):
      
        edge_nodes = list(zip(node_list[:-1], node_list[1:]))
        lines = []
        for u, v in edge_nodes:
            #if there are parallel edges, select the shortest in length
            data = min(G.get_edge_data(u, v).values(),key=lambda x: x['length'])
            #if it has a geometry attribute
            if 'geometry' in data:
                #add them to the list of lines to plot
                xs, ys = data['geometry'].xy
                lines.append(list(zip(xs, ys)))
            else:
                #if it doesn't have a geometry attribute, then the edge is a straight line from node to node
                x1 = G.nodes[u]['x']
                y1 = G.nodes[u]['y']
                x2 = G.nodes[v]['x']
                y2 = G.nodes[v]['y']
                line = [(x1, y1), (x2, y2)]
                lines.append(line)
        return lines

      #getting the list of coordinates from the path (which is a list of nodes)
      lines = node_list_to_path(G, route)
      longi = []
      lat = []
      for i in range(len(lines)):
          z = list(lines[i])
          l1 = list(list(zip(*z))[0])
          l2 = list(list(zip(*z))[1])
          for j in range(len(l1)):
              longi.append(l1[j])
              lat.append(l2[j])
      
      print("Length of initial lat: ", len(initial_lat))
      print("Length of final lat: ", len(lat))

      #Kalman filter estimation of the coordinates of optimal route between source and destination
      import time
      c2=[]
      for i in route:
          c1 = [] 
          point = G.nodes[i]
          longi = point['x']
          lat = point['y']
          c1.append(longi)
          c1.append(lat)
          c2.append(c1)
      measurements = np.asarray((c2))

      #1st Kalman Filter
      initial_state_mean = [measurements[0, 0],
                          0,
                          measurements[0, 1],
                          0]

      transition_matrix = [[1.0, 1, 0, 0],
                          [0, 1.0, 0, 0],
                          [0, 0, 1.0, 1],
                          [0, 0, 0, 1.0]]

      observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]

      kf1 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean)

      kf1 = kf1.em(measurements, n_iter=50)
      (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

      plt.figure(figsize=(40,10))

      times = range(measurements.shape[0])
      plt.plot(times, measurements[:, 0], 'bo',
              times, measurements[:, 1], 'ro',
              times, smoothed_state_means[:, 0], 'b--',
              times, smoothed_state_means[:, 2], 'r--',)
              
      #plt.show()

      #2nd Kalman Filter
      kf2 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean,
                      observation_covariance = 1000*kf1.observation_covariance,
                      em_vars=['transition_covariance', 'initial_state_covariance'])

      kf2 = kf2.em(measurements, n_iter=50)
      (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)

      plt.figure(figsize=(40,10))
      times = range(measurements.shape[0])
      plt.plot(times, measurements[:, 0], 'bo',
              times, measurements[:, 1], 'ro',
              times, smoothed_state_means[:, 0], 'b--',
              times, smoothed_state_means[:, 2], 'r--',)
      #plt.show()

      #3rd Kalman Filter
      time_before = time.time()
      n_real_time = 3

      kf3 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean,
                      observation_covariance = 100*kf1.observation_covariance,
                      em_vars=['transition_covariance', 'initial_state_covariance'])

      kf3 = kf3.em(measurements[:-n_real_time, :], n_iter=50)
      (filtered_state_means, filtered_state_covariances) = kf3.filter(measurements[:-n_real_time,:])

      print("Time to build and train kf3: %s seconds" % (time.time() - time_before))

      x_now = filtered_state_means[-1, :]
      P_now = filtered_state_covariances[-1, :]
      x_new = np.zeros((n_real_time, filtered_state_means.shape[1]))
      i = 0

      for measurement in measurements[-n_real_time:, :]:
        time_before = time.time()
        (x_now, P_now) = kf3.filter_update(filtered_state_mean = x_now,
                                            filtered_state_covariance = P_now,
                                            observation = measurement)
        print("Time to update kf3: %s seconds" % (time.time() - time_before))
        x_new[i, :] = x_now
        i = i + 1
      plt.figure(figsize=(40,10))
      old_times = range(measurements.shape[0] - n_real_time)
      new_times = range(measurements.shape[0]-n_real_time, measurements.shape[0])
      plt.plot(times, measurements[:, 0], 'bo',
              times, measurements[:, 1], 'ro',
              old_times, filtered_state_means[:, 0], 'b--',
              old_times, filtered_state_means[:, 2], 'r--',
              new_times, x_new[:, 0], 'b-',
              new_times, x_new[:, 2], 'r-')

      #plt.show()
      
      #measurements after kalman filter of optimal route
      longi = []
      lat = []
      for i in range(len(measurements)):
        longi.append(measurements[i][0])

      for i in range(len(measurements)):
        lat.append(measurements[i][1])
    
      print("Kalman filter estimation done")

      #reverse geocoding 
      def get_address_by_location(latitude, longitude, language="en"):
        try:
          results = geocoder.reverse_geocode(latitude, longitude , language='de', no_annotations='1')
          if results and len(results):
            required = results[0]['components']
            if required['_type'] in types:
              return required
        except RateLimitExceededError as ex:
          print(ex)
            

      #latitude, longitude and neighbourhoods from source to destination for plotting as markers
      types = ['restaurant','doctors','school','college','hospital','clinic','shop','cinema','cafe']
      am_l1,am_l2 = [],[]
      h_l1,h_l2 = [],[]
      edu_l1,edu_l2 = [],[]
      amenity_spots = []
      health_spots = []
      edu_spots = []

      marker_longi = []
      marker_lat = []

      for i in range(0,len(longi)):
        marker_longi.append(longi[i])
        marker_lat.append(lat[i])

      la = marker_lat
      lo = marker_longi
      try:
        for i in range(len(la)):
          address = get_address_by_location(la[i], lo[i])
          if address:
            address_name = address['_type']
            if address_name == 'restaurant' or address_name == 'shop' or address_name == 'cinema' or address_name == 'cafe':
              am_l1.append(la[i])
              am_l2.append(lo[i])
              amenity_spots.append(address[address_name])
            elif address_name == 'doctors' or address_name == 'hospital' or address_name == 'clinic':
              h_l1.append(la[i])
              h_l2.append(lo[i])
              health_spots.append(address[address_name])
            elif address_name == 'school' or address_name == 'college':
              edu_l1.append(la[i])
              edu_l2.append(lo[i])
              edu_spots.append(address[address_name])
              
              
      except KeyError as ke:
        print(ke) 

      print(amenity_spots)
      print(health_spots)
      print(edu_spots)

      print("Fetched neighbourhoods")
        

      #Plotting the final route on scattermapbox
      def plot_path(lola2, lola1, lat, long, origin_point, destination_point):

          fig = go.Figure()
          
          #all the routes from S to D
          for i in range(0,len(lola1)):
            fig.add_trace(go.Scattermapbox(
                name = "Route",
                mode = "lines",
                lon = lola1[i],
                lat = lola2[i],
                marker = {'size': 10},
                line = dict(width = 4, color = 'blue')))

          #optimal route from S to D
          fig.add_trace(go.Scattermapbox(
                name = "Optimal Route",
                mode = "lines",
                lon = longi,
                lat = lat,
                marker = {'size': 10},
                line = dict(width = 6, color = 'black')))

          #neighbourhoods on optimal route
          fig.add_trace(go.Scattermapbox(
                name = "Amenity",
                mode = "markers",
                lon = am_l2,
                lat = am_l1,
                text = amenity_spots,
                textposition="bottom center",
                marker = {'size': 10, 'color':"papayawhip"}))
          
          fig.add_trace(go.Scattermapbox(
                name = "Health",
                mode = "markers",
                lon = h_l2,
                lat = h_l1,
                text = health_spots,
                textposition="bottom center",
                marker = {'size': 15, 'color':"green"}))
          
          fig.add_trace(go.Scattermapbox(
                name = "Education",
                mode = "markers",
                lon = edu_l2,
                lat = edu_l1,
                text = edu_spots,
                textposition="bottom center",
                marker = {'size': 15, 'color':"violet"}))

          #source marker
          try:
             source_info=weather(source_locn)
             fig.add_trace(go.Scattermapbox(
                  name = "Source",
                  mode = "markers",
                  text = source_locn.capitalize()+ "\nTemperature: "+str(round(float(source_info[0])- 273.15,2))+"°C"+ "\nWeather Description: "+source_info[1].title(),
                  lon = [origin_point[1]],
                  lat = [origin_point[0]],
                  marker = {'size': 20, 'color':"red"}))
          except:
                fig.add_trace(go.Scattermapbox(
                  name = "Source",
                  mode = "markers",
                  text = source_locn.capitalize(),
                  lon = [origin_point[1]],
                  lat = [origin_point[0]],
                  marker = {'size': 20, 'color':"red"}))
                
          #dest marker  
          try: 
              dest_info=weather(dest_locn)
              fig.add_trace(go.Scattermapbox(
                  name = "Destination",
                  mode = "markers",
                  text =  dest_locn.capitalize()+ "\nTemperature: "+str(round(float(dest_info[0])- 273.15,2))+"°C"+ "\nWeather Description: "+dest_info[1].title(),
                  lon = [destination_point[1]],
                  lat = [destination_point[0]],
                  marker = {'size': 20, 'color':'red'}))
          except:
              fig.add_trace(go.Scattermapbox(
                 name = "Destination",
                 mode = "markers",
                 text =  dest_locn.capitalize(),
                 lon = [destination_point[1]],
                 lat = [destination_point[0]],
                 marker = {'size': 20, 'color':'red'}))
              
          #centre of map
          lat_center = origin_point[0]
          long_center = origin_point[1]
          fig.update_layout(mapbox_style="stamen-terrain",
              mapbox_center_lat = 30, mapbox_center_lon=-80)
          fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                            mapbox = {
                                'center': {'lat': lat_center, 
                                'lon': long_center},
                                'zoom': 12})
          

          fig.show()
          return fig
        
      fig=plot_path(lola2, lola1, lat, longi, origin_point, destination_point)
      plotly.offline.plot(fig, filename='route.html')

      #see the travel time for the whole optimal route
      #travel_time = nx.shortest_path_length(G, origin_node, destination_node, weight='travel_time')
      approx_travel_time=distanceMatrix([origin_lat, origin_long], [dest_lat,dest_long])
      approx_travel_time= int(approx_travel_time.split()[0])
      print("Travel time is",approx_travel_time,"mins")
      print("Total travel time : "+str(travel_time)+" minutes approximately")
    
      div = fig.to_html(full_html=False)
    
      
  # except:
  #     print("Some error occurred")
  

    return render_template('index.html', location=locn, div_placeholder=div, travel_time=approx_travel_time, loader=loader,error=False)

@app.route('/map')
def map():
  return render_template('map.html')
  
if __name__ == '__main__':       
  app.run()
