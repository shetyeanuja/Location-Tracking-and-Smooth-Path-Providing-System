# Location-Tracking-and-Smooth-Path-Providing-System

The problem statement is “Location Tracking and
Smooth Path Providing System”. The project aims at
delivering an optimized route from source to
destination by tracking the real time coordinates,
displaying nearby amenities and facilities through the
route, weather conditions and thereby providing the
approximate travel time for the same. Also, sharing of
current location with others in case needed, is also
embedded in the project.
Initially, the source and destination address for
which user wants to travel the route has been
provided by the user. By using various libraries
of python, tracking of all the possible routes and
the optimum amongst them was done using
data science techniques. Dijkstra’s Algorithm
was used for getting the locality of points of the
shortest route and Kalman Filtration Technique
on those points was applied to get a clean and
smooth path on the map. Scattermapbox was
used as the means of representing the route
over a map. Nearby amenities, weather and
temperature conditions and current location of
the user were fetched through API. Travel time
was predicted so the user can decide his journey
beforehand. 
