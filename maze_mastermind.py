import cv2
from PIL import Image, ImageDraw
import random
import numpy as np
from matplotlib import pyplot as plt
import math


list_walkable=[]
list_nodes=[]


cells_in_side=25 # number of cells per side in the maze
def find_nearest_walkable_node(start_coordinates, list_nodes):
    min_distance = float('inf')
    nearest_node_index = -1

    for i, node in enumerate(list_nodes):
        if node.W:  # Check if the node is walkable
            node_center = (node.X, node.Y)
            distance = np.linalg.norm(np.array(start_coordinates) - np.array(node_center))

            if distance < min_distance:
                min_distance = distance
                nearest_node_index = i

    return nearest_node_index
def detect_start(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help circle detection
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    detected_balls = []

    # If circles are found, add their coordinates to the list
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            detected_balls.append(center)

    return detected_balls
def crop_and_save_image(image_path, output_path, crop_size_x=20, crop_size_y=20):
    # Read the original image
    original_image = cv2.imread(image_path)

    # Extract image dimensions
    height, width, _ = original_image.shape

    # Calculate the region of interest
    roi = original_image[crop_size_y:height-crop_size_y, crop_size_x:width-crop_size_x]

    # Save the cropped image
    cv2.imwrite(output_path, roi)

# ... (Previous code)
def generate_directions(path):
    directions = []

    for i in range(1, len(path)):
        current_node = path[i - 1]
        next_node = path[i]

        x_change = next_node.X - current_node.X
        y_change = next_node.Y - current_node.Y

        if x_change > 0 and y_change == 0:
            direction = "right"
        elif x_change < 0 and y_change == 0:
            direction = "left"
        elif x_change == 0 and y_change > 0:
            direction = "down"
        elif x_change == 0 and y_change < 0:
            direction = "up"
        elif x_change > 0 and y_change > 0:
            direction = "bottom-right"
        elif x_change < 0 and y_change > 0:
            direction = "bottom-left"
        elif x_change > 0 and y_change < 0:
            direction = "top-right"
        elif x_change < 0 and y_change < 0:
            direction = "top-left"
        else:
            direction = "unknown"

        directions.append(direction)

    return directions
directions= []
def visualize_shortest_path(start_node, goal_node, frame):
    
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node)
        current_node = current_node.parent

    # Print the shortest path
    print("Shortest Path:")
    for node in reversed(path):
        print(f"Node ({node.X}, {node.Y})")

    
    frame = frame#cv2.imread("generated_image6.PNG")
    for node in path:
        center = (node.X, node.Y)
        frame = cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Red dot for the path

 
 
    directions = generate_directions(path)
  
    print("Generated Directions:")
    for direction in directions:
        print(direction)


    
    cv2.imshow("Shortest Path", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_costs(start_node, goal_node, node_neighbors):
    # Assign an arbitrarily large cost to unwalkable nodes
    infinite_cost = float('inf')
    for node in list_nodes:
        if not node.W:
            node.G = node.H = node.F = infinite_cost

    
    start_node.G = 0
    start_node.H = calculate_heuristic(start_node, goal_node)
    start_node.F = start_node.G + start_node.H

    
    open_list = [start_node]
    closed_list = []

    while open_list:
        
        open_list.sort(key=lambda x: x.F)
        current_node = open_list.pop(0)

        # Skip processing for unwalkable nodes
        if current_node.W and current_node.G != infinite_cost:
            # Move current node from open list to closed list
            closed_list.append(current_node)

            
            if current_node == goal_node:
                break
        
            # Calculate costs for neighboring nodes
            for neighbor in node_neighbors[current_node]:
                if neighbor not in closed_list and neighbor.W:
                    tentative_g = current_node.G + calculate_distance(current_node, neighbor)
                    tentative_h = calculate_heuristic(neighbor, goal_node)
                    tentative_f = tentative_g + tentative_h

                   
                    if neighbor not in open_list or tentative_f < neighbor.F:
                        neighbor.G = tentative_g
                        neighbor.H = tentative_h
                        neighbor.F = tentative_f
                        neighbor.parent = current_node  # Update parent attribute

                        # Add neighbor to open list if not present
                        if neighbor not in open_list:
                            open_list.append(neighbor)

        # Calculate costs for neighboring nodes
        for neighbor in node_neighbors[current_node]:
            if neighbor not in closed_list and neighbor.W:
                tentative_g = current_node.G + calculate_distance(current_node, neighbor)
                tentative_h = calculate_heuristic(neighbor, goal_node)
                tentative_f = tentative_g + tentative_h

                
                if neighbor not in open_list or tentative_f < neighbor.F:
                    neighbor.G = tentative_g
                    neighbor.H = tentative_h
                    neighbor.F = tentative_f
                    neighbor.parent = current_node

                    
                    if neighbor not in open_list:
                        open_list.append(neighbor)

    
    return closed_list

def calculate_distance(node1, node2):
    
    return max(abs(node1.X - node2.X), abs(node1.Y - node2.Y))

def calculate_heuristic(node, goal_node):
    
    return abs(node.X - goal_node.X) + abs(node.Y - goal_node.Y)


class Node:
    def __init__(self, X, Y, W):
        self.X = X  
        self.Y = Y  
        self.G = 0 
        self.H = 0  
        self.F = self.G + self.H
        self.W= W
        self.parent = None

def are_neighbors(node1, node2, cell_size):
    return (
        abs(node1.X - node2.X) <= cell_size
        and abs(node1.Y - node2.Y) <= cell_size
    )

# dictionary to store neighbors for each node
node_neighbors = {}


def walkable(x1, y1, x2, y2, frame):
    #gframe = cv2.resize(frame, (600, 600))
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #frame = cv2.imread("generated_image6.PNG", cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to create a binary image
    gframe = cv2.threshold(gframe, 128, 255, cv2.THRESH_BINARY)[1]
    #gframe= cv2.imshow("bin", gframe)

    counter = 0
    side_length = abs(x1 - x2)
    side_length = int(side_length)

    walkable = True
    x1 = int (x1)
    x2=int(x2)
    y1 = int (y1)
    y2=int(y2)
    for i in range(x1, x2):
        for j in range(y1, y2):
            # Swap j and i for rows and columns
            pixel_value = gframe[j, i]

            if pixel_value < 250:
                counter += 1

    percent = counter / (side_length * side_length)
    #print("counter = " + str(counter))
    #print("percent= " + str(percent))
    #print("side length: " + str(side_length * side_length))
    if percent > 0.8:
        walkable = False
        X= int((abs(x2+x1))/2)
        Y= int((abs(y2+y1))/2)
        W= False
        
        node = Node(X,Y,W)
        print (node.X)
        print (node.Y)
        print (node.W)
        list_nodes.append(node)
    else:
        X= int((abs(x2+x1))/2)
        Y= int((abs(y2+y1))/2)
        W= True
        node = Node(X,Y,W)
        print (node.X)
        print (node.Y)
        print (node.W)
        list_nodes.append(node)
        list_walkable.append(node)
    return walkable


    

side_length=600
#gen_frame = generate_random_image(side_length)
#gen_frame.save("maze15.JPG")

#frame= cv2.imread("mynewmazeC.PNG") #maze5.JPG generated_image10.PNG
input_image_path = "mynewmazeC.PNG"
output_image_path = "sc3.png"
crop_size_x = 80  
crop_size_y = 40  

crop_and_save_image(input_image_path, output_image_path, crop_size_x, crop_size_y)
frame = cv2.imread("sc3.PNG")
frame = cv2.resize(frame, (600, 600))

y1=0

c=side_length/cells_in_side # cell width and height 400/10

y2=c # cell width and height
while (y2<=side_length):
    x1=0
    x2=c
    a=0
    diagonal=cells_in_side
    while(a<diagonal):
        
        is_walkable=  walkable (x1,y1,x2,y2,frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (int((x1+x2)/2), int((y1+y2)/2))  # (x, y) position where the  will be placed
        font_scale = 0.4
        font_color = (155, 155, 155)  # White color in BGR format
        thickness = 2

        # Add text to the image
        #frame=cv2.circle(frame, (x1, y1), 5, (155,155,155), -1)
        #frame=cv2.circle(frame, (x2, y2), 5, (155,155,155), -1)
        textx1y1= "x1: "+ str(x1)+ " y1: "+ str(y1)
        textx2y2= "x2: "+ str(x2)+ " y2: "+ str(y2)
        #cv2.putText(frame, textx1y1, (x1, y1+20), font, font_scale, font_color, thickness)
        #cv2.putText(frame, textx2y2, (x2, y2+20), font, font_scale, font_color, thickness)
        text = str(is_walkable)
        center= position
        #cv2.putText(frame, str(center), position, font, font_scale, font_color, thickness)
        x1+=c
        x2+=c
        a+=1
    y1+=c
    y2+=c


# Iterate through each node to find neighbors
for node in list_nodes:
    neighbors = []
    for other_node in list_nodes:
        if node != other_node and are_neighbors(node, other_node, c):
            neighbors.append(other_node)
    node_neighbors[node] = neighbors

# Find the maximum number of neighbors for any node
max_neighbors_count = max(len(neighbors) for neighbors in node_neighbors.values())


print("Node Neighbors Dictionary:")
for node, neighbors in node_neighbors.items():
    print(f"Node ({node.X}, {node.Y}): Neighbors {[n.X for n in neighbors]}")

print("\nMaximum number of neighbors for any node:", max_neighbors_count)
        
    
start_node = list_nodes[0]
#gnode = Node(472,592, True)
goal_node= list_nodes[-1]
#goal_node = (487,592)#list_nodes[1591]


closed_list = calculate_costs(start_node, goal_node, node_neighbors)


print("Node Costs:")
for node in closed_list:
    print(f"Node ({node.X}, {node.Y}): G={node.G}, H={node.H}, F={node.F}")

visualize_shortest_path(start_node, goal_node, frame)

#print ("walkable: "+ str(walkable))
#frame=cv2.circle(frame, (160, 80), 5, (255,255,255), -1)
frame= cv2.imshow("Output", frame)
#print (list_walkable)
#print (list_nodes)
cv2.waitKey(0)
cv2.destroyAllWindows()




