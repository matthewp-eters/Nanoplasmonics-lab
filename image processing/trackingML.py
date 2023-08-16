import cv2
import matplotlib.pyplot as plt
from roboflow import Roboflow
import os

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rc('axes', labelsize=18) 
plt.rc('font', family='sans-serif')

# Initialize Roboflow
rf = Roboflow(api_key="El2cLimOoUhrGPw9B2bJ")
project = rf.workspace().project("not-proteins")
model = project.version(3).model

# # Load the video
#video_paths = ["/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/BSA007_pretrap_sequential.avi",
#                 "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/BSA008_pretrap_sequential.avi",
#                 "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CTC007_pretrap_sequential.avi",
#                 "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CA001_pretrap_sequential.avi",
#                 "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CA011_pretrap_sequential.avi",
#                 "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CTC001_pretrap_sequential.avi",
#                 "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CTC009_pretrap_sequential.avi"]

video_paths = ["/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/YPep_pre_sequential.avi"]
# Initialize a list to store object positions
positions_list = []
frame_list=[]



def distance(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# Function to perform object detection on a single frame
def detect_objects(frame):
    # Convert frame to RGB (required by the model)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    result = model.predict(frame_rgb, confidence=10)

    # Return the detection results
    return result.json()["predictions"]

# ... (rest of the code)
# Loop through all the videos

# Function to find the closest object to a given point
def find_closest_object(objects, last_x, last_y):
    min_distance = float('inf')
    closest_obj = None

    for obj in objects:
        x, y, width, height = obj["x"], obj["y"], obj["width"], obj["height"]
        distance_to_last = distance(x, y, last_x, last_y)
        
        if distance_to_last < min_distance:
            min_distance = distance_to_last
            closest_obj = obj

    return closest_obj

# Loop through all the videos
# Loop through all the videos
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    last_x, last_y = None, None  # Store the position of the last detected object
    
    # Read the first frame
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        exit()

    # Get width and height information from the frame
    height, width, _ = frame.shape


    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # You can change the codec as needed
    # Create the output video file name
    base_name = os.path.basename(video_path)  # Get the base filename
    name_without_extension, extension = os.path.splitext(base_name)
    output_video_path = f'{name_without_extension}_tracked{extension}'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    
    # # Loop through the video frames
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
        
    #     height, width, channels = frame.shape
    #     center_coords = (width / 2, height / 2)
    
    #     # Perform object detection on the current frame
    #     detection_results = detect_objects(frame)
    
    #     # Process detection results to track the class of interest
    #     class_of_interest = "proteins"
    #     tracked_objects = [d for d in detection_results if d["class"] == class_of_interest]
        
    #     class_of_interest_trap = "trap"
    #     trap_objects = [d for d in detection_results if d["class"] == class_of_interest_trap]
        
    #     if last_x is not None and last_y is not None:
    #         closest_obj = find_closest_object(tracked_objects, last_x, last_y)
            
    #         if closest_obj is not None and not any(obj["class"] == class_of_interest_trap for obj in trap_objects):
    #             x, y, width, height = closest_obj["x"], closest_obj["y"], closest_obj["width"], closest_obj["height"]
    #             x, y, width, height = int(x), int(y), int(width), int(height)
    #             positions_list.append((x, y))
    #             frame_list.append(frame_count)
    #             last_x, last_y = x, y
    #             cv2.rectangle(frame, (round(x - width / 2), round(y - height / 2)),
    #                           (round(x + width / 2), round(y + height / 2)), (0, 255, 0), 2)
    #     else:
    #         if tracked_objects and not any(obj["class"] == class_of_interest_trap for obj in trap_objects):
    #             obj = tracked_objects[0]
    #             x, y, width, height = obj["x"], obj["y"], obj["width"], obj["height"]
    #             x, y, width, height = int(x), int(y), int(width), int(height)
    #             positions_list.append((x, y))
    #             frame_list.append(frame_count)
    #             last_x, last_y = x, y
    #             cv2.rectangle(frame, (round(x - width / 2), round(y - height / 2)),
    #                           (round(x + width / 2), round(y + height / 2)), (0, 255, 0), 2)
    
    # Loop through the video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
    
    
        height, width, channels = frame.shape
        center_coords = (width/2, height/2)
    
        # Perform object detection on the current frame
        detection_results = detect_objects(frame)
    
        # Process detection results to track the class of interest
        # Here, you can use a tracking algorithm to associate detections between frames
        # For simplicity, let's assume the class of interest is "proteins"
        class_of_interest = "proteins"
        tracked_objects = [d for d in detection_results if d["class"] == class_of_interest]
    
        for obj in tracked_objects:
                x, y, width, height = obj["x"], obj["y"], obj["width"], obj["height"]
                # Ensure the values are integers before using them
                x, y, width, height = int(x), int(y), int(width), int(height)
                positions_list.append((x, y))
                frame_list.append(frame_count)
        
    #             # Draw bounding boxes on the frame for visualization (optional)
                cv2.rectangle(frame, (round(x-width/2), round(y-height/2)), (round(x + width/2), round(y + height/2)), (0, 255, 0), 2)
        
    #    # Display the frame with bounding boxes (optional)
    #     cv2.imshow("Object Tracking", frame)
    
    #     #Exit the loop when 'q' key is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
        frame_count += 1


        frame_number_text = f'Frame: {frame_count}'
        cv2.putText(frame, frame_number_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
       
        #out.write(frame)
        
        #cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1



    # # Release video capture and close the window
    # cap.release()
    # cv2.destroyAllWindows()
    
    # Calculate time duration for the current video
    fps = cap.get(cv2.CAP_PROP_FPS)
    first_frame_number = frame_list[0]
    last_frame_number = frame_list[-1]
    time_duration = (last_frame_number - first_frame_number) / fps
    print(f"Time duration for {video_path}: {time_duration:.2f} seconds")
    #positions_list=[]
    #frame_list = []


    
    protein_distances = []
    for i in range(1, len(positions_list)):
        x1, y1 = positions_list[i-1][0], positions_list[i-1][1]
        x2, y2 = positions_list[i][0], positions_list[i][1]
        dist = distance(x1, y1, x2, y2)
        protein_distances.append(dist)
        
        # Print distances traveled by each detected protein
    print(sum(protein_distances))
    
    print(sum(protein_distances)/time_duration)
    
cap.release()
cv2.destroyAllWindows()
 

plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rc('axes', labelsize=18) 
plt.rc('font', family='sans-serif')
# Extract X and Y coordinates for the scatter plot
x_coords = [x for x, y in positions_list]
y_coords = [y for x, y in positions_list]

plt.figure(figsize=(8,6))
plt.scatter(x_coords, y_coords, c=range(len(x_coords)), cmap='inferno', marker='o', s=50, label='Detected Proteins')
#plt.scatter(x_coords, y_coords, marker='o', s=50, label='Detected Proteins', alpha = 0.3)

plt.plot(x_coords, y_coords, '-r', alpha=0.5)  # Connect the dots with a red line
plt.scatter(640, 360, c='r', marker='o', s=100)


# # Add a colorbar to indicate the sequence of points
#cbar = plt.colorbar()
#cbar.set_label('Frame Sequence', rotation=270, labelpad=15)

# Set plot labels and title
#plt.title(f"{video_path}")

plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')

plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rc('font', family='sans-serif')
plt.scatter(x_coords, y_coords, c='b', marker='o', alpha = 0.1)

#plt.hist2d(x_coords, y_coords, bins=25, cmap = 'hot')
# Flip the Y-axis
plt.gca().invert_yaxis()
plt.savefig("ALL.pdf", format="pdf", bbox_inches="tight")

# Show the plot
#plt.legend()
plt.grid(True)
plt.show()


