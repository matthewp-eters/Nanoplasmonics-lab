import cv2
import matplotlib.pyplot as plt
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="El2cLimOoUhrGPw9B2bJ")
project = rf.workspace().project("not-proteins")
model = project.version(1).model

# # Load the video
video_paths = ["/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/BSA007_pretrap_sequential.avi",
                "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/BSA008_pretrap_sequential.avi",
                "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CTC007_pretrap_sequential.avi",
                "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CA001_pretrap_sequential.avi",
                "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CA011_pretrap_sequential.avi",
                "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CTC001_pretrap_sequential.avi",
                "/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/CTC009_pretrap_sequential.avi"]

#video_paths = ["/Users/matthewpeters/Desktop/UVIC/lab/Projects/Videos/code/Sequential removal/BSA008_pretrap_sequential.avi"]
# Initialize a list to store object positions
positions_list = []

# Function to perform object detection on a single frame
def detect_objects(frame):
    # Convert frame to RGB (required by the model)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    result = model.predict(frame_rgb, confidence=15)

    # Return the detection results
    return result.json()["predictions"]

# ... (rest of the code)
# Loop through all the videos
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
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
        
    #             # Draw bounding boxes on the frame for visualization (optional)
    #             cv2.rectangle(frame, (round(x-width/2), round(y-height/2)), (round(x + width/2), round(y + height/2)), (0, 255, 0), 2)
        
    #    # Display the frame with bounding boxes (optional)
    #     cv2.imshow("Object Tracking", frame)
    
    #     #Exit the loop when 'q' key is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
        frame_count += 1

    # # Release video capture and close the window
    # cap.release()
    # cv2.destroyAllWindows()
    
    # Calculate time duration for the current video
    fps = cap.get(cv2.CAP_PROP_FPS)
    first_frame_number = positions_list[0]["frame_number"]
    last_frame_number = positions_list[-1]["frame_number"]
    time_duration = (last_frame_number - first_frame_number) / fps
    print(f"Time duration for {video_path}: {time_duration:.2f} seconds")
    positions_list=[]

# # Extract X and Y coordinates for the scatter plot
# x_coords = [x for x, y in positions_list]
# y_coords = [y for x, y in positions_list]

# plt.figure()
# # plt.scatter(x_coords, y_coords, c=range(len(x_coords)), cmap='inferno', marker='o', s=50, label='Detected Proteins')
# plt.scatter(x_coords, y_coords, marker='o', s=50, label='Detected Proteins', alpha = 0.3)

# # plt.plot(x_coords, y_coords, '-r', alpha=0.5)  # Connect the dots with a red line

# plt.scatter(640, 360, c='r', marker='o', s=100)

# # # Show a red circle at the center of the frame
# # #center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
# # plt.scatter(int(center_coords[0]), int(center_coords[1]), c='r', marker='o', s=100)

# # # Add a colorbar to indicate the sequence of points
# # cbar = plt.colorbar()
# # cbar.set_label('Frame Sequence', rotation=270, labelpad=15)

# # Set plot labels and title
# plt.xlabel('X-coordinate (Pixels)')
# plt.ylabel('Y-coordinate (Pixels)')
# plt.title('tracked protein positions')


# # Flip the Y-axis
# plt.gca().invert_yaxis()

# # Show the plot
# #plt.legend()
# plt.grid(True)
# plt.show()


