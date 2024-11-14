from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your model path

# Perform inference on an image
source = "image_3.jpg"  # Replace with your image path
results = model.predict(source=source,save = True,save_txt = True)
# Iterate through each result
for result in results:
    # Use the plot method to get the annotated image
    annotated_img = result.plot()
    # Display the image using OpenCV
    cv2.imshow("YOLOv8 Predictions", annotated_img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows() 
    # Assume 'annotated_img' is the image with predictions
# scale_percent = 50  # Percentage of the original size
# width = int(annotated_img.shape[1] * scale_percent / 100)
# height = int(annotated_img.shape[0] * scale_percent / 100)
# dim = (width, height)

# # Resize the image
# resized_img = cv2.resize(annotated_img, dim, interpolation=cv2.INTER_AREA)

# # Display the resized image
# cv2.imshow("YOLOv8 Predictions", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()     

# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import cv2
# import numpy
# # Load the trained YOLOv8 model
# model = YOLO("best.pt")  # Replace with your model path

# # Perform inference on an image
# source = "image_3.jpg"  # Replace with your image path
# results = model.predict(source=source)

# # Iterate through each result
# for result in results:
#     # Load the original image
#     original_img = result.orig_img  # This is in BGR format
    
#     # Convert BGR to RGB
#     original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
#     # Iterate through each detected box
#     for box in result.boxes:
#         # Extract bounding box coordinates
#         x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int).flatten()
        
#         # Extract confidence score
#         confidence = box.conf.cpu().numpy().item()
        
#         # Extract class ID and name
#         class_id = int(box.cls.cpu().numpy().item())
#         class_name = model.names[class_id]
        
#         # Draw bounding box
#         cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        
#         # Prepare label text
#         label = f"{class_name} {confidence:.2f}"
        
#         # Calculate text size for background rectangle
#         (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
#         # Draw background rectangle for text
#         cv2.rectangle(original_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (255, 0, 0), -1)
        
#         # Put label text above the bounding box
#         cv2.putText(original_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
#     # Display the annotated image using Matplotlib
#     plt.figure(figsize=(12, 8))
#     plt.imshow(original_img)
#     plt.axis('off')
#     plt.title("YOLOv8 Manual Annotations")
#     plt.show()