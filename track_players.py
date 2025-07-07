import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the YOLOv11 model
model = YOLO("yolov11_players.pt")  # Path to your model file

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Load the input video
cap = cv2.VideoCapture("15sec_input_720p.mp4")

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter("output_tracked.mp4", fourcc, 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv11 inference
    results = model.predict(source=frame, conf=0.4, classes=[0], verbose=False)[0]

    # Format detections for Deep SORT
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw boxes and IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save and show frame
    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
