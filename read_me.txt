Player Re-Identification and Tracking using YOLOv11 and Deep SORT
------------------------------------------------------------------

This project uses a fine-tuned YOLOv11 model to detect players in a video and Deep SORT to assign and maintain tracking IDs even if players leave and re-enter the frame. The goal is to simulate real-time player tracking for a 15-second input video.

How it works:
-------------
- I used a YOLOv11 model that is trained to detect players and the ball.
- The video is processed frame by frame.
- Bounding boxes are drawn on each detected player.
- Deep SORT takes the detection boxes and assigns a consistent ID to each player, helping track them even after they go out of the frame and come back.
- The output is a video with tracking results saved as "output_tracked.mp4".

Files included:
---------------
- track_players.py → the main script
- yolov11_players.pt → trained model file (needs to be placed in the same folder)
- 15sec_input_720p.mp4 → the input video to process
- output_tracked.mp4 → output video with tracking
- README.txt → this file

How to run:
-----------
1. Open this project in PyCharm.
2. Make sure Python is installed and the correct interpreter is selected.
3. Install dependencies using the terminal:
   pip install ultralytics opencv-python deep_sort_realtime
4. Run the script: track_players.py
5. The output video will be generated in the same folder.

Dependencies:
-------------
- Python 3.x
- ultralytics
- opencv-python
- deep_sort_realtime

Note:
-----
Make sure the model file and video file are placed in the same directory as the Python script.

This was a fun and challenging project where I learned how to combine object detection with object tracking. It can be extended further to track the ball or recognize goals later.

– Divyansh Gupta
