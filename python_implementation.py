import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device("cuda")

# Load YOLO model
model = YOLO('paper Implimentation/yolov8n-pose.pt').to(device)

# Video settings
video_path = "videos/WhatsApp Video 2024-05-20 at 10.00.32.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(3, 1280)
cap.set(4, 720)

# Background subtractor
# backsub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Store the track history
track_history = defaultdict(lambda: [])
# Kernel for morphological operations
kernel = None
# Define threshold distance increase for detecting throwing action
threshold_distance = 20

# KCF tracker
tracker = cv2.TrackerKCF_create()
tracker_initialized = False
tracker_bbox = None
current_hand = None  # Keep track of which hand (left or right) is being tracked

def is_contour_connected_to_keypoint(contour, keypoint, threshold=10):
    for point in contour:
        distance = np.linalg.norm(point[0] - keypoint)
        if distance < threshold:
            return True
    return False

def is_throwing_action(bbox, keypoint,threshold=50):
    x_contour, y_contour, width, height = bbox
    obj_center = np.array([x_contour + width / 2, y_contour + height / 2])
    # print("Distances from object center to keypoints:")
    # for i, keypoint in enumerate(keypoints):
    # print(keypoint)
    if(keypoint[0]!=0 and keypoint[1]!=0):
        distance = np.linalg.norm(obj_center - keypoint)
        print(f"Distance to keypoint  [{keypoint}] : {distance:.2f}")
        if distance>=threshold:
            return True
    return False



if not cap.isOpened():
    print(f"Error: Unable to open video file")
    exit()

thrown=False
ret, backframe = cap.read()
tracking_frame = backframe.copy()
backframe = cv2.cvtColor(backframe, cv2.COLOR_RGB2GRAY)
selected_pt=None
selected_hand=None
selected_pt_1=None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame")
        break
    tracking_frame = frame.copy()
    fgframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    fgMask = cv2.absdiff(fgframe, backframe)
    _, fgMask = cv2.threshold(fgMask, 0, 255, cv2.THRESH_OTSU)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=1)

    results = model.track(frame, persist=True, tracker='bytetrack.yaml')
    boundingboxes = results[0].boxes.xywh.cpu().numpy()
    keypoints_list = results[0].keypoints.xy.cpu().numpy()
    ids = {}
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()  # Extract track IDs

    if tracker_initialized:
        success, tracker_bbox = tracker.update(tracking_frame)
        if success:

            p1 = (int(tracker_bbox[0]), int(tracker_bbox[1]))
            p2 = (int(tracker_bbox[0] + tracker_bbox[2]), int(tracker_bbox[1] + tracker_bbox[3]))
            # print_distances_to_keypoints(tracker_bbox, keypoints_list[0])
            temp_tracker_box=(int(tracker_bbox[0]), int(tracker_bbox[1]),int(tracker_bbox[2]),int(tracker_bbox[3]))
            # print(temp_tracker_box)
            pt1 = keypoints_list[0][9].astype(int)  # Left hand
            pt2 = keypoints_list[0][10].astype(int)
            if pt1[0] > pt2[0]:
                pt1, pt2 = pt2, pt1
            if selected_hand=="Left":
                print("left")
                selected_pt_1=pt1
            else:
                print("Right")
                selected_pt_1=pt2
            # print(keypoints_list)
            # print(tracker_bbox)
            # print(selected_pt_1)
            obj_center = (int(tracker_bbox[0]) + int(tracker_bbox[2]) // 2, int(tracker_bbox[1]) + int(tracker_bbox[3]) // 2)
            # print(obj_center)
            thrown=is_throwing_action(temp_tracker_box,selected_pt_1)
            # print(obj_center)
            # print(keypoints_list)
            # print(selected_pt)
            # cv2.line(frame,obj_center,selected_pt_1,(255, 0, 0), 2)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.putText(frame, f"Tracking {current_hand} Hand", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            tracker_initialized = False  # Tracker failed, reinitialize in the next iteration
    else:
        # Object tracking with YOLO
        if len(ids) > 0:
            for bbox, keypoints, id in zip(boundingboxes, keypoints_list, ids):
                x, y, w, h = bbox.astype(int)
                pt1 = keypoints[9].astype(int)  # Left hand
                pt2 = keypoints[10].astype(int)  # Right hand
                if not (0 <= pt1[0] < frame.shape[1] and 0 <= pt1[1] < frame.shape[0]):
                    continue
                if not (0 <= pt2[0] < frame.shape[1] and 0 <= pt2[1] < frame.shape[0]):
                    continue

                height_hand = h // 2
                width_hand = w
                if pt1[0] > pt2[0]:
                    pt1, pt2 = pt2, pt1

                x_l_hand = max(pt1[0] + (width_hand // 5), 0)
                y_l_hand = max(pt1[1], 0)
                x_l_hand_end = max(pt1[0] - width_hand, 0)
                y_l_hand_end = max(pt1[1] + height_hand, 0)

                x_r_hand = max(pt2[0] - (width_hand // 5), 0)
                y_r_hand = max(pt2[1], 0)
                x_r_hand_end = min(pt2[0] + width_hand, frame.shape[1])
                y_r_hand_end = min(pt2[1] + height_hand, frame.shape[0])

                contours_info = []

                if (0 <= x_l_hand < frame.shape[1] and 0 <= y_l_hand < frame.shape[0] and
                        0 <= x_l_hand_end <= frame.shape[1] and 0 <= y_l_hand_end <= frame.shape[0]):

                    mask_l_hand = np.zeros(fgMask.shape[:2], dtype="uint8")
                    cv2.rectangle(mask_l_hand, (x_l_hand, y_l_hand), (x_l_hand_end, y_l_hand_end), 255, -1)
                    fgMask_l_hand = cv2.bitwise_and(fgMask, fgMask, mask=mask_l_hand)
                    contours_l_hand, hierarchy_l_hand = cv2.findContours(fgMask_l_hand, cv2.RETR_EXTERNAL,
                                                                         cv2.CHAIN_APPROX_NONE)

                    if len(contours_l_hand) > 0:
                        largest_contour_l = max(contours_l_hand, key=cv2.contourArea)
                        if is_contour_connected_to_keypoint(largest_contour_l, pt1):
                            x_contour, y_contour, width, height = cv2.boundingRect(largest_contour_l)
                            contours_info.append(("Left", pt1, x_contour, y_contour, width, height, largest_contour_l))

                if (0 <= x_r_hand < frame.shape[1] and 0 <= y_r_hand < frame.shape[0] and
                        0 <= x_r_hand_end <= frame.shape[1] and 0 <= y_r_hand_end <= frame.shape[0]):

                    mask_r_hand = np.zeros(fgMask.shape[:2], dtype="uint8")
                    cv2.rectangle(mask_r_hand, (x_r_hand, y_r_hand), (x_r_hand_end, y_r_hand_end), 255, -1)
                    fgMask_r_hand = cv2.bitwise_and(fgMask, fgMask, mask=mask_r_hand)
                    contours_r_hand, hierarchy_r_hand = cv2.findContours(fgMask_r_hand, cv2.RETR_EXTERNAL,
                                                                         cv2.CHAIN_APPROX_NONE)

                    if len(contours_r_hand) > 0:
                        largest_contour_r = max(contours_r_hand, key=cv2.contourArea)
                        if is_contour_connected_to_keypoint(largest_contour_r, pt2):
                            x_contour, y_contour, width, height = cv2.boundingRect(largest_contour_r)
                            contours_info.append(("Right", pt2, x_contour, y_contour, width, height, largest_contour_r))

                if contours_info:
                    # Sort contours_info based on area in descending order and select the largest
                    contours_info.sort(key=lambda info: info[4] * info[5], reverse=True)
                    # print(contours_info)
                    selected_hand, selected_pt, x_contour, y_contour, width, height, selected_contour = contours_info[0]

                    cv2.rectangle(frame, (x_contour, y_contour), (x_contour + width, y_contour + height), (0, 0, 255), 2)
                    cv2.putText(frame, f"{id} {selected_hand} Hand obj", (x_contour, y_contour - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                    tracker_bbox = (x_contour, y_contour, width, height)
                    tracker = cv2.TrackerKCF_create()
                    thrown=is_throwing_action(tracker_bbox,selected_pt)

                    obj_center = (x_contour + width // 2, y_contour + height // 2)
                    print(selected_pt)
                    print(obj_center)
                    # cv2.line(frame,obj_center,(int(selected_pt[0]),int(selected_pt[0])),(255, 0, 0), 2)
                    tracker.init(tracking_frame, tracker_bbox)
                    tracker_initialized = True
                    current_hand = selected_hand

    if thrown:
        print("thrown")
        cv2.putText(frame, f"Throwing action detected", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Frame', frame)
    # cv2.imshow('Frame2', fgMask)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
