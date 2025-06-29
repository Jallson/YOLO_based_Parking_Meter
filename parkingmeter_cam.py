#!/usr/bin/env python
import device_patches
import cv2
import os
import time
import sys, getopt
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner
from collections import defaultdict

runner = None
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def help():
    print('python parkingmeter_video.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def get_webcams():
    port_ids = []
    print("Probing camera ports 0-4...")
    for port in range(5):
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret, _ = camera.read()
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print(f"SUCCESS: Camera '{backendName}' ({int(w)}x{int(h)}) found at port {port}")
                port_ids.append(port)
            camera.release()
    print("Probe complete.")
    return port_ids

def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    bb1_area = w1 * h1
    bb2_area = w2 * h2
    union_area = bb1_area + bb2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def get_zone(x, y):
    # Zone layout for 320x320 image:
    if 50 <= x <= 100 and 0 <= y <= 105:
        return 'A'
    elif 10 <= x <= 100 and 106 <= y <= 320:
        return 'B'
    elif 180 <= x <= 270 and 0 <= y <= 105:
        return 'C'
    elif 190 <= x <= 320 and 105 <= y <= 320:
        return 'D'
    else:
        return None

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) < 1:
        help()
        sys.exit(2)

    model = args[0]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']

            if len(args) >= 2:
                videoCaptureDeviceId = int(args[1])
            else:
                port_ids = get_webcams()
                if len(port_ids) == 0:
                    raise Exception('Cannot find any webcams')
                if len(port_ids) > 1:
                    raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use this script")
                videoCaptureDeviceId = int(port_ids[0])

            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret = camera.read()[0]
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." % (backendName, h, w, videoCaptureDeviceId))
                camera.release()
            else:
                raise Exception("Couldn't initialize selected camera.")

            next_object_id = 0
            stop_tracker = {}

            for res, img in runner.classifier(videoCaptureDeviceId):
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # The 'res' object from runner.classifier already contains the classification results.
                # We can directly use it. 'img' is the corresponding frame.
                
                current_time = time.time()
                matched_ids = set()

                # Clean up stale trackers (not seen for >3 seconds)
                stale_ids = [obj_id for obj_id, tracker in stop_tracker.items()
                             if current_time - tracker['last_seen_time'] > 3]
                for obj_id in stale_ids:
                    del stop_tracker[obj_id]

                if "bounding_boxes" in res["result"].keys():
                    current_bbs = res["result"]["bounding_boxes"]

                    for bb in current_bbs:
                        if bb['value'] < 0.6:
                            continue

                        x, y, w, h = bb['x'], bb['y'], bb['width'], bb['height']
                        current_box = (x, y, w, h)
                        center = (x + w // 2, y + h // 2)
                        zone = get_zone(center[0], center[1])

                        best_match_id = None
                        best_score = 0
                        for obj_id, tracker in stop_tracker.items():
                            prev_box = tracker['last_box']
                            prev_center = tracker['last_pos']

                            iou_score = iou(current_box, prev_box)
                            center_dist = np.linalg.norm(np.array(center) - np.array(prev_center))
                            size_ratio = abs((w * h) - (prev_box[2] * prev_box[3])) / (w * h + 1e-5)

                            if iou_score > 0.4 and center_dist < 50 and size_ratio < 0.5:
                                if iou_score > best_score:
                                    best_score = iou_score
                                    best_match_id = obj_id

                        if best_match_id is not None:
                            tracker = stop_tracker[best_match_id]
                            time_since_last_seen = current_time - tracker['last_seen_time']
                            moved = np.linalg.norm(np.array(center) - np.array(tracker['last_pos'])) > 10

                            if moved:
                                tracker['start_time'] = current_time
                                tracker['stop_duration'] = 0
                                tracker['last_box'] = current_box
                                tracker['last_pos'] = center
                                tracker['zone'] = zone
                            else:
                                tracker['stop_duration'] = current_time - tracker['start_time']

                            tracker['last_seen_time'] = current_time
                            matched_ids.add(best_match_id)

                            stop_duration = tracker['stop_duration']
                            zone = tracker.get('zone')
                            color = (0, 255, 60)
                            text = ""

                            if zone == 'A' and stop_duration > 30:
                                color = (250, 0, 0)
                                text = f"No Park"
                            elif zone == 'B' and stop_duration > 100:
                                color = (250, 0, 0)
                                text = f"No Park"
                            elif zone == 'C' and stop_duration > 5:
                                color = (250, 0, 0)
                                text = f"No Park"
                            elif zone == 'D' and stop_duration >= 5:
                                dollars = int((stop_duration // 10) * 5)
                                color = (255, 150, 0)
                                text = f"${dollars}"
                            elif stop_duration >= 5:
                                text = f"Free {int(stop_duration)}m"

                            if stop_duration >= 5:
                                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                                if text:
                                    cv2.putText(img, text, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        else:
                            stop_tracker[next_object_id] = {
                                'start_time': current_time,
                                'last_seen_time': current_time,
                                'stop_duration': 0,
                                'last_box': current_box,
                                'last_pos': center,
                                'zone': zone
                            }
                            next_object_id += 1

                if show_camera:
                    cv2.imshow('parking_meter', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break
        finally:
            if runner:
                runner.stop()

if __name__ == "__main__":
    main(sys.argv[1:])