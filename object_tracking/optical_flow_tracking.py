import cv2
import numpy as np
import pickle
from utils.reading import read_annotations_file
from utils.detection import Detection
from typing import List
from object_tracking.block_matching import block_matching_optical_flow
from evaluation.bbox_iou import detection_iou
import math


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class TrackingOF:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.ordered_tracks = []
        self.frame_idx = 0
        self.prev_gray = None

    def track_optical_flow_sequence(self, video_path, visualise_of=False):
        """
        Based on https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py
        """
        capture = cv2.VideoCapture(video_path)
        track_flow = []

        while capture.isOpened():
            valid, frame = capture.read()
            if not valid:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            frame_tracks = []

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    frame_tracks.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (255, 10, 180))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                        # frame_tracks.append((x,y))
            # print("Frame: {}, tracks: {}\n".format(self.frame_idx, frame_tracks))
            self.frame_idx += 1
            self.prev_gray = frame_gray
            self.ordered_tracks.append(frame_tracks)
            if visualise_of:
                cv2.imshow('lk_track', vis)
                ch = cv2.waitKey(1)
                if ch == 27:
                    break

    def check_optical_flow(self, detections_on_frame, frame_id):
        if frame_id==0:
            return detections_on_frame
        else:
            of_points = self.ordered_tracks[frame_id]
            # of_velocities = [x-y for x,y in zip(self.ordered_tracks[frame_id],self.ordered_tracks[frame_id-1])] 
            # of_data = [of_points, of_velocities]
            for det in detections_on_frame:
                relevant_data = [p for p in of_points if det.xtl<p[0]<det.xtl+det.width]
                relevant_data = [p for p in relevant_data if det.ytl<p[1]<det.ytl+det.height]
                speed = math.sqrt(relevant_data[0]^2+relevant_data[1]^2)
                print("Frame: {}, speed: {}".format(frame_id, speed))
            return detections_on_frame


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


