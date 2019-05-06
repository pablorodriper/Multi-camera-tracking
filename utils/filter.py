import cv2
import numpy as np
from utils.detection import Detection

PIXELS_SHIFT = 8
WINDOW_FRAME = 15
def filtering_parked(detections, video_path):
    # for detection in detections:
    #     print(detection)
    capture = cv2.VideoCapture(video_path)
    n_frame = 0
    final_det = list()

    while capture.isOpened():
        valid, frame = capture.read()
        if not valid:
            break
        fr = frame.copy()
        detections_on_frame = [x for x in detections if x.frame == n_frame]
        detections_on_next_frame = [x for x in detections if x.frame == n_frame+PIXELS_SHIFT]

        detections_bboxes = [o.bbox for o in detections_on_frame]
        detections_bboxes_next = [o.bbox for o in detections_on_next_frame]

        new_candidates = []
        # print(len(detections_bboxes))
        for candidate in detections_bboxes:
            minc, minr, maxc, maxr = candidate
            new_candidate = [minc, minr, maxc, maxr, 1]
            for next_candidate in detections_bboxes_next:
                n_minc, n_minr, n_maxc, n_maxr = next_candidate
                if (minc-WINDOW_FRAME <= n_minc <= minc+WINDOW_FRAME) and (minr-WINDOW_FRAME <= n_minr <= minr+WINDOW_FRAME) and (maxc-WINDOW_FRAME <= n_maxc <= maxc+WINDOW_FRAME) and (n_maxr-WINDOW_FRAME <= n_maxr <= n_maxr+WINDOW_FRAME):
                    if new_candidate[4] != 0:
                        new_candidate[4] = 0
            new_candidates.append(new_candidate)

        #no_parked = [x for x in new_candidates if x[4] != 0]
        # print(len(new_candidates))

        for n,candidate in enumerate(new_candidates):
            if candidate[4] != 0:
                minc, minr, maxc, maxr, parked = candidate

                final_det.append(Detection(detections_on_frame[n].frame, detections_on_frame[n].label, minc, minr, maxc-minc, maxr-minr, detections_on_frame[n].confidence))

        # det_on_frame = [x for x in final_det if x.frame == n_frame]
        # det_bboxes = [o.bbox for o in det_on_frame]
        # for candidate in det_bboxes:
        #     minc, minr, maxc, maxr = candidate
        #     cv2.rectangle(fr, (minc, minr), (maxc, maxr), (0, 0, 255), 8)  # Red
        # cv2.imshow('fr', fr)
        # cv2.waitKey(0)
        n_frame += 1

    return final_det


def filtering_nms(detections, video_path):
    capture = cv2.VideoCapture(video_path)
    n_frame = 0
    final_det = list()

    while capture.isOpened():
        valid, frame = capture.read()
        if not valid:
            break
        fr = frame.copy()
        detections_on_frame = [x for x in detections if x.frame == n_frame]

        detections_bboxes = [o.bbox for o in detections_on_frame]
        pick = non_max_suppression_fast(detections_bboxes)
        for n,candidate in enumerate(detections_bboxes):
            if n in pick:
                minc, minr, maxc, maxr = candidate
                w = maxc-minc
                h=maxr-minr
                if w<75 or h<75:
                    pass
                else:
                    #cv2.rectangle(fr, (minc, minr), (maxc, maxr), (0, 0, 255), 8)  # Red
                    final_det.append(Detection(detections_on_frame[n].frame, detections_on_frame[n].label, minc, minr, maxc - minc,
                              maxr - minr, detections_on_frame[n].confidence))
        # cv2.imshow('fr', fr)
        # cv2.waitKey(10)
        n_frame += 1

    return final_det


    # Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh=0.2):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes

    x1 = np.asarray(list(zip(*boxes))[0], dtype=float)
    y1 =  np.asarray(list(zip(*boxes))[1], dtype=float)
    x2 =  np.asarray(list(zip(*boxes))[2], dtype=float)
    y2 =  np.asarray(list(zip(*boxes))[3], dtype=float)

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type)
    # boxes[pick].astype("int")

    return pick



