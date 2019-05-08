from itertools import product
import numpy as np
import os
import pickle

# Tracking
from evaluation.evaluation_funcs import compute_mAP
from object_tracking.tracking import track_objects, compute_embeddings_for_tracked_detections
from object_tracking.multi_camera import match_tracks_by_frame, create_dataset, predict_bbox, bboxes_correspondences, match_tracks
#from utils.plotting import draw_video_bboxes
from utils.reading import read_annotations_file, read_homography_matrix, get_framenum, get_timestamp
from utils.filter import filtering_parked,filtering_nms
#from object_tracking.kalman_tracking import kalman_track_objects
from utils.detection import Detection
import motmetrics as mm


if __name__ == '__main__':

    display_frames = False
    export_frames = True
    load_pkl = False

    root_path = os.path.abspath('')
    dataset_path = os.path.join(root_path, 'datasets', 'aic19-track1-mtmc-train')
    #sequences = [('train','S01'), ('test','S02'), ('train','S03'), ('train','S04'), ('test','S05')]
    # For testing
    sequences = [('train', 'S01')]  # [('train','S01'), ('test','S02'), ('train','S03'), ('train','S04'), ('test','S05')]

    #cameras_path = ["../datasets/aic19-track1-mtmc-train/train/S01/c001/", "../datasets/aic19-track1-mtmc-train/train/S01/c002/"]
    video_challenge_path = "/vdo.avi"
    detections_challenge_path = "/det/"
    #detectors = ["/det_ssd512.txt", "/det_mask_rcnn.txt", "/det_yolo3.txt"]
    detector = "det_ssd512.txt"
    groundtruth_challenge_path = "/gt/gt.txt"
    homography_path_start = os.path.join(root_path, 'datasets', 'calibration')
    homography_path = "/calibration.txt"
    path_experiment = '../../siamese/experiments/Wed_Apr_10_08_49_36_2019'

    #timestamps = [0, 1.640]
    #framenum = [1955, 2110]
    fps = 10

    for split, sequence in sequences:

        sequence_path = os.path.join(dataset_path, split, sequence)

        # Get sorted camera folders
        cameras = [f for f in os.listdir(sequence_path) if not f.startswith('.')]
        cameras_path = [os.path.join(sequence_path, f) for f in cameras]
        cameras_path.sort()

        timestamps = get_timestamp(dataset_path, sequence)
        framenum = get_framenum(dataset_path, sequence)

        tracked_detections = {}
        tracks_by_camera = {}
        embeddings = {}
        homography_cameras = {}
        groundtruth_list = {}
        video_path = {}

        for cam_num, camera in enumerate(cameras_path):

            if camera[-4:] == 'c015':
                fps = 8

            video_path[cam_num] = camera + video_challenge_path
            detections_list, _ = read_annotations_file(camera + detections_challenge_path + detector, camera + video_challenge_path)

            groundtruth_list[cam_num], _ = read_annotations_file(camera + groundtruth_challenge_path, camera + video_challenge_path)
            homography_cameras[cam_num] = read_homography_matrix(homography_path_start + camera[-5:] + homography_path)

            print("\nComputing tracking by overlap")
            if load_pkl and os.path.exists('output/pickle/detections' + str(sequence) + str(cam_num)+'.pkl') and os.path.exists('output/pickle/tracks' + str(sequence) + str(cam_num)+'.pkl'):
                with open('output/pickle/detections' + str(sequence) + str(cam_num)+'.pkl', 'rb') as p:
                    print("Reading tracked detections from pkl")
                    tracked_detections[cam_num] = pickle.load(p)
                    #print(tracked_detections[cam_num])
                    print("Tracked detections loaded\n")

                with open('output/pickle/tracks' + str(sequence) + str(cam_num)+'.pkl', 'rb') as p:
                    print("Reading tracks from pkl")
                    tracks_by_camera[cam_num] = pickle.load(p)
                    print("Tracks loaded\n")

                with open('output/pickle/embeddings' + str(sequence) + str(cam_num)+'.pkl', 'rb') as p:
                    print("Reading tracks from pkl")
                    embeddings[cam_num] = pickle.load(p)
                    print("Embeddings loaded\n")

            else:
                detections_list = filtering_nms(detections_list, video_path[cam_num])
                detections_list = filtering_parked(detections_list, video_path[cam_num])

                tracked_detections[cam_num], tracks_by_camera[cam_num], embeddings[cam_num] = track_objects(camera + video_challenge_path, detections_list, groundtruth_list[cam_num],
                                                display=display_frames, export_frames=export_frames, idf1=False, name_pkl=str(sequence) + str(cam_num))

                #embeddings[cam_num] = compute_embeddings_for_tracked_detections(tracked_detections[cam_num], video_path[cam_num], name_pkl=str(sequence) + str(cam_num))
            # Compute mAP
            compute_mAP(groundtruth_list[cam_num], tracked_detections[cam_num])



        #correspondences = bboxes_correspondences(groundtruth_list, timestamps, framenum, fps)
        #multicamera_tracks = match_tracks(tracked_detections, tracks_by_camera, homography_cameras, timestamps, framenum, fps, video_path, path_experiment, embeddings)
        #match_tracks_by_frame(tracked_detections, homography_cameras, timestamps, framenum, fps, cameras_path[0] + video_challenge_path, cameras_path[1] + video_challenge_path, correspondences)


        with open('output/pickle/multitracksresultsS04_2.pkl', 'rb') as p:
            print("Reading tracks from pkl")
            multicamera_tracks = pickle.load(p)
            print("Tracks loaded\n")

        print('Compute IDF1 for multicamera tracking')
        acc = mm.MOTAccumulator(auto_id=True)
        for cam_num, camera in enumerate(cameras_path):

            tracks = tracks_by_camera[cam_num]
            detections_list = []
            for track in tracks:
                for detection in track.detections:
                    detections_list.append(detection)

            gt_list = groundtruth_list[cam_num]
            for n_frame in range(framenum[cam_num] + 1):

                detec_bboxes = []
                detec_ids = []
                for id in multicamera_tracks.keys():
                    id_cams = multicamera_tracks[id][0]
                    if len(id_cams) > 1:
                        continue
                    for id_cam in id_cams:
                        track = [x for x in tracks if x.id == id_cam]
                        if len(track)>0:
                            detections_on_frame = [x for x in track[0].detections if x.frame == n_frame]
                            for detec in detections_on_frame:
                                detec_bboxes.append(detec.bbox)
                                detec_ids.append(id)

                gt_on_frame = [x for x in gt_list if x.frame == n_frame]

                gt_bboxes = []
                gt_ids = []
                for gt in gt_on_frame:
                    gt_bboxes.append(gt.bbox)
                    gt_ids.append(gt.track_id)

                # detec_bboxes = []
                # detec_ids = []
                # for det in detections_on_frame:
                #     detec_bboxes.append(det.bbox)
                #     detec_ids.append(det.track_id)

                mm_gt_bboxes = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]] for
                                bbox in gt_bboxes]
                mm_detec_bboxes = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]]
                                   for bbox in detec_bboxes]
                distances_gt_det = mm.distances.iou_matrix(mm_gt_bboxes, mm_detec_bboxes, max_iou=1.)
                acc.update(gt_ids, detec_ids, distances_gt_det)

        print(acc.mot_events)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
        print(summary.to_string())
