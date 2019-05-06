import os
import cv2
import numpy as np
from numpy.linalg import inv
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import pickle


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.plotting import visualize_tracks, visualize_tracks_opencv
from evaluation.bbox_iou import bbox_iou, bbox_intersection, intersection_over_area, bbox_area
from utils.reading import read_homography_matrix, read_annotations_file
from utils.detection import Detection
from siamese.one_tower import One_tower


def create_dataset(gt_detections, timestamps, framenum, fps):
    train_data = list()
    train_labels = list()

    #Assume cameras are in order of ascendent timestamp
    for frame in range(framenum[0]+1):
        detecs_frame_0 = [detec for detec in gt_detections[0] if detec.frame == frame]

        for cam in range(1, len(framenum)):
            if frame >= int(timestamps[cam]*fps) and frame <= framenum[cam]:
                detecs_frame_1 = [detec for detec in gt_detections[1] if detec.frame == frame - int(timestamps[cam]*fps)]

                for detec_0 in detecs_frame_0:
                    matches = [detec_1 for detec_1 in detecs_frame_1 if detec_1.track_id == detec_0.track_id]
                    if len(matches) == 1:
                        train_data.append(np.array(detec_0.bbox))
                        train_labels.append(np.array(matches[0].bbox))


    return np.vstack(train_data), np.vstack(train_labels)


def bboxes_correspondences(gt_detections, timestamps, framenum, fps):
    correspondences = []

    #Assume cameras are in order of ascendent timestamp
    for frame in range(framenum[0]+1):
        detecs_frame_0 = [detec for detec in gt_detections[0] if detec.frame == frame]

        for cam in range(1, len(framenum)):
            if frame >= int(timestamps[cam]*fps) and frame <= framenum[cam]:
                detecs_frame_1 = [detec for detec in gt_detections[1] if detec.frame == frame - int(timestamps[cam]*fps)]

                for detec_0 in detecs_frame_0:
                    matches = [detec_1 for detec_1 in detecs_frame_1 if detec_1.track_id == detec_0.track_id]
                    if len(matches) == 1:

                        correspondences.append((detec_0.bbox, matches[0].bbox))

    return correspondences


def build_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=[4]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4))

    optimizer = RMSprop(0.01)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def predict_bbox(train_data, train_labels):
    model = build_model()
    #result = model.predict(train_data)

    class PrintDot(Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            #print('.', end='')

    EPOCHS = 100000
    history = model.fit(
        train_data, train_labels, steps_per_epoch=int(len(train_data)*0.8/32),
        epochs=EPOCHS, validation_split=0.2, validation_steps=int(len(train_data)*0.2/32), verbose=0,
        callbacks=[PrintDot()])

    print(history.history.keys())
    print(history.history['loss'][-1])

    # plt.plot(history.history['epoch'], history.history['mean_absolute_error'])
    # plt.plot(history.history['epoch'], history.history['val_mean_absolute_error'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    # plt.savefig('first_try.jpg')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.savefig('loss_valloss_2.jpg')
    plt.close()


def intersection(u, v):
    """
    Compare histograms based on their intersection.

    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.

    Returns:
        float: distance between histograms.
    """
    return cv2.compareHist(np.array(u), np.array(v), cv2.HISTCMP_BHATTACHARYYA)


def compute_3d_detecs(detections, homography):
    detecs_3d = {}
    for detec in detections:
        minc, minr, maxc, maxr = detec.bbox
        x = (maxc + minc)/2
        y = maxr
        H = np.array(homography)
        Hinv = inv(H)
        pos_2d = np.array([x, y, 1])
        detecs_3d[detec.track_id] =dict(position=Hinv.dot(pos_2d), histogram=detec.histogram)

    return detecs_3d


def apply_homography_to_point(x, y, H1, H2):
    pos_1 = np.array([x, y, 1])
    det_0_hom = inv(H1).dot(pos_1)
    det_0_hom = np.array([det_0_hom[0] / det_0_hom[2], det_0_hom[1] / det_0_hom[2], 1])
    det_1_hom = H2.dot(det_0_hom)

    return [det_1_hom[0]/det_1_hom[2], det_1_hom[1]/det_1_hom[2], 1]


def get_homography_IoU(det_0, det_1, homography1, homography2):

    predicted_correspondence = transform_detection(det_0, homography1, homography2)
    predicted_bbox_1 = predicted_correspondence.bbox
    IoU = bbox_iou(det_1.bbox, predicted_bbox_1)

    return IoU


def transform_detection(det_0, homography1, homography2):
    minc, minr, maxc, maxr = det_0.bbox
    H1 = np.array(homography1)
    H2 = np.array(homography2)

    x = minc
    y = maxr
    det_1_hom = apply_homography_to_point(x, y, H1, H2)

    x_br = maxc
    y_br = maxr
    det_1_hom_br = apply_homography_to_point(x_br, y_br, H1, H2)

    original_ratio = (maxr - minr) / (maxc - minc)  # height/width of bbox
    width_transformed = abs(det_1_hom[0] - det_1_hom_br[0])
    height_transformed = original_ratio * width_transformed

    predicted_bbox_1 = [min(det_1_hom[0], det_1_hom_br[0]), min(det_1_hom[1], det_1_hom_br[1]) - height_transformed,
                        max(det_1_hom[0], det_1_hom_br[0]), max(det_1_hom[1], det_1_hom_br[1])]

    predicted_correspondence = Detection(det_0.frame, det_0.label, predicted_bbox_1[0], predicted_bbox_1[1], predicted_bbox_1[2]-predicted_bbox_1[0], predicted_bbox_1[3]-predicted_bbox_1[1], histogram=det_0.histogram)
    return predicted_correspondence


def check_inside_image(detection, h=1080, w=1920):
    minc, minr, maxc, maxr = detection.bbox

    if minc<0 or minr < 0:
        return False

    if maxc>w or maxr > h:
        return False

    return True


def match_correspondence(detection, correspondences):
    highest_IoU = 0
    for correspondence in correspondences:
        IoU = bbox_iou(correspondence[0], detection.bbox)

        if IoU > highest_IoU:
            highest_IoU = IoU
            best_match = correspondence

    if highest_IoU > 0:
        return best_match
    else:
        return None


def get_correspondence_IoU(det_0, det_1, correspondences):
    closer_det = match_correspondence(det_0, correspondences)
    if closer_det:
        closer_det_1 = closer_det[1]
        IoU = bbox_iou(closer_det_1, det_1.bbox)
        return IoU
    return 0


def optimize_matching(match_tracks):
    final_matches = {}
    used = []
    for track_0 in match_tracks:
        for track_1 in match_tracks[track_0].most_common():
            if track_1[0] not in used:
                final_matches[track_0] = track_1[0]
                used.append(track_1[0])
                break

    print(final_matches)
    return final_matches


def visualize_matches(matched_tracks, cameras_tracks_0, cameras_tracks_1, video_path_0, video_path_1):
    colors = np.random.rand(500, 3)
    colors[499] = [1, 0, 0]
    capture = cv2.VideoCapture(video_path_0)
    n_frame = 0

    while capture.isOpened():
        frame_tracks_0 = defaultdict(list)
        valid, image = capture.read()
        if not valid:
            break
        for detec in cameras_tracks_0:
            if detec.frame == n_frame:
                if matched_tracks.get(detec.track_id):
                    for track2 in matched_tracks[detec.track_id]:
                        frame_tracks_0[track2].append(detec.bbox)
                # else:
                #     frame_tracks_0[499] = dict(bbox=detec.bbox)

        visualize_tracks_opencv(image, frame_tracks_0, colors, export_frames=True,
                                export_path="output_frames/c001/frame_{:04d}.png".format(n_frame))
        # if n_frame%2 == 0:
        #     visualize_tracks(image, frame_tracks_0, colors, display=False)
        n_frame += 1

    capture = cv2.VideoCapture(video_path_1)
    n_frame = 0

    while capture.isOpened():
        frame_tracks_1 = defaultdict(list)
        valid, image = capture.read()
        if not valid:
            break
        for detec in cameras_tracks_1:
            if detec.frame == n_frame:
                if any(detec.track_id in lst for lst in matched_tracks.values()):
                    #if detec.track_id in matched_tracks.values():
                    frame_tracks_1[detec.track_id].append(detec.bbox)
                # else:
                #     frame_tracks_1[499] = dict(bbox=detec.bbox)

        visualize_tracks_opencv(image, frame_tracks_1, colors, export_frames=True,
                                export_path="output_frames/c002/frame_{:04d}.png".format(n_frame))
        # if n_frame%2 == 0:
        #     visualize_tracks(image, frame_tracks_1, colors, display=False)
        n_frame += 1


def distances_to_tracks(track1, tracks_camera2, homographies, camera1, camera2):
    match_tracks_metrics = np.ones(len(tracks_camera2))*(-1)
    transformed_track1 = []
    for detec in track1.detections:
        transformed_detec = transform_detection(detec, homographies[camera1], homographies[camera2])
        if check_inside_image(transformed_detec):
            transformed_track1.append(transformed_detec)
    for track2 in tracks_camera2:
        comparison_bboxes = []
        total_area_projected_bboxes = 0
        for detec1 in transformed_track1:
            total_area_projected_bboxes += bbox_area(detec1.bbox)
            for detec2 in track2.detections:
                comparison_bboxes.append((detec1.frame, detec2.frame, intersection_over_area(detec1.bbox, detec2.bbox),
                                          bbox_intersection(detec1.bbox, detec2.bbox), bbox_area(detec1.bbox)))
        sorted_bbox_comparisons = sorted(comparison_bboxes, key=lambda relation: relation[2], reverse=True)
        best_match = None
        tracks_intersection = 0
        area_projected_bboxes = 0
        num_matches = 0
        for match in sorted_bbox_comparisons:
            if match[2] > 0:
                if best_match == None:
                    best_match = match
                    last_id1 = match[0]
                    last_id2 = match[1]
                    first_id1 = match[0]
                    first_id2 = match[1]
                    num_matches += 1
                    tracks_intersection += match[3]
                    area_projected_bboxes += match[4]

                else:
                    id1 = match[0]
                    id2 = match[1]
                    if id1 > last_id1 and id2 > last_id2:
                        last_id1 = id1
                        last_id2 = id2
                        num_matches += 1
                        tracks_intersection += match[3]
                        area_projected_bboxes += match[4]
                    elif id1 < first_id1 and id2 < first_id2:
                        first_id1 = id1
                        first_id2 = id2
                        num_matches += 1
                        tracks_intersection += match[3]
                        area_projected_bboxes += match[4]
        if total_area_projected_bboxes > 0:
            match_tracks_metrics[track2.id] = tracks_intersection / total_area_projected_bboxes

    return match_tracks_metrics


def get_candidates_by_trajectory_in_camera2(match_tracks_metrics, intersection_threshold = 0.3):
    candidates_by_trajectory = []
    for track2id, dist in enumerate(match_tracks_metrics):
        if dist > intersection_threshold or dist == -1:
            #print(dist)
            candidates_by_trajectory.append(track2id)

    return candidates_by_trajectory


def interval_times_intersect(start_time, end_time, t1, t2):
    if t1 > end_time or t2 < start_time:
        return False
    return True


def filter_by_time_coherence(candidates_by_trajectory, track1, tracks_camera2, camera1, camera2, timestamps, fps, framenum, time_margin = 150):
    sorted_detections_track1 = sorted(track1.detections, key=lambda x: x.frame, reverse=True)
    start_time = sorted_detections_track1[0].frame - int(timestamps[camera1]*fps) - time_margin
    end_time = sorted_detections_track1[-1].frame - int(timestamps[camera1]*fps) + time_margin
    final_candidates = []
    for track2 in tracks_camera2:
        if track2.id in candidates_by_trajectory:
            sorted_detections_track2 = sorted(track2.detections, key=lambda x: x.frame, reverse=True)
            t1 = sorted_detections_track2[0].frame - int(timestamps[camera2]*fps)
            t2 = sorted_detections_track2[-1].frame - int(timestamps[camera2]*fps)
            if interval_times_intersect(start_time, end_time, t1, t2):
                final_candidates.append(track2.id)
    return final_candidates


def compute_track_embedding(detections, camera, video_path, path_experiment, one_tower, embeddings, image_size = 64):
    #capture = cv2.VideoCapture(video_path)
    n_frame = 0
    sum_embeds = 0
    num_detections = 0
    embed_cam = embeddings[camera]
    for detec in detections:
        #print('Detection to embed: {}'.format(detec))
        for emb_det in embed_cam:
            if emb_det['detection'].frame == detec.frame:
                embed = emb_det['embedding']
                embed = embed[0]

        #print('embedding: {}'.format(embed))
        sum_embeds += embed
        num_detections += 1

    return sum_embeds/num_detections


def get_candidates_embeddings(reference_track, reference_camera, candidates_matches, cameras_tracks, video_path, path_experiment, one_tower, embeddings):
    candidate_tracks_embeddings = []
    candidate_tracks_ids = []
    emb_ref = compute_track_embedding(reference_track.detections, reference_camera, video_path[reference_camera], path_experiment, one_tower, embeddings)
    candidate_tracks_embeddings.append(emb_ref)
    candidate_tracks_ids.append((reference_camera, reference_track.id))
    for camera in candidates_matches:
        candidates_camera = candidates_matches[camera]
        print('Candidates camera: {}'.format(candidates_camera))
        for track in cameras_tracks[camera]:
            if track.id in candidates_camera:
                emb = compute_track_embedding(track.detections, camera, video_path[camera], path_experiment, one_tower, embeddings)
                candidate_tracks_ids.append((camera, track.id))
                candidate_tracks_embeddings.append(emb)
    return candidate_tracks_embeddings, candidate_tracks_ids


def compute_distances_to_candidates(candidate_tracks_embeddings):
    ref_track = candidate_tracks_embeddings[0]
    print('Distances:')
    dist = 0
    num = 0
    for track_emb in candidate_tracks_embeddings[1:]:
        dist += distance.euclidean(ref_track, track_emb)
        print(distance.euclidean(ref_track, track_emb))
        num += 1
    if num > 0:
        print('Mean distance: {}'.format(dist/num))


def cluster_embeddings(tracks_embeddings):
    X = np.array(tracks_embeddings)
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
    assignations = clustering.labels_
    return assignations


def assign_track(candidates_embeddings, candidate_tracks_ids, already_matched_tracks):
    multitracks_id = defaultdict(list)
    assignations = cluster_embeddings(candidates_embeddings)
    assigned_label = assignations[0]
    if assigned_label != -1:
        for n, candidate in enumerate(candidate_tracks_ids):
            if assignations[n] == assigned_label:
                camera = candidate[0]
                trackid = candidate[1]
                if trackid not in already_matched_tracks[camera]:
                    already_matched_tracks[camera].append(trackid)
                    multitracks_id[camera].append(trackid)
        return already_matched_tracks, multitracks_id
    return already_matched_tracks, None


def match_tracks(tracked_detections, cameras_tracks, homographies, timestamps, framenum, fps, video_path, path_experiment, embeddings_by_camera):
    general_track_id = 1
    multitrack_assignations = {}
    one_tower = One_tower(64, 64)
    embeddings = defaultdict(list)
    already_matched_tracks = defaultdict(list)
    for camera1 in cameras_tracks:
        print(camera1)
        tracks_camera1 = cameras_tracks[camera1]
        for track1 in tracks_camera1:
            if track1.id not in already_matched_tracks[camera1]:
                candidate_matches = {}
                for camera2 in cameras_tracks:
                    if camera2 != camera1:
                        tracks_camera2 = cameras_tracks[camera2]
                        not_matched_tracks_camera2 = [t for t in tracks_camera2 if (t not in already_matched_tracks[camera2])]
                        match_tracks_metrics = distances_to_tracks(track1, not_matched_tracks_camera2, homographies, camera1, camera2)
                        print('Trajectory distances')
                        print(match_tracks_metrics)
                        candidates_by_trajectory = get_candidates_by_trajectory_in_camera2(match_tracks_metrics)
                        #print(candidates_by_trajectory)
                        candidates_by_time_and_trajectory = filter_by_time_coherence(candidates_by_trajectory, track1, tracks_camera2, camera1, camera2, timestamps, fps, framenum)
                        print('Candidates in camera: {}'.format(candidates_by_time_and_trajectory))
                        candidate_matches[camera2] = candidates_by_time_and_trajectory
                print('Candidate_matches: {}'.format(candidate_matches))
                candidates_embeddings, candidates_tracksids = get_candidates_embeddings(track1, camera1, candidate_matches, cameras_tracks, video_path, path_experiment, one_tower, embeddings_by_camera)
                compute_distances_to_candidates(candidates_embeddings)
                already_matched_tracks, multitrack = assign_track(candidates_embeddings, candidates_tracksids, already_matched_tracks)
                if multitrack != None:
                    multitrack_assignations[general_track_id] = multitrack
                    general_track_id += 1

                print(candidate_matches)

        #visualize_matches(candidate_matches, tracked_detections[0], tracked_detections[1], video_path_0, video_path_1)
    with open('output/pickle/multitracksresultsS04_2.pkl', 'wb') as f:
        pickle.dump(multitrack_assignations, f, protocol=2)
    print(multitrack_assignations)


def match_tracks_by_frame(cameras_tracks, homographies, timestamps, framenum, fps, video_path_0, video_path_1, correspondences):
    dist = defaultdict(list)

    #Assume cameras are in order of ascendent timestamp
    for frame in range(framenum[0]+1):
        detecs_frame_0 = [detec for detec in cameras_tracks[0] if detec.frame == frame]
        #detecs_3d_0 = compute_3d_detecs(detecs_frame_0, homographies[0])
        for cam in range(1, len(framenum)):
            for frame2 in range(frame - int(timestamps[cam]*fps)-5, frame - int(timestamps[cam]*fps) + 6):
                if frame2 >= 0 and frame2 <= framenum[cam]:
                    detecs_frame_1 = [detec for detec in cameras_tracks[1] if detec.frame == frame2]
                    #detecs_3d_1 = compute_3d_detecs(detecs_frame_1, homographies[cam])

                    for det_0 in detecs_frame_0:
                        for det_1 in detecs_frame_1:
                            dist[det_0.track_id].append({det_1.track_id: (intersection(det_0.histogram, det_1.histogram), get_homography_IoU(det_0, det_1, homographies[0], homographies[1]))}) #dict of tuples (hist intersection, correspondence IoU)

    total_distances = []
    for key, dist_list in dist.items():
        for item in dist_list:
            d = tuple(item.values())[0]
            total_distances.append(float(d[0])) #histograms


    print(total_distances)
    max_dist = max(total_distances)
    min_dist = min(total_distances)
    average = float(sum(total_distances)) / len(total_distances)
    print('max dist: {}'.format(max_dist))
    print('min dist: {}'.format(min_dist))
    print('average dist: {}'.format(average))

    threshold = average

    candidates = defaultdict(list)
    for track_0, dist_list in dist.items():
        for track_1_key in dist_list:
            k = list(track_1_key.keys())
            track_1 = k[0]
            d = tuple(track_1_key.values())[0]
            if float(d[0]) < threshold:
                candidates[track_0].append(track_1)

    match_tracks = {}
    for track_0, dist_list in dist.items():
        possible_tracks = []
        max_IoU = 0
        cnt = Counter()
        for track_1_key in dist_list:
            k = list(track_1_key.keys())
            track_1 = k[0]
            if track_1 in candidates[track_0]:
                possible_tracks.append(track_1)
                d = tuple(track_1_key.values())[0]
                if float(d[1]) > max_IoU:
                    max_IoU = float(d[1])
                    best_track = track_1
        # if max_IoU == 1:
        #     cnt[best_track] += 1
        for track_pos in possible_tracks:
            if max_IoU > 0.30:
                if track_pos == best_track:
                    cnt[track_pos] += 1
            # else:
            #     cnt[track_pos] += 1
        #else:
        #    for track_pos in possible_tracks:
        #        cnt[track_pos] += 1
        match_tracks[track_0] =cnt

    print(match_tracks)

    matched_tracks = optimize_matching(match_tracks)

    visualize_matches(matched_tracks, cameras_tracks[0], cameras_tracks[1], video_path_0, video_path_1)


def lon_lat_to_cartesian(lon, lat, R = 6378137):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z

#
# if __name__ == '__main__':
#     x1, y1, z1 = lon_lat_to_cartesian(42.525821, -90.723853)
#     x2, y2, z2 = lon_lat_to_cartesian(42.525473, -90.723319)
#
#     print('camera1: {0}, {1}, {2}'.format(x1, y1, z1))
#     print('camera2: {0}, {1}, {2}'.format(x2, y2, z2))
#

if __name__ == '__main__':
    frame_path_1 = '../video_frames/c001/frame_0357.png'
    groundtruth_challenge_path = "gt/gt.txt"
    video_challenge_path = "vdo.avi"

    frame_path_2 = '../../week4/video_frames/c002/frame_0338.png'
    homography_path = "calibration.txt"
    camera1 = "../../datasets/AICity_data/train/S01/c001/"
    camera2 = "../../datasets/AICity_data/train/S01/c002/"
    homography_path_start = "../../datasets/calibration/"

    groundtruth_list, _ = read_annotations_file(camera1 + groundtruth_challenge_path,
                                                         camera1 + video_challenge_path)

    detecs_car = [detec for detec in groundtruth_list if detec.track_id == 54]
    print(detecs_car)


    homography1 = read_homography_matrix(homography_path_start + camera1[(len(camera1)-5):] + homography_path)
    homography2 = read_homography_matrix(homography_path_start + camera2[(len(camera2)-5):] + homography_path)

    #image1 = cv2.imread(frame_path_1)
    #fig, ax = plt.subplots()
    #ax.imshow(image1)
    image2 = cv2.imread(frame_path_2)
    h, w = image2.shape[:2]
    print('h: {}'.format(h))
    print("w: {}".format(w))
    fig, ax = plt.subplots()
    ax.imshow(image2)
    for detec in detecs_car:
        bbox_1 = detec.bbox
        print(bbox_1)

        #minc, minr, maxc, maxr = bbox_1
        #rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='green', linewidth=4)
        #ax.add_patch(rect)
        #plt.show()

        H1 = np.array(homography1)
        H2 = np.array(homography2)
        minc, minr, maxc, maxr = bbox_1
        #h, w = image1.shape[:2]
        x = minc
        y = maxr

        det_1_hom = apply_homography_to_point(x, y, H1, H2)

        x_br = maxc
        y_br = maxr
        det_1_hom_br = apply_homography_to_point(x_br, y_br, H1, H2)

        print('Left: {}'.format(det_1_hom))
        print('Right: {}'.format(det_1_hom_br))
        print('')


        original_ratio = (maxr - minr)/(maxc - minc) #height/width of bbox
        width_transformed = abs(det_1_hom[0] - det_1_hom_br[0])
        height_transformed = original_ratio*width_transformed

        predicted_bbox_1 = [min(det_1_hom[0],det_1_hom_br[0]), min(det_1_hom[1],det_1_hom_br[1])-height_transformed, max(det_1_hom[0],det_1_hom_br[0]), max(det_1_hom[1],det_1_hom_br[1])]



        #gt_bbox = [1163,420,(1163+524),(420+245)]
        predicted_bbox_1 = [predicted_bbox_1[0], predicted_bbox_1[1], predicted_bbox_1[2], predicted_bbox_1[3]]
        print('Predicted bbox: {}'.format(predicted_bbox_1))
        #print('Ground truth bbox: {}'.format(gt_bbox))
        #minc, minr, maxc, maxr = gt_bbox
        #rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='green',
        #                          linewidth=4)
        #ax.add_patch(rect)
        minc, minr, maxc, maxr = predicted_bbox_1
        rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='red',
                                  linewidth=4)
        ax.add_patch(rect)

    groundtruth_list2, _ = read_annotations_file(camera2 + groundtruth_challenge_path,
                                                camera2 + video_challenge_path)
    detecs_car2 = [detec for detec in groundtruth_list2 if detec.track_id == 56]

    for detec in detecs_car2:
        bbox_1 = detec.bbox

        minc, minr, maxc, maxr = bbox_1

        rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='green',
                                  linewidth=4)
        ax.add_patch(rect)
    plt.show()
