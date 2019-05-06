import cv2
import os
from tqdm import tqdm

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

########################## Precision-Recall

def plot_precision_recall_curve(precision, recall, max_precision_per_step, title="plot", title2=""):

    thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    # Data for plotting
    fig, ax = plt.subplots()
    #print(len(recall), len(precision))
    ax.plot(recall, precision)
    ax.plot(thresholds, max_precision_per_step, 'ro')   # Red points

    ax.set(xlabel='Recall', ylabel='Precision',
           title='Precision-Recall Curve')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.grid()

    fig.savefig("precision-recall-" + title + title2 + ".png")
    # plt.show()

def plot_multiple_precision_recall_curves(groundtruth_list, detections_list, thresholds, detector):
    fig, ax = plt.subplots()
    for th in thresholds:
        precision, recall = compute_mAP(groundtruth_list, detections_list, IoU_threshold=th)
        ax.plot(recall, precision, label="threshold: %.2f" % th)
    ax.set(xlabel='Recall', ylabel='Precision',
           title='Precision-Recall Curve')
    ax.grid()
    ax.legend(loc='best')

    fig.savefig("precision-recall-thresholds" + detector + ".png")

########################## Bounding boxes

def draw_video_bboxes(video_path, groundtruth, detections, display=False, export_frames=False):
    """
    Send each frame Draw the groundtruth (green) and the detections (red) bboxes on a video

    """
    capture = cv2.VideoCapture(video_path)
    n_frame = 0

    pbar = tqdm(total=2140)

    while capture.isOpened():
        valid, image = capture.read()
        if not valid:
            break

        #if n_frame >= 600 and n_frame <= 700:
        # Get groundtruth and detections of the target frame
        gt_on_frame = [x for x in groundtruth if x.frame == n_frame]
        gt_bboxes = [o.bbox for o in gt_on_frame]
        detections_on_frame = [x for x in detections if x.frame == n_frame]
        detections_bboxes = [o.bbox for o in detections_on_frame]

        # Plot image
        frame = draw_image_bboxes_opencv(image, gt_bboxes, detections_bboxes)
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        if display:
            cv2.imshow(str(n_frame), frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if export_frames:
            cv2.imwrite('output_frames/mask-rcnn/frame_{:04d}.png'.format(n_frame), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 4, int(cv2.IMWRITE_JPEG_QUALITY), 70])

        pbar.update(1)
        n_frame += 1

    pbar.close()

def draw_image_bboxes(pixel_candidates, gt_candidate, detection_candidate):
    """
    Draw the groundtruth (green) and the detections (red) bboxes on one image
    THE OPENCV VERSION IS MORE EFFICIENT

    """
    fig, ax = plt.subplots()
    ax.imshow(pixel_candidates, cmap='gray')

    for candidate in detection_candidate:
        minc, minr, maxc, maxr = candidate
        rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    for candidate in gt_candidate:
        minc, minr, maxc, maxr = candidate
        rect = mpatches.Rectangle((minc, minr), maxc-minc+1, maxr-minr+1, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)

    #plt.show()

def draw_image_bboxes_opencv(image, gt_candidate, detection_candidate):
    """
    Draw the groundtruth (green) and the detections (red) bboxes on an image and return the image

    """
    for candidate in detection_candidate:
        minc, minr, maxc, maxr = candidate
        cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 0, 255), 8)    # Red

    for candidate in gt_candidate:
        minc, minr, maxc, maxr = candidate
        cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 255, 0), 5)    # Green

    return image

########################## Tracks

def visualize_tracks(image, frame_tracks, colors, display=False, export_frames=False, export_path="test.png"):
    """
    Draw on an image the bounding boxes on frame_tracks, with a different color for each track
    THE OPENCV VERSION IS MORE EFFICIENT

    """
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.xticks([]), plt.yticks([])

    for id in frame_tracks.keys():
        bbox = frame_tracks[id]
        minc, minr, maxc, maxr = bbox['bbox']
        rect = mpatches.Rectangle((minc, minr), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor=colors[id], linewidth=2)
        ax.add_patch(rect)

    if display:
        plt.show()

    if export_frames:
        plt.savefig(export_path, bbox_inches='tight', pad_inches=0, dpi=72)


def visualize_tracks_opencv(image, frame_tracks, colors, display=False, export_frames=False, export_path="test.png"):
    """
    Draw on an image the bounding boxes on frame_tracks, with a different color for each track

    """
    for id in frame_tracks.keys():
        bbox = frame_tracks[id]
        minc, minr, maxc, maxr = bbox['bbox']
        cv2.rectangle(image, (minc, minr), (maxc, maxr), colors[id]*255, 20)

    # for id in frame_tracks.keys():
    #     bbox = frame_tracks[id]
    #     minc, minr, maxc, maxr = bbox['bbox']
    #     cv2.rectangle(image, (minc, minr), (maxc, maxr), colors[id]*255, 20)


    if display:
        cv2.imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if export_frames:
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        cv2.imwrite(export_path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 4, int(cv2.IMWRITE_JPEG_QUALITY), 70])

########################## Week 2

def plot3D(X,Y,Z):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    axis = ["Ro", "Alpha", "mAP"]
    ax.set_xlabel(axis[0])
    ax.set_ylabel(axis[1])
    ax.set_zlabel(axis[2])
    plt.savefig('grid_search.png', dpi=300)
    plt.show()

def plot2D(X,Y, xlab, ylab, tit):

    plt.plot(X, Y, linewidth=2.0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([, 160, 0, 0.03])
    plt.grid(True)
    plt.show()