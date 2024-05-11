import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from os.path import join
from L1optimal_saliency_lpp1 import stabilize
from face_tracking import get_landmarks_array, visualize_salient_points

# Constants
FOURCC_AVI = cv.VideoWriter_fourcc(*'XVID')
FOURCC_MP4 = cv.VideoWriter_fourcc(*'mp4v')
EYE_INDICES = [33, 133, 160, 144, 159, 145, 386, 385, 384, 398, 263, 362]

def calculate_crop_corners(image_shape, crop_ratio):
    center_x = round(image_shape[1] / 2)
    center_y = round(image_shape[0] / 2)
    crop_width = round(image_shape[1] * crop_ratio)
    crop_height = round(image_shape[0] * crop_ratio)
    top_left_x = round(center_x - crop_width / 2)
    top_left_y = round(center_y - crop_height / 2)
    return top_left_x, top_left_x + crop_width, top_left_y, top_left_y + crop_height

def plot_camera_trajectory(original_trajectory, stabilized_trajectory, output_path):
    axes = ['x', 'y']
    for idx, coord in enumerate(axes):
        plt.figure()
        plt.plot(original_trajectory[:, idx], label='Original')
        plt.plot(stabilized_trajectory[:, idx], label='Stabilized')
        plt.xlabel('Frame Number')
        plt.ylabel(f'2D Camera {coord} coord. (pixels)')
        plt.title(f'Original vs Stabilized {coord.upper()}')
        plt.legend()
        plt.savefig(f"{output_path}_traj_{coord}.png")
        plt.close()

def track_inter_frame_transformations(video_capture, transforms_matrix, prev_frame_gray, input_file):
    frame_count = transforms_matrix.shape[0]
    landmarks = get_landmarks_array(input_file)
    for i in range(frame_count):
        features = cv.goodFeaturesToTrack(prev_frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        if features is not None:
            for corner in np.int0(features):
                x, y = corner.ravel()
                cv.circle(prev_frame_gray, (x, y), 10, (255, 0, 0), -1)
            cv.imshow('frame', prev_frame_gray)
            if cv.waitKey(25) == 27:
                break

        current_points = np.array(landmarks[i], dtype=np.float32).reshape(-1, 1, 2)
        success, current_frame = video_capture.read()
        if not success:
            break

        current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
        next_points, status, _ = cv.calcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, current_points, None)
        transforms_matrix[i + 1, :, :2] = cv.estimateAffine2D(next_points[status == 1], current_points[status == 1])[0].T
        prev_frame_gray = current_frame_gray

def apply_stabilization(video_capture, video_writer, stabilization_matrices, frame_dimensions, crop_factor):
    video_capture.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_count = stabilization_matrices.shape[0]
    for i in range(frame_count):
        success, frame = video_capture.read()
        if not success:
            break
        scale_x, scale_y = 1 / crop_factor, 1 / crop_factor
        scale_matrix = np.diag([scale_x, scale_y, 1])
        shift_to_center_matrix = np.eye(3)
        shift_to_center_matrix[0, 2] = -frame_dimensions[0] / 2
        shift_to_center_matrix[1, 2] = -frame_dimensions[1] / 2

        shift_back_matrix = np.eye(3)
        shift_back_matrix[0, 2] = frame_dimensions[0] / 2
        shift_back_matrix[1, 2] = frame_dimensions[1] / 2

        transform_matrix = np.eye(3)
        transform_matrix[:2, :] = stabilization_matrices[i, :, :2].T
        final_transform = shift_back_matrix @ scale_matrix @ shift_to_center_matrix @ np.linalg.inv(transform_matrix)
        stabilized_frame = cv.warpAffine(frame, final_transform[:2, :], frame_dimensions)
        video_writer.write(stabilized_frame)
    video_writer.release()

def process_video(input_path, output_path, codec):
    video_capture = cv.VideoCapture(input_path)
    total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv.CAP_PROP_FPS))
    transforms = np.zeros((total_frames, 3, 3), np.float32)
    transforms[:, :, :] = np.eye(3)
    _, initial_frame = video_capture.read()
    initial_frame_gray = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)
    track_inter_frame_transformations(video_capture, transforms, initial_frame_gray, input_path)
    stabilization_params = stabilize(transforms[:-1], initial_frame.shape, True, None, 0.7, input_path)  # exclude the last frame if not needed
    #trajectory = np.cumprod([np.eye(3), *transforms[:-1]], axis=0)  # exclude the last identity matrix if not needed
    #stabilized_trajectory = trajectory @ stabilization_params
    #plot_camera_trajectory(trajectory, stabilized_trajectory, join(os.path.dirname(output_path), 'plots', os.path.basename(input_path).split('.')[0]))
    video_writer = cv.VideoWriter(output_path, codec, fps, (frame_width, frame_height))
    apply_stabilization(video_capture, video_writer, stabilization_params, (frame_width, frame_height), 0.7)
    video_capture.release()

if __name__ == "__main__":
    process_video("video_short3.mp4", "video_short3_stab12.mp4", FOURCC_MP4)

