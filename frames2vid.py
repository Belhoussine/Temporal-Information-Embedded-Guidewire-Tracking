import os
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from src.utils.dataset import mask_to_bbox, stack_frames
from src.models.robustPCA import RPCA

nii_path = '/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/rawimg.nii.gz'
result_folder_path = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/results"

# Load image (3D) [X,Y,time]
nii_img = nib.load(nii_path)
frames = np.transpose(nii_img.get_fdata(), (2, 0, 1))

number_of_frames = frames.shape[0]
resolution = frames.shape[1:]

########
# Directory containing the image sequence
image_dir = '/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/datasets/baochang_nifti'

# Output video file name and properties
output_video = 'baochang_video.mp4'
frame_rate = 10  # You can adjust this frame rate as needed
frame_size = resolution  # Change to the resolution you desire

# Get the list of image files in the directory
image_files = [os.path.join(image_dir, f'baochang_frame{i}.png') for i in range(150)]

# # Sort the image files to ensure they are in the correct order
# image_files.sort()

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, frame_rate, frame_size)

for image_file in image_files:
    frame = cv2.imread(image_file)
    
    if frame is not None:
        # Resize the frame to match the desired frame size
        frame = cv2.resize(frame, frame_size)
        
        # Write the frame to the video
        out.write(frame)

# Release the VideoWriter and close the video file
out.release()

print(f"Video saved as {output_video}")


#########


# thresholds = [170, 190]

# frame_steps = [5, 10]
# iterations = 2000

# i = 0

# for frame_step in frame_steps:
#     for frame_id in range(0, number_of_frames, frame_step):
#         # print(range(frame_id, frame_id + frame_step))
#         current_stack = frames[frame_id:frame_id + frame_step]
#         D = stack_frames(current_stack, resolution=resolution,
#                          n_frames=frame_step, frames_loaded=True)
#         rpca = RPCA(D)
#         rpca.fit(max_iter=iterations, iter_print=100)
#         # rpca.plot_results(resolution=resolution, thresholds=thresholds)
#         rpca.save_results("baochang_test", frame_step, iterations,
#                           thresholds, result_folder_path, resolution=resolution, frame_ids=range(frame_id, frame_id + frame_step))