import nibabel as nib
import numpy as np
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
thresholds = [170, 190]

frame_steps = [5, 10]
iterations = 2000

for frame_step in frame_steps:
    for frame_id in range(0, number_of_frames, frame_step):
        # print(range(frame_id, frame_id + frame_step))
        current_stack = frames[frame_id:frame_id + frame_step]
        D = stack_frames(current_stack, resolution=resolution,
                         n_frames=frame_step, frames_loaded=True)
        rpca = RPCA(D)
        rpca.fit(max_iter=iterations, iter_print=100)
        # rpca.plot_results(resolution=resolution, thresholds=thresholds)
        rpca.save_results("baochang_test", frame_step, iterations,
                          thresholds, result_folder_path, resolution=resolution, frame_ids=range(frame_id, frame_id + frame_step))
# fig, ax = plt.subplots(1, number_of_frames, constrained_layout=True)
# fig.canvas.set_window_title('Nifti Image Sequence')
# fig.suptitle(f'Nifti 1 slices {number_of_frames} time Frames', fontsize=16)
# #-------------------------------------------------------------------------------
# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()

# for frame_id in range(number_of_frames):
#     frame = frames[frame_id]
#     ax[frame_id].imshow(frames[frame_id],cmap='gray', interpolation=None)
#     ax[frame_id].set_title(f"layer {frame_id}")
#     ax[frame_id].axis('off')

# plt.show()
