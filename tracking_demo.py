import torch

from src.models.tracker import CoTracker
from src.utils.io import read_video_from_path, Visualizer


save_dir = "/home/tracking/Temporal-Information-Embedded-Guidewire-Tracking/results/Co-tracker/DHM_single_gw/"
vid_name = "2_data_slice37.mp4"
video_path = f"{save_dir}/{vid_name}"

vid = read_video_from_path(video_path)
video = torch.from_numpy(vid).permute(0, 3, 1, 2)[None].float()
model = CoTracker()

# Need to get pixel coordinates from segmentation mask
queries = torch.tensor([
    [0., 430., 410.], # frame number, x_coord, y_coord
    # [0., 435., 401.], 
    # [0., 443., 396.],
    # [0., 456., 397.],
    # [0., 461., 401.],
    # [0., 467., 407.],
    # [0., 471., 410.],
    # [0., 474., 412.],
    [0., 477., 410.],
])

if torch.cuda.is_available():
    video = video.cuda()
    queries = queries.cuda()

pred_tracks, pred_visibility = model(
        video, 
        queries = queries[None]
        # grid_size=grid_size, 
        # grid_query_frame=grid_query_frame, 
        # backward_tracking=backward_tracking
        )


vis = Visualizer(
    save_dir=save_dir,
    linewidth=1,
    mode='Blues',
    tracks_leave_trace=3
)
vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename=vid_name.split('.')[0])