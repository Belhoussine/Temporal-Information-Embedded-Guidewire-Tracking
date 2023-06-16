from models.robustPCA import RPCA
from utils.dataset import mask_to_bbox, stack_frames


if __name__ == "__main__":
    frame_paths = [
        f"/home/belhoussine/dev/TUM/CS - Practical/Temporal-Information-Embedded-Guidewire-Tracking/datasets/Guide_wire_sythetic_data/sequence01_frame00000_sequence05_frame00050/png/navi0000000{i}.png"
        for i in range(10)
    ]

    D = stack_frames(frame_paths, resolution=(
        512, 512), n_frames=len(frame_paths))

    # Use RPCA to estimate the data as L + S, where L is low rank, and S is sparse
    rpca = RPCA(D)
    L, S = rpca.fit(max_iter=1000, iter_print=100)
    rpca.plot_results()