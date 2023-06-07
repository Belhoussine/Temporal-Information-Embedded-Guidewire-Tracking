from utils.dataset import mask_to_bbox

if __name__ == "__main__":
    img_path = "/home/belhoussine/dev/TUM/CS - Practical/Temporal-Information-Embedded-Guidewire-Tracking/datasets/Guide_wire_sythetic_data/sequence01_frame00000_sequence05_frame00050/gt/navi00000000.png"
    mask_to_bbox(image_path=img_path)
