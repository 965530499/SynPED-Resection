
from src.models.modeling import SynPED

def main():

    clip_path = f".../best_checkpoint.pt"
    
    dino_config_file = "src/config/GroundingDINO_SwinT_OGC.py"
    dino_path = f".../checkpoint_best_avg.pth"

    sam_path = f".../medsam_model_11.pth"
    

    model = SynPED()
    model.load_classifier(clip_path)
    model.load_detector(dino_config_file, dino_path)
    model.load_segmenter(sam_path)


    img_path = ".../xxx.jpg"
    cls_label = model.classify_one_img(img_path)
    # cls_label: 0 or 1, 0 - 无癌变；1 - 癌前病变
    bboxes, masks, _ = model.segment_one_img(img_path, classes=["lesion"])  
    # bboxes: 带有矩形框的图片
    # masks: 带有mask的图片

    return cls_label, bboxes, masks


if __name__ == "__main__":
    main()
