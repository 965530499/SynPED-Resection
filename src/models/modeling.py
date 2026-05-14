from collections import OrderedDict
import cv2
import torch
import copy
import os

import supervision as sv
import torchvision
import numpy as np

import clip.clip as clip
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor

from src.models import utils
from src.utils.dino_inference_utils import *


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)

class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        # print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)



class SynPED:
    """Inference model for SynPED: CLIP + GroundingDINO + SAM"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = None
        self.detector = None
        self.segmenter = None


    # %% load models

    def load_classifier(self, ckpt_path):
        self.classifier = ImageClassifier.load(ckpt_path)
        self.classifier.to(self.device)

    def load_detector(self, config_file, checkpoint_path, cpu_only=False):
        def clean_state_dict(state_dict):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7] == "module.":
                    k = k[7:]  # remove `module.`
                new_state_dict[k] = v
            return new_state_dict

        args = SLConfig.fromfile(config_file)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        self.detector = model
    
    def load_segmenter(self, checkpoint_path):
        sam = sam_model_registry['medsam_vit_b'](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.segmenter = SamPredictor(sam)

    # %% Ablation study model loading methods
    
    def load_swin_classifier(self, ckpt_path):
        """Load Swin-B classifier for ablation study"""
        import torchvision
        from torchvision import transforms
        
        # Load Swin-B model
        self.swin_classifier = torchvision.models.swin_b(weights=None)
        # Modify the final layer for binary classification
        self.swin_classifier.head = torch.nn.Linear(self.swin_classifier.head.in_features, 2)
        
        # Load trained weights
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.swin_classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.swin_classifier.load_state_dict(checkpoint)
        
        self.swin_classifier.to(self.device)
        self.swin_classifier.eval()
        
        # Define preprocessing for Swin-B
        self.swin_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_faster_rcnn(self, ckpt_path):
        """Load Faster R-CNN detector for ablation study"""
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        # Load Faster R-CNN model (same as your baseline script)
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        
        # Modify for binary classification (background + lesion)
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        
        # Load trained weights first, then modify classifier (following your baseline script pattern)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        try:
            # Try loading state dict first (untrained model checkpoint)
            if 'model_state_dict' in checkpoint:
                self.faster_rcnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.faster_rcnn.load_state_dict(checkpoint)
            self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        except:
            # If loading fails, modify classifier first then load (trained model checkpoint)
            self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
            if 'model_state_dict' in checkpoint:
                self.faster_rcnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.faster_rcnn.load_state_dict(checkpoint)
        
        self.faster_rcnn.to(self.device)
        self.faster_rcnn.eval()
    
    def load_deeplabv3(self, ckpt_path):
        """Load DeepLabv3 segmenter for ablation study"""
        import torchvision
        from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet101
        
        # Load DeepLabv3 model (following your baseline script)
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        self.deeplabv3_model = deeplabv3_resnet101(weights=None, weights_backbone=None, num_classes=1)
        
        # Load backbone weights manually like in your baseline script
        backbone_path = "/date/jc/models/Endo/baseline/resnet101-5d3b4d8f.pth"
        if os.path.exists(backbone_path):
            backbone = torch.load(backbone_path, map_location='cpu')
            self.deeplabv3_model.backbone.load_state_dict(backbone, strict=False)
        
        # Load trained model weights
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            # Remove 'model.' prefix from keys if present
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
            self.deeplabv3_model.load_state_dict(new_state_dict)
        else:
            # Remove 'model.' prefix from keys if present
            new_state_dict = {k[6:] if k.startswith('model.') else k: v for k, v in checkpoint.items()}
            self.deeplabv3_model.load_state_dict(new_state_dict)
        self.deeplabv3_model.to(self.device)
        self.deeplabv3_model.eval()

    
    # %% inference

    def classify_one_img(self, image) -> int:
        """
        predict the class of the image
        param:
            image_path: str, the path of the image
        return:
            class_id: int, 0 or 1, means benign or malignant
        """
        # image = Image.open(image_path).convert('RGB')
        image = image.to(self.device)
        logits = self.classifier(image)
        # outputs = logits_per_image.softmax(dim=1)
        outputs = logits    # bs, 2(num_classes)

        predicted = outputs.softmax(dim=1)  # bs, 2(num_classes)
        _, predicted = torch.max(outputs.data, 1)   # 获取预测结果, 0或1. (bs,)
        return predicted

    def classify_one_img_swin(self, image) -> int:
        """
        Classify image using Swin-B for ablation study
        param:
            image: tensor, preprocessed image tensor
        return:
            class_id: int, 0 or 1, means benign or malignant
        """
        image = image.to(self.device)
        with torch.no_grad():
            logits = self.swin_classifier(image)
        
        predicted = logits.softmax(dim=1)
        _, predicted = torch.max(logits.data, 1)
        return predicted


    def detect_one_img(self, image, caption, box_threshold, text_threshold=None, 
                       with_logits=True, cpu_only=False, token_spans=None, keep_max_if_no_box=False):
        """
        detect the bounding boxes of the image
        param:
            image: PIL.Image.Image
            caption: str, the caption of the image
            box_threshold: float, the threshold of the bounding box
            text_threshold: float, the threshold of the text
            with_logits: bool, whether to return the logits
            cpu_only: bool, whether to run on cpu
            token_spans: list, the spans of the tokens
            keep_max_if_no_box: bool, whether to keep the max if no box
        return:
            boxes: list, the bounding boxes of the image
            pred_phrases: list, the predicted phrases of the image
        """
        assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        self.detector.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.detector(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        if token_spans is None:
            logits_unfilt = logits.cpu().clone()
            boxes_unfilt = boxes.cpu().clone()

            filt_mask = logits_unfilt.max(dim=1)[0] > box_threshold
            logits_filt = logits_unfilt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_unfilt[filt_mask]  # num_filt, 4
            
            tmp_text_threshold = text_threshold
            if keep_max_if_no_box:
                if logits_filt.shape[0] == 0:
                    max_confidence_index = logits_unfilt.max(dim=1)[0].argmax()
                    logits_filt = logits_unfilt[max_confidence_index].unsqueeze(0)
                    boxes_filt = boxes_unfilt[max_confidence_index].unsqueeze(0)
                    tmp_text_threshold = 0.0

            # get phrase
            tokenlizer = self.detector.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > tmp_text_threshold, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def detect_one_img_faster_rcnn(self, image, box_threshold=0.25):
        """
        Detect bounding boxes using Faster R-CNN for ablation study
        param:
            image: tensor, preprocessed image tensor
            box_threshold: float, confidence threshold for boxes
        return:
            boxes: tensor, detected bounding boxes
            pred_phrases: list, predicted phrases (dummy for compatibility)
        """
        image = image.to(self.device)
        
        # Faster R-CNN expects 3D tensor (C, H, W), but we need to add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            predictions = self.faster_rcnn([image.squeeze(0)])  # Remove batch dim for input list
        
        prediction = predictions[0]  # Get first (and only) prediction
        
        # Filter by confidence threshold
        scores = prediction['scores'].cpu()
        boxes = prediction['boxes'].cpu()
        labels = prediction['labels'].cpu()
        
        # Keep only lesion class (label=1) and high confidence
        lesion_mask = (labels == 1) & (scores >= box_threshold)
        
        if lesion_mask.sum() == 0:
            # If no high-confidence lesions, take the highest scoring lesion
            lesion_indices = (labels == 1).nonzero(as_tuple=True)[0]
            if len(lesion_indices) > 0:
                best_lesion_idx = lesion_indices[scores[lesion_indices].argmax()]
                lesion_mask = torch.zeros_like(lesion_mask)
                lesion_mask[best_lesion_idx] = True
        
        filtered_boxes = boxes[lesion_mask]
        filtered_scores = scores[lesion_mask]
        
        # Convert from xyxy to normalized xywh format (to match GroundingDINO output)
        if len(filtered_boxes) > 0:
            # Get image dimensions
            height, width = image.shape[-2], image.shape[-1]
            
            # Convert xyxy to center_x, center_y, width, height and normalize
            x1, y1, x2, y2 = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            
            normalized_boxes = torch.stack([center_x, center_y, box_width, box_height], dim=1)
        else:
            normalized_boxes = torch.empty((0, 4))
        
        # Create dummy phrases for compatibility
        pred_phrases = [f"lesion({score:.3f})" for score in filtered_scores]
        
        return normalized_boxes, pred_phrases


    def segment_one_img(
        self,
        img_path: str,
        classes=["lesion"], box_threshold=0.25, text_threshold=0.25, NMS_THRESHOLD=0.8
    ):
        """
        predict both detection boxes and segmentations masks of the image
        param:
            img_path: str, the path of the image
            classes: list, the classes of the image
            box_threshold: float, the threshold of the bounding box
            text_threshold: float, the threshold of the text
            NMS_THRESHOLD: float, the threshold of the NMS
        return:
            annotated_frame: PIL.Image.Image, the detections of the image
            annotated_image: PIL.Image.Image, the segmentations of the image
            detections: sv.Detections, the detections of the image
        """
        # load image
        image = cv2.imread(img_path)

        # get detections
        caption = ". ".join(classes)
        processed_image = preprocess_image(image_bgr=image).to(self.device)
        caption = preprocess_caption(caption=caption)
        self.detector = self.detector.to(self.device)
        processed_image = processed_image.to(self.device)

        with torch.no_grad():
            outputs = self.detector(processed_image[None], captions=[caption])

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        # 如果没有检测到任何物体，选择置信度最高的那个
        tmp_text_threshold = text_threshold
        if logits.shape[0] == 0:
            max_confidence_index = prediction_logits.max(dim=1)[0].argmax()
            logits = prediction_logits[max_confidence_index].unsqueeze(0)
            boxes = prediction_boxes[max_confidence_index].unsqueeze(0)
            tmp_text_threshold = 0.0

        tokenizer = self.detector.tokenizer
        tokenized = tokenizer(caption)

        phrases = [
            get_phrases_from_posmap(logit > tmp_text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]

        # return boxes, logits.max(dim=1)[0], phrases
        boxes, logits, phrases = boxes, logits.max(dim=1)[0], phrases


        # NMS post process
        source_h, source_w, _ = image.shape
        detections = post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id


        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        # save the annotated grounding dino image
        # cv2.imwrite(gdino_out_path, annotated_frame)


        # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)


        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.segmenter,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # save the annotated grounded-sam image
        # cv2.imwrite(gsam_out_path, annotated_image)
        return annotated_frame, annotated_image, detections

    def segment_one_img_with_boxes(self, img_path: str, boxes_normalized: np.ndarray):
        """
        Segment image using SAM with pre-computed boxes (for ablation study)
        param:
            img_path: str, path to the image
            boxes_normalized: np.ndarray, normalized boxes in xywh format
        return:
            annotated_frame: np.ndarray, the detections of the image
            annotated_image: np.ndarray, the segmentations of the image  
            detections: sv.Detections, the detections of the image
        """
        # Load image
        image = cv2.imread(img_path)
        source_h, source_w, _ = image.shape
        
        if len(boxes_normalized) == 0:
            # Return empty detections if no boxes
            detections = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int),
                mask=np.empty((0, source_h, source_w), dtype=bool)
            )
            return image.copy(), image.copy(), detections
        
        # Convert normalized boxes to pixel coordinates and xyxy format
        boxes_xyxy = []
        for box in boxes_normalized:
            if len(box) >= 4:
                cx, cy, w, h = box[:4]
                x1 = (cx - w/2) * source_w
                y1 = (cy - h/2) * source_h
                x2 = (cx + w/2) * source_w
                y2 = (cy + h/2) * source_h
                boxes_xyxy.append([x1, y1, x2, y2])
        
        if len(boxes_xyxy) == 0:
            # Return empty detections if no valid boxes
            detections = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int),
                mask=np.empty((0, source_h, source_w), dtype=bool)
            )
            return image.copy(), image.copy(), detections
        
        boxes_xyxy = np.array(boxes_xyxy)
        
        # Use SAM to segment with the provided boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)
        
        # Create detections object
        detections = sv.Detections(
            xyxy=boxes_xyxy,
            confidence=np.ones(len(boxes_xyxy)) * 0.8,  # Dummy confidence
            class_id=np.zeros(len(boxes_xyxy), dtype=int)  # All lesion class
        )
        
        # Get masks from SAM
        detections.mask = segment(
            sam_predictor=self.segmenter,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        
        # Annotate images
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [f"lesion {conf:.2f}" for conf in detections.confidence]
        
        # Annotated frame (boxes only)
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        
        # Annotated image (masks + boxes)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        return annotated_frame, annotated_image, detections

    def segment_one_img_deeplabv3(self, img_path: str, pred_boxes):
        """
        Segment image using DeepLabv3 within detection boxes for ablation study
        param:
            img_path: str, path to the image
            pred_boxes: list or tensor, predicted bounding boxes in normalized format
        return:
            annotated_frame: np.ndarray, the detections of the image
            annotated_image: np.ndarray, the segmentations of the image  
            detections: sv.Detections, the detections of the image
        """
        import torchvision.transforms.functional as TF
        
        # Load image
        image = cv2.imread(img_path)
        source_h, source_w, _ = image.shape
        
        # Convert boxes to supervision format
        if len(pred_boxes) == 0:
            # Return empty detections if no boxes
            detections = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int),
                mask=np.empty((0, source_h, source_w), dtype=bool)
            )
            return image.copy(), image.copy(), detections
        
        # Convert normalized boxes to pixel coordinates
        if isinstance(pred_boxes, list):
            pred_boxes = np.array(pred_boxes)
        
        # Convert from normalized xywh to xyxy pixel coordinates
        boxes_xyxy = []
        for box in pred_boxes:
            if len(box) >= 4:
                cx, cy, w, h = box[:4]
                x1 = (cx - w/2) * source_w
                y1 = (cy - h/2) * source_h
                x2 = (cx + w/2) * source_w
                y2 = (cy + h/2) * source_h
                boxes_xyxy.append([x1, y1, x2, y2])
        
        if len(boxes_xyxy) == 0:
            # Return empty detections if no valid boxes
            detections = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int),
                mask=np.empty((0, source_h, source_w), dtype=bool)
            )
            return image.copy(), image.copy(), detections
        
        boxes_xyxy = np.array(boxes_xyxy)
        
        # Process image for DeepLabv3 (following your baseline script - no normalization!)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = TF.to_pil_image(image_rgb)
        # Resize to 1024x1024 like in your baseline script
        image_pil = TF.resize(image_pil, (1024, 1024))
        image_tensor = TF.to_tensor(image_pil)
        # NOTE: Your baseline script doesn't use ImageNet normalization
        # image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Get segmentation from DeepLabv3 (binary classification with sigmoid)
        with torch.no_grad():
            output = self.deeplabv3_model(image_tensor)['out']
            # Apply sigmoid and threshold (following your baseline script)
            output = torch.sigmoid(output)
            segmentation = (output > 0.25).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # Debug: print segmentation statistics
            print(f"Debug DeepLabv3: segmentation shape={segmentation.shape}, sum={segmentation.sum()}, max={segmentation.max()}")
        
        # Resize segmentation to original image size (from 1024x1024 back to source size)
        if segmentation.shape != (source_h, source_w):
            segmentation = cv2.resize(segmentation, (source_w, source_h), interpolation=cv2.INTER_NEAREST)
        
        # Create masks by intersecting segmentation with detection boxes
        masks = []
        confidences = []
        class_ids = []
        final_boxes = []
        
        for box in boxes_xyxy:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(source_w, x2), min(source_h, y2)
            
            # Create mask within the bounding box
            mask = np.zeros((source_h, source_w), dtype=bool)
            box_segmentation = segmentation[y1:y2, x1:x2]
            
            # Since it's binary segmentation (1 = lesion, 0 = background)
            lesion_mask = (box_segmentation == 1)
            
            if lesion_mask.sum() > 0:  # If there's any lesion segmentation in this box
                mask[y1:y2, x1:x2] = lesion_mask
                masks.append(mask)
                confidences.append(0.8)  # Dummy confidence
                class_ids.append(0)  # Lesion class
                final_boxes.append(box)
                print(f"Debug: Found lesion in box {box}, mask sum={lesion_mask.sum()}")
            else:
                print(f"Debug: No lesion found in box {box}, box_seg sum={box_segmentation.sum()}")
        
        if len(masks) == 0:
            # Return empty detections if no valid masks
            detections = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty(0),
                class_id=np.empty(0, dtype=int),
                mask=np.empty((0, source_h, source_w), dtype=bool)
            )
        else:
            # Create detections object
            detections = sv.Detections(
                xyxy=np.array(final_boxes),
                confidence=np.array(confidences),
                class_id=np.array(class_ids),
                mask=np.array(masks)
            )
        
        # Annotate images
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        # Create labels
        labels = [f"lesion {conf:.2f}" for conf in detections.confidence]
        
        # Annotated frame (boxes only)
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        
        # Annotated image (masks + boxes)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        return annotated_frame, annotated_image, detections
