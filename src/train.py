import os
import cv2
import numpy as np
import torch
import random
import wandb
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo


# Funktion zum Laden des Datensatzes
def load_combined_dataset(json_file, image_dir):
    return load_coco_json(json_file, image_dir)


# Registrierung des Datensatzes
def register_combined_dataset():
    train_json = "/Users/nicoclerici/Documents/Bewerbung/DBew/Prototyp_eBew/data/annotations/instances_Train_with_segments_fixed.json"
    train_images = "/Users/nicoclerici/Documents/Bewerbung/DBew/Prototyp_eBew/data/train"

    DatasetCatalog.register(
        "train_dataset_combined",
        lambda: load_combined_dataset(train_json, train_images)
    )
    MetadataCatalog.get("train_dataset_combined").set(
        thing_classes=[
            "Strassenname", "Gebaeude Neu", "Gebaeude Bestehend", "GeometerStempel", "Nordpfeil",
            "Massstab", "Parzellengrenze", "Masslinie", "Unterschriften", "Titelinformation",
            "Gebaeude Untergrund", "Legende", "M.ü.M", "Gebaeude Abbruch", "Parzellennummer",
            "Gebaeudenummer"
        ]
    )


# Mapper Funktion für Augmentation
def combined_mapper(dataset_dict):
    dataset_dict = dataset_dict.copy()
    image = cv2.imread(dataset_dict["file_name"])  # RGB-Bild laden

    aug = T.AugmentationList([
        T.RandomFlip(horizontal=True, vertical=False),  # Zufälliges horizontales Flippen
        T.RandomBrightness(0.8, 1.2),  # Helligkeit anpassen
        T.RandomContrast(0.8, 1.2)  # Kontrast anpassen
    ])
    aug_input = T.AugInput(image)
    aug(aug_input)
    image = aug_input.image

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    return dataset_dict


# Eigener Trainer mit WandB-Integration
class WandbTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        wandb.init(project="Construction_Plan_Training", config=cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)


# Hauptfunktion zum Starten des Trainings
def main():
    register_combined_dataset()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train_dataset_combined",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1280
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.5, 0.5]
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = [1500, 3000]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.OUTPUT_DIR = "/Users/nicoclerici/Documents/Bewerbung/DBew/Prototyp_eBew/src/outputs_optimized"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = WandbTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
