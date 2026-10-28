import json
import os
from typing import List, Dict, Tuple, Any


class COCODatasetAdapter:
    """
    Adapter per caricare dataset in formato COCO (es. HomeObjects-3K)
    e gestire il ciclo di etichettatura assistita per il progetto BOSS (RF1, RF3).
    """

    def __init__(self, annotation_path: str, config_vocab_path: str):
        self.annotation_path = annotation_path
        self.config_vocab_path = config_vocab_path
        self.vocabulary = self._load_vocabulary()
        self.target_class_names = list(self.vocabulary.keys())
        self.coco_data = self._load_coco_json()

    def _load_json(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File non trovato: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_vocabulary(self) -> Dict[str, List[str]]:
        """Carica il vocabolario e i prompt testuali da configurazione (RF3)."""
        data = self._load_json(self.config_vocab_path)
        return data.get("target_classes", {})

    def _load_coco_json(self) -> Dict[str, Any]:
        return self._load_json(self.annotation_path)

    def get_unlabeled_pool_and_ground_truth(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Separa il dataset in:
        1. Immagini totali (unlabeled pool per i modelli a vocabolario aperto).
        2. Annotazioni visibili (classi tradizionali già note).
        3. Annotazioni nascoste delle nuove classi (per simulare l'active learning - RF1).
        """
        images = self.coco_data.get("images", [])
        annotations = self.coco_data.get("annotations", [])
        categories = self.coco_data.get("categories", [])

        # Mappa degli ID COCO relativi alle nostre classi target
        target_cat_ids = {
            cat["id"]: cat["name"]
            for cat in categories
            if cat["name"] in self.target_class_names
        }

        visible_annotations = []
        hidden_ground_truth = []

        for ann in annotations:
            if ann["category_id"] in target_cat_ids:
                # Etichetta della classe nuova da nascondere al modello
                hidden_ground_truth.append(ann)
            else:
                # Etichetta tradizionale già nota
                visible_annotations.append(ann)

        return images, visible_annotations, hidden_ground_truth

    def export_to_boss_perception_format(self, image_info: Dict[str, Any], detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Converte i rilevamenti nel formato standard comune boss.perception/0.1 (specificato nella traccia).
        """
        boss_frame = {
            "schema": "boss.perception/0.1",
            "source_id": str(image_info.get("id", "source_0")),
            "frame_idx": image_info.get("id", 0),
            "ts_ms": 0.0,
            "models": {"det": "open-vocab-model@1.0"},
            "detections": []
        }

        for det in detections:
            boss_frame["detections"].append({
                "bbox_xyxy": det["bbox_xyxy"],  # Coordinate [x1, y1, x2, y2]
                "cls": det["cls"],
                "score": float(det["score"])
            })

        return boss_frame