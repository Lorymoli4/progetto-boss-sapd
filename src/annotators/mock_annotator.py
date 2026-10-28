import random
from typing import List, Dict, Any
from .base_annotator import BaseAnnotator


class MockAnnotator(BaseAnnotator):
    """
    Simulatore di etichettatore per test rapidi e verifica dell'architettura Strategy (RF2).
    Genera bounding box fittizie ma valide.
    """

    def predict(self, image_path: str, text_prompts: List[str]) -> List[Dict[str, Any]]:
        detections = []
        # Simula il rilevamento casuale di 0, 1 o 2 oggetti
        num_detections = random.randint(0, 2)

        for _ in range(num_detections):
            score = random.uniform(self.confidence_threshold, 0.95)
            selected_cls = random.choice(text_prompts) if text_prompts else "obstacle"

            # Coordinate simulate [x1, y1, x2, y2]
            x1 = random.randint(50, 200)
            y1 = random.randint(50, 200)
            x2 = x1 + random.randint(100, 300)
            y2 = y1 + random.randint(100, 300)

            detections.append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "cls": selected_cls,
                "score": score
            })

        return detections