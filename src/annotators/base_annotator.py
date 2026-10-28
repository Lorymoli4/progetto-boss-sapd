from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseAnnotator(ABC):
    """
    Interfaccia Strategy per gli etichettatori (RF2).
    Permette di scambiare il modello (YOLO-World, Grounding DINO, ecc.)
    senza modificare la pipeline di active learning.
    """

    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def predict(self, image_path: str, text_prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Esegue l'inferenza su un'immagine usando i prompt testuali (RF3).

        Ritorna una lista di rilevamenti nel formato:
        [
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "cls": "door",
                "score": 0.85
            },
            ...
        ]
        """
        pass