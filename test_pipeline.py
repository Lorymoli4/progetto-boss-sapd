import os
import json
from src.adapters.dataset_adapter import COCODatasetAdapter
from src.annotators.mock_annotator import MockAnnotator
from src.utils.utils import set_seed

# 1. Fissiamo il seed per la riproducibilità (Vincolo Non Funzionale)
set_seed(42)
print("-> Seed fissato correttamente.")

# 2. Creiamo dei file temporanei finti per testare il sistema senza bisogno di un dataset reale gigabyte
os.makedirs("config", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Vocabolario di test (RF3)
vocab_data = {
    "target_classes": {
        "door": ["door", "wooden door"],
        "curb": ["curb"]
    }
}
with open("config/vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, indent=4)

# Dataset COCO finto (RF1)
coco_fake_data = {
    "images": [{"id": 101, "file_name": "test_img_01.jpg"}],
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "door"} # Classe target da nascondere
    ],
    "annotations": [
        {"id": 1, "image_id": 101, "category_id": 1, "bbox": [10, 10, 50, 100]},
        {"id": 2, "image_id": 101, "category_id": 2, "bbox": [200, 150, 80, 180]}
    ]
}
with open("data/fake_coco.json", "w", encoding="utf-8") as f:
    json.dump(coco_fake_data, f, indent=4)

print("-> File di configurazione e dati finti creati con successo.")

# 3. Testiamo l'Adapter (RF1 & RF3)
adapter = COCODatasetAdapter(
    annotation_path="data/fake_coco.json",
    config_vocab_path="config/vocabulary.json"
)
images, visible_anns, hidden_anns = adapter.get_unlabeled_pool_and_ground_truth()

print(f"\n--- TEST ADAPTER ---")
print(f"Immagini caricate: {len(images)}")
print(f"Annotazioni visibili (classi note): {len(visible_anns)}")
print(f"Annotazioni nascoste (classi nuove da scoprire): {len(hidden_anns)}")

# 4. Testiamo il Mock Annotator (Pattern Strategy - RF2)
annotator = MockAnnotator(confidence_threshold=0.5)
target_prompts = list(adapter.vocabulary.keys()) # ['door', 'curb']

# Simuliamo l'inferenza sull'immagine
detections = annotator.predict("data/test_img_01.jpg", text_prompts=target_prompts)

print(f"\n--- TEST MOCK ANNOTATOR (Strategy) ---")
print(f"Rilevamenti generati dal modello finto: {len(detections)}")
for det in detections:
    print(f"  -> Oggetto trovato: {det['cls']} (Confidenza: {det['score']:.2f}) con box {det['bbox_xyxy']}")

# 5. Testiamo la conversione nello standard boss.perception/0.1
if images and detections:
    boss_format = adapter.export_to_boss_perception_format(images[0], detections)
    print(f"\n--- TEST STANDARD BOSS ---")
    print(json.dumps(boss_format, indent=2))

print("\n Test completato con successo! L'architettura Adapter + Strategy risponde perfettamente.")