# progetto-boss-sapd
# Progetto BOSS - SAPD
> Toolkit di etichettatura assistita e active learning per il rilevamento ostacoli.

Repository ufficiale del **Gruppo 46** per il corso di **Sviluppo di Applicazioni e Piattaforme Dati (SAPD)**.  
Autore: **Lorenzo Moliterni**  
Tag di rilascio Milestone 1: `m1`

---

##  Architettura Software (Milestone 1)

L'obiettivo primario della Milestone 1 è definire e validare l'infrastruttura software di base, garantendo il disaccoppiamento tra ingestione dati, configurazione semantica e motori di inferenza tramite pattern architetturali standard:

* **Strategy Pattern (`BaseAnnotator` & `MockAnnotator` - RF2):**  
  Definizione dell'interfaccia astratta `BaseAnnotator`, che stabilisce il contratto unificato per il metodo di inferenza `predict(image_path, text_prompts)`. Per consentire una pipeline di Continuous Integration deterministica e priva di vincoli computazionali pesanti (evitando il caricamento di pesi neurali di grandi dimensioni), è implementato un `MockAnnotator` che simula la generazione di bounding box e punteggi di confidenza controllati su dati sintetici.
* **Adapter Pattern (`COCODatasetAdapter` - RF1):**  
  Modulo per l'ingestione e il parsing dei metadati in formato COCO standard, deputato a separare il pool non etichettato e mascherare la ground truth delle classi target (`hidden_ground_truth`) per la successiva simulazione dell'oracolo.
* **Configurazione Semantica Esterna (RF3):**  
  Le categorie e i prompt testuali non sono cablati nel codice (*hardcoded*), ma risiedono nel file di configurazione `config/vocabulary.json`.
* **Standard Dati Comune (RF4):**  
  Normalizzazione dell'output di percezione secondo lo schema condiviso `boss.perception/0.1`.
* **Riproducibilità (RNF1):**  
  Fissaggio centralizzato dei seed per garantire l'assoluta riproducibilità delle esecuzioni.

---

## Struttura del Repository (Milestone 1)

```text
progetto-boss-sapd/
│
├── config/
│   └── vocabulary.json        # Configurazione categorie e prompt semantici
│
├── src/
│   ├── annotators/
│   │   ├── base_annotator.py  # Interfaccia astratta (Strategy Pattern)
│   │   └── mock_annotator.py  # Annotatore simulato per test architetturali
│   └── adapters/
│       └── dataset_adapter.py # Adapter per dataset COCO e schema boss.perception
│
├── test_pipeline.py           # Script di verifica end-to-end su dati sintetici
├── requirements.txt           # Dipendenze Python minime di progetto
└── README.md                  # Documentazione formale del repository