import os

import torch

from observatory.datasets.turl_wiki_tables import build_entity_id_map
from observatory.properties.entity_stability import run_bert, run_turl


if __name__ == "__main__":
    # --------------------------------------------------
    # Property test configuration.
    # --------------------------------------------------
    DATA_DIR = "../data/"
    OUTPUT_DIR = "../artifacts/entity_stability/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    MIN_ENT_COUNT = 2
    SAMPLE_SIZE = 1000
    """
    Whether use metadata of an entity when inferring embeddings.
        "no": no metadata,
        "single": use metadata from a single table,
        "all": use metadata from all tables.
    """
    USE_METADATA = "all"
    
    MODELS_TO_RUN = ["bert-base-uncased", "turl"]
    if "turl" in MODELS_TO_RUN:
        TURL_CONFIG = "/home/congtj/observatory/observatory/models/TURL/configs/table-base-config_v2.json"
        TURL_CKPT_PATH = "/ssd/congtj/observatory/pytorch_model.bin"
    DEVICE = torch.device("cuda:0")

    # --------------------------------------------------
    # Run property test.
    # --------------------------------------------------
    entity_vocab, entity_id_map = build_entity_id_map(DATA_DIR, MIN_ENT_COUNT, SAMPLE_SIZE)
    
    for model_name in MODELS_TO_RUN:
        output_file_path = os.path.join(OUTPUT_DIR, f"{model_name}_embeddings_{USE_METADATA}-metadata.pkl")

        if os.path.exists(output_file_path):
            print("=" * 50)
            print(f"*{output_file_path}* already exists!")
            continue

        if model_name.startswith("bert"):
            run_bert(DATA_DIR, entity_id_map, model_name, SAMPLE_SIZE, USE_METADATA, output_file_path, DEVICE)
        elif model_name == "turl":
            run_turl(DATA_DIR, entity_vocab, entity_id_map, TURL_CONFIG, TURL_CKPT_PATH, SAMPLE_SIZE, USE_METADATA, output_file_path, DEVICE)


    # TODO: Add evaluation code
    # for model_name in MODELS_TO_RUN