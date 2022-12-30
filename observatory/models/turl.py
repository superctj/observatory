import torch

from observatory.models.TURL.model.configuration import TableConfig
from observatory.models.TURL.model.model import HybridTableMaskedLM


def load_turl_model(config_name: str, ckpt_path: str, device):
    config = TableConfig.from_pretrained(config_name)
    model = HybridTableMaskedLM(config, is_simple=True)

    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)
    model.eval()

    return model