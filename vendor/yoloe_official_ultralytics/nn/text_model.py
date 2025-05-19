# Ultralytics YOLO 🚀, AGPL-3.0 license

from abc import abstractmethod
import clip 
import mobileclip 
import torch.nn as nn
from ultralytics.utils.torch_utils import smart_inference_mode
import torch
from ultralytics.utils import LOGGER
import os # Para os.path.exists

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def tokenize(self, texts): # Añadido self
        pass
    
    @abstractmethod
    def encode_text(self, texts, dtype): # Añadido self
        pass

class CLIP(TextModel):
    def __init__(self, size, device):
        super().__init__()
        self.model = clip.load(size, device=device)[0]
        self.to(device)
        self.device = device
        self.eval()
    
    def tokenize(self, texts):
        return clip.tokenize(texts).to(self.device)
    
    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32): # 'texts' aquí son los tokens
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats
        
class MobileCLIP(TextModel):
    
    config_size_map = {
        "s0": "s0",
        "s1": "s1",
        "s2": "s2",
        "b": "b",
        "blt": "b" 
    }
    
    def __init__(self, size="blt", device=None): 
        super().__init__()
        config = self.config_size_map.get(size, size) 
        
        # --- INICIO DE LA MODIFICACIÓN CON RUTA HARDCODEADA ---
        # Ruta absoluta y específica a tu archivo .pt
        hardcoded_checkpoint_path = r"C:\Users\Deagle\Pictures\PROYECTOFINAL\models_base\mobileclip_blt.pt"
        
        # El nombre del archivo que la lógica original construiría (para comparación o fallback)
        original_expected_filename = f"mobileclip_{size}.pt" 

        actual_pretrained_path_to_use: str

        if os.path.exists(hardcoded_checkpoint_path) and checkpoint_filename_matches_size(hardcoded_checkpoint_path, size):
            actual_pretrained_path_to_use = hardcoded_checkpoint_path
            LOGGER.info(f"MobileCLIP: Usando checkpoint local HARDCODEADO: {actual_pretrained_path_to_use}")
        else:
            # Fallback al comportamiento original si el archivo hardcodeado no existe o no coincide con 'size'
            # (esto es para seguridad, pero idealmente el hardcodeado siempre existirá)
            actual_pretrained_path_to_use = original_expected_filename 
            LOGGER.warning(f"MobileCLIP: Checkpoint HARDCODEADO {hardcoded_checkpoint_path} NO encontrado o no coincide con size '{size}'. Usando nombre base '{actual_pretrained_path_to_use}'. La carga podría fallar.")

        try:
            self.model = mobileclip.create_model_and_transforms(
                model_name=f'mobileclip_{config}', 
                pretrained=actual_pretrained_path_to_use,
                device=device
            )[0] 
        except FileNotFoundError as e_fnf:
            LOGGER.error(f"MobileCLIP FileNotFoundError al cargar '{actual_pretrained_path_to_use}': {e_fnf}")
            LOGGER.error(f"Asegúrate de que el archivo existe en la ruta especificada o que el fallback al nombre base puede ser resuelto por la librería mobileclip.")
            raise # Relanzar el error para que sea visible
        except Exception as e_load:
            LOGGER.error(f"MobileCLIP Error genérico al cargar '{actual_pretrained_path_to_use}': {e_load}")
            raise
        # --- FIN DE LA MODIFICACIÓN ---
        
        self.tokenizer = mobileclip.get_tokenizer(f'mobileclip_{config}')
        self.to(device)
        self.device = device
        self.eval()
    
    def tokenize(self, texts: List[str]): 
        text_tokens = self.tokenizer(texts).to(self.device)
        return text_tokens

    @smart_inference_mode()
    def encode_text(self, texts_tokens: torch.Tensor, dtype=torch.float32): 
        text_features = self.model.encode_text(texts_tokens).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

def checkpoint_filename_matches_size(filepath: str, size_key: str) -> bool:
    """Verifica si el nombre de archivo del checkpoint parece coincidir con la clave de tamaño."""
    filename = os.path.basename(filepath)
    # ej. mobileclip_blt.pt debe coincidir con size_key 'blt'
    # ej. mobileclip_s0.pt debe coincidir con size_key 's0'
    expected_part = f"mobileclip_{size_key}.pt"
    return filename == expected_part

def build_text_model(variant: str, device=None):
    LOGGER.info(f"Build text model {variant}")
    try:
        base, size = variant.split(":")
    except ValueError:
        LOGGER.error(f"Formato de variante de TextModel incorrecto: '{variant}'. Debe ser 'base:size' (ej. 'clip:ViT-B/32' o 'mobileclip:blt').")
        raise
        
    if base.lower() == 'clip':
        return CLIP(size, device)
    elif base.lower() == 'mobileclip':
        return MobileCLIP(size, device) 
    else:
        LOGGER.error(f"Variante de TextModel desconocida: '{base}'. Soportadas: 'clip', 'mobileclip'.")
        raise ValueError(f"Variante de TextModel desconocida: {base}")