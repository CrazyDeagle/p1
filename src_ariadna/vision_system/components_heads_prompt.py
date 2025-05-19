# src_ariadna/vision_system/components_heads_prompt.py
import torch
import torch.nn as nn
import math

try:
    from ultralytics.nn.modules.conv import Conv 
except ImportError:
    print("ADVERTENCIA (components_heads_prompt.py): No se pudo importar Conv de Ultralytics. Usando nn.Conv2d como fallback.")
    Conv = None 

# Importar desde los otros módulos en el mismo paquete 'vision_system'
try:
    from .text_encoder_utils import get_text_embedding_dim
    EMBEDDING_DIMENSION = get_text_embedding_dim()
    if EMBEDDING_DIMENSION is None: # Fallback si la inicialización del text encoder falló
        print("WARN (components_heads_prompt.py): get_text_embedding_dim() devolvió None. Usando 512 como fallback.")
        EMBEDDING_DIMENSION = 512
except ImportError:
    print("WARN (components_heads_prompt.py): No se pudo importar get_text_embedding_dim de text_encoder_utils. Usando 512 como fallback.")
    EMBEDDING_DIMENSION = 512

try:
    from .components_extractor import (
        FeatureExtractorYOLOE_MultiScale, 
        CHANNELS_P3_EXTRACTOR, 
        CHANNELS_P4_EXTRACTOR, 
        CHANNELS_P5_EXTRACTOR
    )
except ImportError:
    print("ERROR CRITICO (components_heads_prompt.py): No se pudo importar FeatureExtractorYOLOE_MultiScale o sus canales.")
    print("  Asegúrate de que 'components_extractor.py' esté en la misma carpeta y sea correcto.")
    # Definir dummies para que el script no falle al cargar, pero el entrenamiento fallará
    class FeatureExtractorYOLOE_MultiScale(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); print("FeatureExtractor DUMMY")
        def forward(self, x): return [None, None, None]
    CHANNELS_P3_EXTRACTOR, CHANNELS_P4_EXTRACTOR, CHANNELS_P5_EXTRACTOR = 256, 512, 512


print(f"INFO (components_heads_prompt.py): Configurando cabezas con EMBEDDING_DIMENSION = {EMBEDDING_DIMENSION}")

class SingleScaleDetectionHead_Prompt(nn.Module):
    def __init__(self, in_channels: int, stride: int, reg_max_val: int, embedding_dim: int):
        super().__init__()
        self.stride = stride
        self.reg_max = reg_max_val
        self.embedding_dim = embedding_dim
        
        self.num_regression_ch = 4 * self.reg_max 
        self.num_objectness_ch = 1 
        self.num_embedding_ch = self.embedding_dim
        
        self.num_outputs_per_cell = self.num_regression_ch + self.num_objectness_ch + self.num_embedding_ch
        
        intermediate_channels = max(in_channels // 2, self.num_outputs_per_cell // 4, 64) 
        
        if Conv is not None:
            self.conv_block = nn.Sequential(
                Conv(in_channels, intermediate_channels, k=3, s=1, p=1), 
                Conv(intermediate_channels, intermediate_channels, k=3, s=1, p=1) 
            )
            self.prediction_layer = nn.Conv2d(intermediate_channels, self.num_outputs_per_cell, kernel_size=1, stride=1, padding=0)
        else: 
            print(f"WARN (HeadPrompt s{stride}): Usando nn.Conv2d (Ultralytics Conv no disponible).")
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(intermediate_channels), nn.SiLU(inplace=True),
                nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(intermediate_channels), nn.SiLU(inplace=True)
            )
            self.prediction_layer = nn.Conv2d(intermediate_channels, self.num_outputs_per_cell, kernel_size=1, stride=1, padding=0)
        
        self._initialize_biases() 

    def _initialize_biases(self): 
        if hasattr(self.prediction_layer, 'bias') and self.prediction_layer.bias is not None:
            try:
                # Bias para el canal de objectness (asumiendo que está después de los de regresión)
                objectness_channel_index = self.num_regression_ch 
                initial_objectness_bias = -math.log((1 - 0.01) / 0.01) # Para p_obj_inicial ~0.01
                
                with torch.no_grad():
                    # Inicializar todos los biases a cero podría ser útil
                    self.prediction_layer.bias.data.fill_(0.0)
                    if 0 <= objectness_channel_index < self.prediction_layer.bias.data.shape[0]:
                        self.prediction_layer.bias.data[objectness_channel_index] = initial_objectness_bias
                        # print(f"  HeadPrompt s{self.stride}: Bias de objectness (canal {objectness_channel_index}) inicializado a {initial_objectness_bias:.4f}")
                    else:
                        print(f"WARN (HeadPrompt s{self.stride}): Índice de canal de objectness ({objectness_channel_index}) fuera de rango para biases (shape {self.prediction_layer.bias.data.shape}).")
            except Exception as e_bias:
                print(f"ERROR inicializando biases para HeadPrompt s{self.stride}: {e_bias}")


    def forward(self, x):
        x = self.conv_block(x)
        x = self.prediction_layer(x)
        return x

class MultiScaleDetectionHead_Prompt(nn.Module):
    def __init__(self, channels_list: list[int], strides_list: list[int], reg_max_val: int, embedding_dim: int):
        super().__init__()
        self.reg_max = reg_max_val 
        self.embedding_dim = embedding_dim
        self.strides = strides_list 
        self.nl = len(strides_list) 
        self.no = (4 * self.reg_max) + 1 + self.embedding_dim # DFL + Objectness + EmbeddingDim

        self.head_p3 = SingleScaleDetectionHead_Prompt(channels_list[0], self.strides[0], self.reg_max, self.embedding_dim) 
        self.head_p4 = SingleScaleDetectionHead_Prompt(channels_list[1], self.strides[1], self.reg_max, self.embedding_dim) 
        self.head_p5 = SingleScaleDetectionHead_Prompt(channels_list[2], self.strides[2], self.reg_max, self.embedding_dim) 
        
        self.register_buffer('stride_tensor', torch.tensor(self.strides, dtype=torch.float32), persistent=False)

    def forward(self, features_list: list[torch.Tensor]):
        if len(features_list) != self.nl:
            raise ValueError(f"MultiScaleDetectionHead_Prompt: Se esperaban {self.nl} feature maps, pero se recibieron {len(features_list)}")
        if any(f is None for f in features_list):
             raise ValueError("MultiScaleDetectionHead_Prompt: Una o más feature maps de entrada son None.")
        
        pred_p3 = self.head_p3(features_list[0]) 
        pred_p4 = self.head_p4(features_list[1]) 
        pred_p5 = self.head_p5(features_list[2]) 
        return [pred_p3, pred_p4, pred_p5] 

class ModeloYOLOE_MultiScale_PoC_Prompt(nn.Module):
    def __init__(self, yoloe_base_instance_for_extractor, 
                 reg_max_default: int, 
                 embedding_dim_default: int): 
        super().__init__()
        # Aquí se usa la clase FeatureExtractorYOLOE_MultiScale importada
        self.feature_extractor = FeatureExtractorYOLOE_MultiScale(yoloe_base_instance_for_extractor)
        
        channels_for_heads = [CHANNELS_P3_EXTRACTOR, 
                              CHANNELS_P4_EXTRACTOR, 
                              CHANNELS_P5_EXTRACTOR]
        
        self.reg_max = reg_max_default 
        self.embedding_dim = embedding_dim_default
        self.strides_config = [8, 16, 32]     

        self.detection_head = MultiScaleDetectionHead_Prompt(
            channels_list=channels_for_heads, 
            strides_list=self.strides_config, 
            reg_max_val=self.reg_max, 
            embedding_dim=self.embedding_dim
        )
        
        # Atributo 'model' para compatibilidad con algunas funciones de pérdida o utilidades
        # que esperan iterar sobre self.model.model o acceder a self.model[-1]
        # Aquí, la cabeza de detección es el único componente "entrenable" o "variable"
        self.model = nn.ModuleList([self.detection_head])

    @property 
    def stride(self): 
        return self.detection_head.stride_tensor

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        list_pan_features = self.feature_extractor(x) 
        if any(f is None for f in list_pan_features):
            # Esto podría pasar si el extractor no está bien configurado o los índices Pk son incorrectos
            print("ERROR en ModeloYOLOE_MultiScale_PoC_Prompt: El extractor de features devolvió None.")
            # Crear tensores dummy con la forma que la cabeza esperaría para evitar un crash total
            # pero esto indica un problema serio en la extracción.
            dummy_shapes_expected_by_head = [
                (x.shape[0], CHANNELS_P3_EXTRACTOR, x.shape[2]//8, x.shape[3]//8),
                (x.shape[0], CHANNELS_P4_EXTRACTOR, x.shape[2]//16, x.shape[3]//16),
                (x.shape[0], CHANNELS_P5_EXTRACTOR, x.shape[2]//32, x.shape[3]//32),
            ]
            list_pan_features = [torch.randn(s, device=x.device) for s in dummy_shapes_expected_by_head]
            print("  Se usarán features dummy para la cabeza de detección.")
            
        predictions_list = self.detection_head(list_pan_features) 
        return predictions_list