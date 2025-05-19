# src_ariadna/vision_system/__init__.py
print("DEBUG: Cargando src_ariadna.vision_system.__init__.py...")
try:
    from .components_extractor import (
        FeatureExtractorYOLOE_MultiScale,
        CHANNELS_P3_EXTRACTOR,
        CHANNELS_P4_EXTRACTOR,
        CHANNELS_P5_EXTRACTOR
    )
    print("INFO (vision_system.__init__): componentes_extractor importados.")
except ImportError as e:
    print(f"WARN (vision_system.__init__): No se pudo importar de components_extractor: {e}")
    class FeatureExtractorYOLOE_MultiScale: pass
    CHANNELS_P3_EXTRACTOR, CHANNELS_P4_EXTRACTOR, CHANNELS_P5_EXTRACTOR = 0,0,0

try:
    from .text_encoder_utils import (
        initialize_text_encoder,
        get_text_embedding_dim,
        get_text_embeddings_batch
    )
    # EMBEDDING_DIMENSION se obtendrá de get_text_embedding_dim en components_heads_prompt
    print("INFO (vision_system.__init__): text_encoder_utils importados.")
except ImportError as e:
    print(f"WARN (vision_system.__init__): No se pudo importar de text_encoder_utils: {e}")
    def initialize_text_encoder(*args, **kwargs): pass
    def get_text_embedding_dim(*args, **kwargs): return 512
    def get_text_embeddings_batch(*args, **kwargs): return None

try:
    from .components_heads_prompt import (
        SingleScaleDetectionHead_Prompt,
        MultiScaleDetectionHead_Prompt,
        ModeloYOLOE_MultiScale_PoC_Prompt,
        EMBEDDING_DIMENSION # Esta es la que se usa en otros módulos
    )
    print(f"INFO (vision_system.__init__): components_heads_prompt importados. EMBEDDING_DIMENSION={EMBEDDING_DIMENSION}")
except ImportError as e:
    print(f"WARN (vision_system.__init__): No se pudo importar de components_heads_prompt: {e}")
    class SingleScaleDetectionHead_Prompt: pass
    class MultiScaleDetectionHead_Prompt: pass
    class ModeloYOLOE_MultiScale_PoC_Prompt: pass
    EMBEDDING_DIMENSION = 512

try:
    from .losses_prompt import YOLOEPromptLoss
    print("INFO (vision_system.__init__): losses_prompt importado.")
except ImportError as e:
    print(f"INFO (vision_system.__init__): losses_prompt.py no encontrado o con error de importación: {e}")
    class YOLOEPromptLoss: pass # DUMMY