# src_ariadna/vision_system/text_encoder_utils.py
import torch
import torch.nn.functional as F
import os
import pathlib
import traceback
from typing import Optional, List

# Importar YOLOE de Ultralytics para usar su get_text_pe
try:
    from ultralytics import YOLO as YOLOE_Ultralytics 
    ULTRALYTICS_YOLOE_AVAILABLE = True
    print("INFO (text_encoder_utils): Ultralytics YOLOE importado correctamente.")
except ImportError:
    ULTRALYTICS_YOLOE_AVAILABLE = False
    YOLOE_Ultralytics = None 
    print("ERROR CRITICO (text_encoder_utils): No se pudo importar YOLOE de Ultralytics. Las funciones de embedding de texto no funcionarán.")

_YOLOE_TEXT_ENCODER_MODEL_INSTANCE: Optional[YOLOE_Ultralytics] = None # type: ignore
_TEXT_ENCODER_DEVICE: str = 'cpu' 
_EMBEDDING_DIM: Optional[int] = None

try:
    _THIS_FILE_DIR = pathlib.Path(__file__).parent.resolve()
    _PROJECT_ROOT_FOR_MODELS = _THIS_FILE_DIR.parent.parent 
    DEFAULT_YOLOE_CHECKPOINT_PATH_FOR_TEXT_ENCODER = str(_PROJECT_ROOT_FOR_MODELS / "models_base" / "yoloe-11l-seg.pt")
except NameError: 
    DEFAULT_YOLOE_CHECKPOINT_PATH_FOR_TEXT_ENCODER = "models_base/yoloe-11l-seg.pt"
    print(f"WARN (text_encoder_utils): __file__ no definido, usando ruta por defecto para YOLOE: {DEFAULT_YOLOE_CHECKPOINT_PATH_FOR_TEXT_ENCODER}")


def initialize_text_encoder(device: str = 'cpu', yoloe_model_path_str: Optional[str] = None) -> bool:
    """
    Inicializa el "codificador de texto" utilizando una instancia del modelo YOLOE
    para acceder a su método get_text_pe(). Carga el modelo YOLOE en el device especificado.
    """
    global _YOLOE_TEXT_ENCODER_MODEL_INSTANCE, _TEXT_ENCODER_DEVICE, _EMBEDDING_DIM

    current_model_path = yoloe_model_path_str if yoloe_model_path_str else DEFAULT_YOLOE_CHECKPOINT_PATH_FOR_TEXT_ENCODER
    
    # Verificar si ya está inicializado con los mismos parámetros
    if _YOLOE_TEXT_ENCODER_MODEL_INSTANCE is not None and \
       _TEXT_ENCODER_DEVICE == device and \
       hasattr(_YOLOE_TEXT_ENCODER_MODEL_INSTANCE, 'ckpt_path') and \
       _YOLOE_TEXT_ENCODER_MODEL_INSTANCE.ckpt_path == str(pathlib.Path(current_model_path).resolve()):
        # print(f"INFO (text_encoder_utils): YOLOE text encoder ya inicializado en {device} con {current_model_path}.")
        return True

    if not ULTRALYTICS_YOLOE_AVAILABLE:
        print("ERROR CRITICO (text_encoder_utils): Librería YOLOE (Ultralytics) no disponible.")
        return False

    _TEXT_ENCODER_DEVICE = device # Actualizar el device global
    actual_yoloe_path = pathlib.Path(current_model_path).resolve()

    if not actual_yoloe_path.is_file():
        print(f"ERROR (text_encoder_utils): Checkpoint de YOLOE para Text Encoder no encontrado en {actual_yoloe_path}")
        _YOLOE_TEXT_ENCODER_MODEL_INSTANCE = None; _EMBEDDING_DIM = None; return False

    try:
        print(f"INFO (text_encoder_utils): Cargando modelo YOLOE desde '{actual_yoloe_path}' para Text Encoder en {_TEXT_ENCODER_DEVICE}...")
        _YOLOE_TEXT_ENCODER_MODEL_INSTANCE = YOLOE_Ultralytics(str(actual_yoloe_path))
        _YOLOE_TEXT_ENCODER_MODEL_INSTANCE.ckpt_path = str(actual_yoloe_path) 
        
        _YOLOE_TEXT_ENCODER_MODEL_INSTANCE.to(_TEXT_ENCODER_DEVICE)
        _YOLOE_TEXT_ENCODER_MODEL_INSTANCE.eval()
        print(f"INFO (text_encoder_utils): Modelo YOLOE base cargado en {_TEXT_ENCODER_DEVICE} para text_pe.")

        if not hasattr(_YOLOE_TEXT_ENCODER_MODEL_INSTANCE, 'get_text_pe'):
            print("ERROR (text_encoder_utils): La instancia YOLOE cargada no tiene el método 'get_text_pe'.")
            _YOLOE_TEXT_ENCODER_MODEL_INSTANCE = None; _EMBEDDING_DIM = None; return False

        dummy_text = ["hello world"] 
        with torch.no_grad():
            dummy_output_pe = _YOLOE_TEXT_ENCODER_MODEL_INSTANCE.get_text_pe(dummy_text)
        
        final_embedding_tensor: Optional[torch.Tensor] = None
        if isinstance(dummy_output_pe, torch.Tensor):
            final_embedding_tensor = dummy_output_pe
        elif isinstance(dummy_output_pe, dict) and 'pe' in dummy_output_pe and isinstance(dummy_output_pe['pe'], torch.Tensor):
            final_embedding_tensor = dummy_output_pe['pe']
        
        if final_embedding_tensor is None or final_embedding_tensor.ndim < 2:
             raise ValueError(f"Salida de get_text_pe no es un tensor válido o no tiene al menos 2 dimensiones. Recibido: {type(dummy_output_pe)}")
        
        _EMBEDDING_DIM = final_embedding_tensor.shape[-1]
        print(f"INFO (text_encoder_utils): EMBEDDING_DIM determinado (desde YOLOE.get_text_pe): {_EMBEDDING_DIM}")
        return True

    except Exception as e:
        print(f"ERROR (text_encoder_utils): No se pudo cargar/usar el modelo YOLOE desde {actual_yoloe_path} para text_pe: {e}")
        traceback.print_exc()
        _YOLOE_TEXT_ENCODER_MODEL_INSTANCE = None; _EMBEDDING_DIM = None; return False

def get_text_embedding_dim() -> int:
    global _EMBEDDING_DIM
    if _EMBEDDING_DIM is None:
        print("WARN (text_encoder_utils): EMBEDDING_DIM no ha sido determinado. Intentando inicialización por defecto en CPU...")
        # Intenta inicializar con el device 'cpu' por defecto si no se ha llamado antes explícitamente
        if not initialize_text_encoder(device='cpu'): 
            print("ERROR CRITICO (text_encoder_utils): Fallo en inicialización para determinar EMBEDDING_DIM. Usando 512 como fallback.")
            return 512 
    return _EMBEDDING_DIM if _EMBEDDING_DIM is not None else 512

def get_text_embeddings_batch(prompt_list: List[str], target_device: str = 'cpu') -> torch.Tensor:
    global _YOLOE_TEXT_ENCODER_MODEL_INSTANCE, _TEXT_ENCODER_DEVICE
    
    current_embedding_dim = get_text_embedding_dim() 

    if _YOLOE_TEXT_ENCODER_MODEL_INSTANCE is None or _TEXT_ENCODER_DEVICE != target_device:
        print(f"INFO (text_encoder_utils): YOLOE text encoder no listo para {target_device}. (Re)Inicializando...")
        if not initialize_text_encoder(device=target_device): 
            print("ERROR (text_encoder_utils): Falló (re)inicialización de YOLOE para text_pe. Devolviendo DUMMY embeddings.")
            if not prompt_list: return torch.empty((0, current_embedding_dim), device=target_device)
            return F.normalize(torch.randn(len(prompt_list), current_embedding_dim, device=target_device), p=2, dim=1)

    if not prompt_list: 
        return torch.empty((0, current_embedding_dim), device=_TEXT_ENCODER_DEVICE)

    try:
        with torch.no_grad():
            # get_text_pe se ejecuta en el device del modelo (_TEXT_ENCODER_DEVICE)
            output_pe = _YOLOE_TEXT_ENCODER_MODEL_INSTANCE.get_text_pe(prompt_list)
        
        embeddings: Optional[torch.Tensor] = None
        if isinstance(output_pe, torch.Tensor):
            embeddings = output_pe
        elif isinstance(output_pe, dict) and 'pe' in output_pe and isinstance(output_pe['pe'], torch.Tensor):
            embeddings = output_pe['pe']
        
        if embeddings is None:
             raise ValueError(f"Salida de get_text_pe no fue un Tensor o Dict esperado: {type(output_pe)}")
        if embeddings.shape[-1] != current_embedding_dim: # Comparar con la dimensión obtenida
            raise ValueError(f"Dimensión de embedding de get_text_pe ({embeddings.shape[-1]}) no coincide con la esperada ({current_embedding_dim})")

        # Asegurar que el embedding final esté en el target_device solicitado
        return embeddings.to(target_device) 
    except Exception as e:
        print(f"ERROR (text_encoder_utils): Durante la generación de embeddings con YOLOE.get_text_pe: {e}"); traceback.print_exc()
        if not prompt_list: return torch.empty((0, current_embedding_dim), device=target_device)
        return F.normalize(torch.randn(len(prompt_list), current_embedding_dim, device=target_device),p=2,dim=1)

if __name__ == '__main__':
    print("Ejecutando prueba de text_encoder_utils.py (usando YOLOE.get_text_pe)...")
    
    try:
        this_file_dir = pathlib.Path(__file__).parent.resolve()
        project_root_dir_for_test = this_file_dir.parent.parent 
        
        yoloe_checkpoint_for_test_str = DEFAULT_YOLOE_CHECKPOINT_PATH_FOR_TEXT_ENCODER
        if not pathlib.Path(yoloe_checkpoint_for_test_str).is_absolute():
             yoloe_checkpoint_for_test = (project_root_dir_for_test / yoloe_checkpoint_for_test_str).resolve()
        else:
             yoloe_checkpoint_for_test = pathlib.Path(yoloe_checkpoint_for_test_str).resolve()
        
        print(f"  Ruta YOLOE base (para prueba): {yoloe_checkpoint_for_test}")
    except NameError: 
        yoloe_checkpoint_for_test = pathlib.Path(DEFAULT_YOLOE_CHECKPOINT_PATH_FOR_TEXT_ENCODER).resolve()
        print(f"  WARN: __file__ no definido, usando ruta por defecto y resolviendo: {yoloe_checkpoint_for_test}")

    print("\n--- Probando en CPU ---")
    if initialize_text_encoder(device='cpu', yoloe_model_path_str=str(yoloe_checkpoint_for_test)):
        test_prompts_cpu = ["a photo of a dog", "a drawing of a cat playing with yarn"]
        embeddings_cpu = get_text_embeddings_batch(test_prompts_cpu, target_device='cpu')
        if embeddings_cpu is not None and embeddings_cpu.numel() > 0 :
            print(f"CPU Embeddings shape: {embeddings_cpu.shape}")
            print(f"  Dimensionalidad de embedding (de get_text_embedding_dim): {get_text_embedding_dim()}")
            print(f"  Primer embedding CPU (primeros 5 valores):\n{embeddings_cpu[0, :5]}")
    else:
        print("Fallo inicialización en CPU para prueba.")

    if torch.cuda.is_available():
        print("\n--- Probando en GPU ---")
        if initialize_text_encoder(device='cuda', yoloe_model_path_str=str(yoloe_checkpoint_for_test)):
            test_prompts_gpu = ["an apple on a table", "a red car driving on a street"]
            embeddings_gpu = get_text_embeddings_batch(test_prompts_gpu, target_device='cuda')
            if embeddings_gpu is not None and embeddings_gpu.numel() > 0:
                print(f"GPU Embeddings shape: {embeddings_gpu.shape}")
                print(f"  Primer embedding GPU (primeros 5 valores):\n{embeddings_gpu[0, :5]}")
        else:
            print("Fallo inicialización en GPU para prueba.")
    else:
        print("\nCUDA no disponible, saltando prueba en GPU.")
    
    print("\nPrueba de text_encoder_utils completada.")