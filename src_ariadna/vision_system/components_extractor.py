# src_ariadna/vision_system/components_extractor.py
import torch
import torch.nn as nn
from typing import List, Optional, Union
import traceback 

try:
    from ultralytics.nn.modules.block import Concat
    ULTRALYTICS_CONCAT_AVAILABLE = True
except ImportError:
    try:
        from ultralytics.nn.modules import Concat
        ULTRALYTICS_CONCAT_AVAILABLE = True
    except ImportError:
        print("ADVERTENCIA (components_extractor.py): No se pudo importar Concat de Ultralytics.")
        ULTRALYTICS_CONCAT_AVAILABLE = False
        Concat = None # Definir como None para que el código no falle si se usa en type hints


# Canales de SALIDA de las capas P3, P4, P5 del neck de yoloe-11l-seg.pt
# Estos valores son cruciales para que las cabezas de detección personalizadas
# se conecten correctamente. Debes verificarlos inspeccionando la arquitectura
# del modelo YOLOE base si tienes dudas.
CHANNELS_P3_EXTRACTOR = 256
CHANNELS_P4_EXTRACTOR = 512
CHANNELS_P5_EXTRACTOR = 512 # Comúnmente, P5 tiene el mismo número de canales que P4 o el doble. Verifica para yoloe-11l-seg.pt.


class FeatureExtractorYOLOE_MultiScale(nn.Module):
    def __init__(self, yoloe_model_instance_for_extraction):
        super().__init__()
        self.yoloe_internal_model_modulelist: Optional[nn.Sequential] = None
        self.save_list_from_yoloe_base: Optional[List[int]] = None 
        self.structure_ok_for_extraction: bool = False
        
        # Índices de los módulos en nn.Sequential cuyas SALIDAS son P3, P4, P5.
        # Estos módulos DEBEN estar en la lista 'save' del modelo base.
        self.idx_P3: int = 16 
        self.idx_P4: int = 19
        self.idx_P5: int = 22 
        
        required_module_indices_for_p_layers = [self.idx_P3, self.idx_P4, self.idx_P5]

        if hasattr(yoloe_model_instance_for_extraction, 'model') and \
           isinstance(yoloe_model_instance_for_extraction.model, nn.Module):
            
            internal_full_model = yoloe_model_instance_for_extraction.model 
            
            for param in internal_full_model.parameters():
                param.requires_grad = False 
            
            if hasattr(internal_full_model, 'model') and \
               isinstance(internal_full_model.model, nn.Sequential) and \
               hasattr(internal_full_model, 'save') and \
               isinstance(internal_full_model.save, list):
                
                self.yoloe_internal_model_modulelist = internal_full_model.model 
                self.save_list_from_yoloe_base = internal_full_model.save 
                
                valid_pk_indices_range = all(idx < len(self.yoloe_internal_model_modulelist) for idx in required_module_indices_for_p_layers)
                pk_indices_are_in_save_list = all(pk_idx in self.save_list_from_yoloe_base for pk_idx in required_module_indices_for_p_layers)

                if valid_pk_indices_range and pk_indices_are_in_save_list:
                    self.structure_ok_for_extraction = True
                    print(f"INFO (FeatureExtractor): Estructura OK.")
                    print(f"  Módulos para P3, P4, P5 en índices: {self.idx_P3}, {self.idx_P4}, {self.idx_P5}")
                    print(f"  Lista 'save' del modelo base: {self.save_list_from_yoloe_base}")
                else:
                    print(f"WARN (FeatureExtractor): No se cumplen todas las condiciones para la extracción.")
                    if not valid_pk_indices_range: print(f"  - Al menos un índice Pk ({required_module_indices_for_p_layers}) está fuera del rango del ModuleList (longitud {len(self.yoloe_internal_model_modulelist)}).")
                    if not pk_indices_are_in_save_list: print(f"  - Al menos un índice Pk ({required_module_indices_for_p_layers}) no se encuentra en la lista 'save' del modelo base ({self.save_list_from_yoloe_base}). ¡ESTO ES IMPORTANTE!")
            else:
                print(f"WARN (FeatureExtractor): Estructura interna de YOLOE no esperada (nn.Sequential o .save).")
        else:
            print(f"WARN (FeatureExtractor): Instancia de yoloe_model_instance_for_extraction no válida o sin atributo .model.")
        
        if not ULTRALYTICS_CONCAT_AVAILABLE:
             print("WARN (FeatureExtractor): Concat de Ultralytics no disponible. La extracción podría fallar si el modelo lo usa.")
        
        if not self.structure_ok_for_extraction:
            print(f"ERROR (FeatureExtractor): La estructura no es adecuada para la extracción. Se usarán DUMMY features. Revisa los mensajes anteriores y los índices Pk.")

    def forward(self, x: torch.Tensor) -> List[Optional[torch.Tensor]]:
        dummy_p3 = torch.randn(x.shape[0], CHANNELS_P3_EXTRACTOR, x.shape[2]//8, x.shape[3]//8, device=x.device)
        dummy_p4 = torch.randn(x.shape[0], CHANNELS_P4_EXTRACTOR, x.shape[2]//16, x.shape[3]//16, device=x.device)
        dummy_p5 = torch.randn(x.shape[0], CHANNELS_P5_EXTRACTOR, x.shape[2]//32, x.shape[3]//32, device=x.device)
        
        if not self.structure_ok_for_extraction or self.yoloe_internal_model_modulelist is None:
            # self.save_list_from_yoloe_base podría ser una lista vacía si el modelo no guarda nada,
            # pero si es None, es un problema de inicialización.
            print("DEBUG (FeatureExtractor Forward): Usando dummy features (structure_ok es False o modulelist es None).")
            return [dummy_p3, dummy_p4, dummy_p5]
        
        # y almacena la salida de CADA módulo, para que m.f pueda indexar correctamente
        y: List[Optional[torch.Tensor]] = [] 
        current_x_for_next_layer: Union[torch.Tensor, List[torch.Tensor]] = x 

        for i, m_layer in enumerate(self.yoloe_internal_model_modulelist):
            # No procesar la cabeza de detección/segmentación original del modelo base.
            # Algunos modelos pueden incluirla antes del último índice, por lo que
            # se comprueba el tipo de módulo independientemente de su posición.
            is_original_head_module = False
            if hasattr(m_layer, 'type') and m_layer.type in [
                'Detect', 'Segment', 'YOLOEDetect', 'YOLOESegment',
                'Pose', 'OBB', 'WorldDetect', 'v10Detect']:
                is_original_head_module = True
            
            if is_original_head_module:
                # print(f"DEBUG (FeatureExtractor): Saltando cabeza original '{m_layer.type}' en índice {i}.")
                break # Salir del bucle, ya no necesitamos procesar más.

            # Determinar la entrada para la capa actual 'm_layer'
            input_for_current_layer: Union[torch.Tensor, List[torch.Tensor]]
            if m_layer.f != -1:  # Si la entrada no es la salida de la capa inmediatamente anterior
                if isinstance(m_layer.f, int): # Índice único
                    if m_layer.f < 0: # Índice negativo (ej. -1 para la salida anterior en y)
                        input_for_current_layer = y[m_layer.f]
                    elif m_layer.f < len(y): # Índice positivo dentro del rango de y
                        input_for_current_layer = y[m_layer.f]
                    else:
                        print(f"ERROR (FeatureExtractor Forward): Índice m.f={m_layer.f} fuera de rango para y (len={len(y)}) en capa {i} ({m_layer.type}).")
                        return [dummy_p3, dummy_p4, dummy_p5]
                else: # Es una lista de índices (para Concat)
                    input_for_current_layer = []
                    for j_idx in m_layer.f:
                        if j_idx == -1 : 
                            input_for_current_layer.append(current_x_for_next_layer) # La salida de la rama principal antes de Concat
                        elif j_idx < len(y):
                            input_for_current_layer.append(y[j_idx])
                        else:
                            print(f"ERROR (FeatureExtractor Forward): Índice {j_idx} en m.f={m_layer.f} fuera de rango para y (len={len(y)}) en capa {i} ({m_layer.type}).")
                            return [dummy_p3, dummy_p4, dummy_p5]
            else: # f == -1, la entrada es la salida de la capa anterior
                input_for_current_layer = current_x_for_next_layer
            
            # Validar que las entradas no sean None
            if isinstance(input_for_current_layer, list):
                if any(t is None for t in input_for_current_layer):
                    print(f"ERROR (FeatureExtractor Forward): Input para capa {i} ({m_layer.type}) CONTIENE None en lista. f={m_layer.f}")
                    valid_inputs_shapes = [t.shape if hasattr(t,'shape') and t is not None else 'None' for t in input_for_current_layer]
                    print(f"  Shapes de las entradas para Concat: {valid_inputs_shapes}")
                    return [dummy_p3, dummy_p4, dummy_p5]
            elif input_for_current_layer is None :
                print(f"ERROR (FeatureExtractor Forward): Input para capa {i} ({m_layer.type}) es None. f={m_layer.f}")
                return [dummy_p3, dummy_p4, dummy_p5]

            try:
                output_from_module = m_layer(input_for_current_layer) 
            except Exception as e_fwd:
                print(f"ERROR (FeatureExtractor Forward): Capa {i} ({m_layer.type}) falló. f={m_layer.f}.")
                print(f"  Input type: {type(input_for_current_layer)}")
                if isinstance(input_for_current_layer, list):
                    for idx_f, f_tensor in enumerate(input_for_current_layer): print(f"    Input {idx_f} shape: {f_tensor.shape if hasattr(f_tensor, 'shape') and f_tensor is not None else type(f_tensor)}")
                elif hasattr(input_for_current_layer, 'shape'):
                    print(
                        f"  Input shape: {input_for_current_layer.shape if hasattr(input_for_current_layer, 'shape') else 'N/A'}"
                    )
                print(f"  Error: {e_fwd}"); traceback.print_exc(); return [dummy_p3, dummy_p4, dummy_p5]
            
            y.append(output_from_module)
            current_x_for_next_layer = output_from_module # Actualizar para la siguiente iteración si f=-1
        
        # Extraer las features Pk de la lista 'y' usando los índices correctos.
        # Los self.idx_Pk son los índices de los módulos cuyas salidas son P3, P4, P5.
        # Estas salidas deberían estar en 'y' en las posiciones self.idx_Pk.
        try:
            # Asegurarse de que los índices Pk estén dentro del rango de lo que se procesó en 'y'
            if not (self.idx_P3 < len(y) and self.idx_P4 < len(y) and self.idx_P5 < len(y)):
                print(f"ERROR (FeatureExtractor): Índices Pk fuera de rango para la lista 'y' (longitud {len(y)}).")
                print(f"  idx_P3={self.idx_P3}, idx_P4={self.idx_P4}, idx_P5={self.idx_P5}")
                return [dummy_p3, dummy_p4, dummy_p5]

            features_P3 = y[self.idx_P3]
            features_P4 = y[self.idx_P4]
            features_P5 = y[self.idx_P5]
        except IndexError: # No debería ocurrir si la comprobación anterior pasa
            print(f"ERROR CRITICO (FeatureExtractor): IndexError al acceder a y[idx_Pk]. Esto no debería pasar después de la comprobación de rango.")
            return [dummy_p3, dummy_p4, dummy_p5]

        # Verificar si las features extraídas son None (esto podría pasar si el módulo Pk no estaba en self.save_list_from_yoloe_base
        # Y la lógica anterior de `y.append(output if i in self.save_list_from_yoloe_base else None)` se usara.
        # Con la lógica actual de `y.append(output_from_module)`, no deberían ser None si el módulo se ejecutó.
        if features_P3 is None or features_P4 is None or features_P5 is None:
            print(f"ERROR (FeatureExtractor): No se extrajeron todas las features Pk (una o más son None en la lista 'y' en los índices Pk).")
            print(f"  Esto es INESPERADO si los módulos en idx_Pk se ejecutaron y los índices son correctos.")
            print(f"  y[{self.idx_P3}] (P3) es {'None' if features_P3 is None else 'Tensor OK'}")
            print(f"  y[{self.idx_P4}] (P4) es {'None' if features_P4 is None else 'Tensor OK'}")
            print(f"  y[{self.idx_P5}] (P5) es {'None' if features_P5 is None else 'Tensor OK'}")
            # Imprimir la lista save para referencia
            print(f"  Lista 'save' del modelo base: {self.save_list_from_yoloe_base}")
            # Verificar si los módulos Pk están en la lista save (esto ya se hace en __init__)
            if not all(pk_idx in self.save_list_from_yoloe_base for pk_idx in [self.idx_P3, self.idx_P4, self.idx_P5]):
                print("  ADVERTENCIA ADICIONAL: Uno o más de tus idx_P3,P4,P5 NO están en la lista 'save' del modelo base. Sus salidas en 'y' podrían ser None si la lógica de guardado fuera diferente.")
            return [dummy_p3, dummy_p4, dummy_p5]
            
        # print(f"DEBUG (FeatureExtractor): P3 shape: {features_P3.shape}, P4 shape: {features_P4.shape}, P5 shape: {features_P5.shape}")
        return [features_P3, features_P4, features_P5]