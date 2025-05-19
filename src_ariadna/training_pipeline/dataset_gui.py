# src_ariadna/training_pipeline/dataset_gui.py
from typing import List, Optional
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json
import pathlib
import traceback # Para imprimir excepciones detalladas

# Función de preprocesamiento de imagen (ajusta esto a tus necesidades exactas)
# Esta es una versión básica, la que usabas en tu train_detection_heads.py podría ser más completa.
def preprocesar_frame_para_yoloe_train(frame_bgr: np.ndarray, target_size_tuple: tuple[int, int]=(640, 640)) -> Optional[torch.Tensor]:
    try:
        # Mantener aspect ratio y rellenar
        h0, w0 = frame_bgr.shape[:2]
        r = min(target_size_tuple[0] / h0, target_size_tuple[1] / w0)
        
        new_unpad_w, new_unpad_h = int(round(w0 * r)), int(round(h0 * r))
        
        if w0 != new_unpad_w or h0 != new_unpad_h:
            img_resized = cv2.resize(frame_bgr, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = frame_bgr.copy()

        dw, dh = target_size_tuple[1] - new_unpad_w, target_size_tuple[0] - new_unpad_h
        dw /= 2; dh /= 2 # padding en ambos lados

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_rgb.transpose(2,0,1))).float() / 255.0
        return img_tensor
    except Exception as e_prep:
        print(f"ERROR en preprocesar_frame_para_yoloe_train: {e_prep}")
        traceback.print_exc()
        return None

class GuiPromptDataset(Dataset):
    def __init__(self, json_annotations_dir_str: str, image_dir_str: str, img_target_size: int = 640):
        self.json_dir = pathlib.Path(json_annotations_dir_str)
        self.img_dir = pathlib.Path(image_dir_str)
        self.img_size = img_target_size 
        self.samples = [] 

        if not self.img_dir.is_dir(): 
            print(f"ERROR DS (GuiPromptDataset): Directorio de imágenes no encontrado: {self.img_dir}"); return
        if not self.json_dir.is_dir(): 
            print(f"ERROR DS (GuiPromptDataset): Directorio de anotaciones JSON no encontrado: {self.json_dir}"); return

        print(f"DS (GuiPromptDataset): Cargando dataset desde JSONs en '{self.json_dir}' e imágenes en '{self.img_dir}'...")
        json_files_in_dir = sorted([f for f in os.listdir(self.json_dir) if f.endswith('.json')])
        
        loaded_samples_count = 0
        processed_files_count = 0

        for i, json_file_name in enumerate(json_files_in_dir):
            processed_files_count += 1
            json_path = self.json_dir / json_file_name
            img_filename_base = json_file_name[:-5] 
            
            img_path_found: Optional[pathlib.Path] = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                if (p := self.img_dir / (img_filename_base + ext)).exists():
                    img_path_found = p; break
            
            if not img_path_found:
                if i < 10 or (i % 50 == 0 and i > 0): print(f"  DS WARN: No se encontró imagen para '{json_file_name}' en '{self.img_dir}'. Saltando JSON."); 
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
                if not all(k in data for k in ["image_width", "image_height", "annotations"]):
                    if i<5: print(f"  DS WARN: JSON '{json_file_name}' sin estructura esperada. Saltando."); 
                    continue
                
                iw_orig, ih_orig = data["image_width"], data["image_height"]
                if iw_orig <=0 or ih_orig <=0:
                    if i<5: print(f"  DS WARN: JSON '{json_file_name}' con dimensiones de imagen inválidas ({iw_orig}x{ih_orig}). Saltando.");
                    continue

                annotations_for_this_image = []
                for ann_idx, ann in enumerate(data.get("annotations", [])):
                    if not all(k in ann for k in ["prompt_text", "bbox_xyxy_pixel"]): 
                        if i<5: print(f"  DS WARN: Anotación {ann_idx} en '{json_file_name}' incompleta. Saltando anotación."); 
                        continue
                    
                    prompt = str(ann["prompt_text"]).strip()
                    bbox_px = ann["bbox_xyxy_pixel"]
                    obj_tgt = float(ann.get("objectness_target", 1.0))

                    if not prompt: 
                        if i<5: print(f"  DS WARN: Prompt vacío en anotación {ann_idx}, '{json_file_name}'. Usando fallback o saltando anotación."); 
                        prompt = "objeto_desconocido" # O saltar con 'continue'

                    if not(len(bbox_px)==4 and all(isinstance(c,(int, float)) for c in bbox_px)): 
                        if i<5: print(f"  DS WARN: Bbox malformada {bbox_px} en anotación {ann_idx}, '{json_file_name}'. Saltando anotación."); 
                        continue
                    
                    x1,y1,x2,y2=map(float, bbox_px) # Convertir a float para cálculos
                    if x1>=x2 or y1>=y2 or x1<0 or y1<0 or x2>iw_orig or y2>ih_orig: 
                        if i<5: print(f"  DS WARN: Bbox inválida ({x1},{y1},{x2},{y2}) vs ({iw_orig}x{ih_orig}) en anotación {ann_idx}, '{json_file_name}'. Saltando anotación."); 
                        continue
                    
                    cx_n=((x1+x2)/2)/iw_orig; cy_n=((y1+y2)/2)/ih_orig; w_n=(x2-x1)/iw_orig; h_n=(y2-y1)/ih_orig
                    current_bbox_norm = [cx_n, cy_n, w_n, h_n]
                    if not all(0<=val<=1.0001 for val in current_bbox_norm): # Margen pequeño para flotantes
                        if i<5: print(f"  DS WARN: Bbox norm fuera [0,1] ({current_bbox_norm}) en anotación {ann_idx}, '{json_file_name}'. Saltando anotación."); 
                        continue
                    
                    annotations_for_this_image.append({
                        "prompt_text": prompt, "bbox_cxcywh_norm": current_bbox_norm, "objectness_target": obj_tgt
                    })
                
                if annotations_for_this_image:
                    self.samples.append((str(img_path_found), annotations_for_this_image))
                    loaded_samples_count += 1
            except json.JSONDecodeError:
                if i<5: print(f"  DS WARN: Error decodificando JSON '{json_file_name}'. Saltando.")
            except Exception as e_ds_item:
                if i<5: print(f"  DS ERROR: Procesando JSON '{json_file_name}': {e_ds_item}"); traceback.print_exc()
        
        print(f"DS (GuiPromptDataset): {loaded_samples_count} muestras (imágenes con anotaciones válidas) cargadas de {processed_files_count} archivos JSON procesados.")
        if loaded_samples_count == 0 and processed_files_count > 0 : 
            print(f"  ¡¡¡ERROR DS!!! No se cargaron muestras válidas. Revisa tus archivos JSON y las rutas de las imágenes.")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Optional[torch.Tensor], List[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
        img_path_str, annotations_list = self.samples[idx]
        try:
            img_bgr = cv2.imread(img_path_str)
            if img_bgr is None: 
                raise IOError(f"cv2.imread devolvió None para la imagen: {img_path_str}")
            
            # El preproceso debe devolver HWC para ToTensor o CHW si ya lo hace
            img_tensor = preprocesar_frame_para_yoloe_train(img_bgr, (self.img_size, self.img_size))
            if img_tensor is None: 
                raise ValueError("preprocesar_frame_para_yoloe_train devolvió None")

        except Exception as e:
            print(f"DS __getitem__ ERROR al cargar/preprocesar imagen '{os.path.basename(img_path_str)}': {e}"); 
            return None, [], None, None # Devolver Nones para ser filtrado por collate_fn

        prompts_gt_list = [ann["prompt_text"] for ann in annotations_list]
        bboxes_gt_tensor = torch.tensor([ann["bbox_cxcywh_norm"] for ann in annotations_list], dtype=torch.float32)
        objectness_gt_tensor = torch.tensor([ann["objectness_target"] for ann in annotations_list], dtype=torch.float32)
        
        return img_tensor, prompts_gt_list, bboxes_gt_tensor, objectness_gt_tensor