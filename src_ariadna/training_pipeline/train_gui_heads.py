# src_ariadna/training_pipeline/train_gui_heads.py
print("DEBUG: Iniciando train_gui_heads.py (Entrenamiento con Prompts y JSON)...")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pathlib # Para manejar rutas
import shutil 
import argparse 
import traceback
import time 

# --- Importar Componentes del Proyecto ---
try:
    from ultralytics import YOLO as YOLOE_Ultralytics
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False; YOLOE_Ultralytics = None; print("ERROR: Ultralytics YOLOE no disponible.")

try:
    # Importar desde el paquete vision_system (un nivel arriba y luego dentro de vision_system)
    from ..vision_system import (
        ModeloYOLOE_MultiScale_PoC_Prompt,
        FeatureExtractorYOLOE_MultiScale,
        initialize_text_encoder,
        get_text_embedding_dim,
        # EMBEDDING_DIMENSION, # Ya no es necesario importarla aquí si get_text_embedding_dim la maneja
        YOLOEPromptLoss # Importar la clase de pérdida (dummy por ahora)
    )
    # Importar Dataset y Collate localmente desde este paquete training_pipeline
    from .dataset_gui import GuiPromptDataset
    from .collate_gui import collate_fn_gui_prompt
    MODEL_COMPONENTS_AVAILABLE = True
    print(f"INFO (train_gui_heads): Componentes de src_ariadna importados.")
except ImportError as e:
    print(f"ERROR CRÍTICO (train_gui_heads): Faltan componentes de src_ariadna: {e}"); traceback.print_exc(); MODEL_COMPONENTS_AVAILABLE = False
# --- Fin Importaciones ---

# --- Configuración ---
# Rutas relativas a la raíz del proyecto (PROYECTOFINAL)
# El script se ejecuta con python -m src_ariadna.training_pipeline.train_gui_heads desde PROYECTOFINAL
try:
    PROJECT_ROOT_ABS = pathlib.Path(__file__).resolve().parent.parent.parent
except NameError: # Fallback si __file__ no está definido
    PROJECT_ROOT_ABS = pathlib.Path(os.getcwd()).resolve() # Asumir CWD es PROYECTOFINAL
    print(f"WARN (train_gui_heads): __file__ no definido, asumiendo PROJECT_ROOT es {PROJECT_ROOT_ABS}")


NOMBRE_TAREA = "GUI_Prompt_Detector_v1_FluxTest" 
PATH_DATASET_BASE_ABS = PROJECT_ROOT_ABS / "datasets" / "dataset_gui"
PATH_JSON_ANNOTATIONS_DIR_ABS = PATH_DATASET_BASE_ABS / "train" / "json_annotations" 
PATH_IMAGES_DIR_ABS = PATH_DATASET_BASE_ABS / "train" / "images"
# Para validación (opcional para esta prueba, pero las rutas deben estar definidas)
PATH_VALID_JSON_ANNOTATIONS_DIR_ABS = PATH_DATASET_BASE_ABS / "valid" / "json_annotations"
PATH_VALID_IMAGES_DIR_ABS = PATH_DATASET_BASE_ABS / "valid" / "images"


PATH_YOLOE_PT_BASE_ABS = PROJECT_ROOT_ABS / "models_base" / "yoloe-11l-seg.pt" 
RUTA_BASE_GUARDADO_TAREA_ABS = PROJECT_ROOT_ABS / "pesos_entrenados_ariadna" / NOMBRE_TAREA
RUTA_CHECKPOINT_MULTIHEAD_LAST = RUTA_BASE_GUARDADO_TAREA_ABS / f"last_head_prompt.pth"
RUTA_CHECKPOINT_MULTIHEAD_BEST = RUTA_BASE_GUARDADO_TAREA_ABS / f"best_head_prompt.pth"

EPOCHS = 2 # SOLO 2 ÉPOCAS PARA PRUEBA DE FLUJO
BATCH_SIZE = 1 # Probar con BATCH_SIZE > 1 si tienes al menos 2 imágenes
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 0.025 
MOMENTUM_ADAMW_B1 = 0.9 
WARMUP_EPOCHS = 0 # Sin warmup para esta prueba simple (o 1 si BATCH_SIZE es pequeño)
IMG_SIZE = 640
REG_MAX_DFL = 16 # Asegúrate que coincida con lo que espera tu modelo/pérdida

# Pesos de la Pérdida (para la futura YOLOEPromptLoss real)
LOSS_BOX_WEIGHT = 7.0; LOSS_DFL_WEIGHT = 1.5; LOSS_OBJ_WEIGHT = 1.0; LOSS_CLS_EMBED_WEIGHT = 2.5
ASSIGNER_CFG = {'topk': 10, 'alpha': 0.5, 'beta': 6.0, 'num_classes': 1} 
# ---------------------------------------------------------------------

def test_dataloader_and_collate(device_str: str):
    print("\n--- INICIANDO PRUEBA DE DATALOADER Y COLLATE FUNCTION ---")
    if not MODEL_COMPONENTS_AVAILABLE:
        print("ERROR: Componentes del modelo no disponibles. Saltando prueba de dataloader.")
        return

    print(f"Usando device para text_embeddings en collate_fn: {device_str}")
    if not initialize_text_encoder(device=device_str, yoloe_model_path_str=str(PATH_YOLOE_PT_BASE_ABS)):
        print("ERROR: Fallo al inicializar text_encoder. Saltando prueba de dataloader.")
        return
    
    print(f"Dimensionalidad de Embedding según text_encoder: {get_text_embedding_dim()}")

    test_train_dataset = GuiPromptDataset(str(PATH_JSON_ANNOTATIONS_DIR_ABS), str(PATH_IMAGES_DIR_ABS), IMG_SIZE)
    if len(test_train_dataset) == 0:
        print(f"ERROR: El dataset de prueba (train) está vacío. Verifica las rutas y los JSONs.")
        print(f"  Ruta JSONs: {PATH_JSON_ANNOTATIONS_DIR_ABS}")
        print(f"  Ruta Imágenes: {PATH_IMAGES_DIR_ABS}")
        return
    
    # Pasar el device al collate_fn para la generación de embeddings
    collate_fn_with_device_for_test = lambda batch: collate_fn_gui_prompt(batch, device=device_str)
    
    # Usar drop_last=False si el número de muestras es menor que BATCH_SIZE para asegurar que se procese algo
    actual_batch_size_test = min(BATCH_SIZE, len(test_train_dataset))
    print(f"Probando DataLoader con BATCH_SIZE={actual_batch_size_test} (drop_last={len(test_train_dataset) >= actual_batch_size_test})")

    test_train_dataloader = DataLoader(
        test_train_dataset, 
        batch_size=actual_batch_size_test, 
        shuffle=True, # Shuffle para ver diferentes muestras si hay más que el tamaño del batch
        collate_fn=collate_fn_with_device_for_test, 
        num_workers=0, # Importante para depuración y evitar problemas con CUDA en workers
        pin_memory=False, # Poner a False si num_workers=0 y hay problemas
        drop_last= len(test_train_dataset) >= actual_batch_size_test # Solo si hay suficientes muestras
    )

    num_batches_to_test = min(2, len(test_train_dataloader)) # Probar unos pocos batches
    if num_batches_to_test == 0 and len(test_train_dataset) > 0 :
        print("WARN: No se pudieron crear batches para el DataLoader de prueba, aunque el dataset tiene muestras.")
        return

    print(f"Iterando sobre {num_batches_to_test} batches del DataLoader de prueba...")
    for i, batch_data in enumerate(test_train_dataloader):
        if i >= num_batches_to_test:
            break
        
        print(f"\n--- Batch de Prueba {i+1}/{num_batches_to_test} ---")
        if batch_data is None or batch_data[0] is None:
            print("  ERROR: collate_fn devolvió None o un batch inválido.")
            continue

        images_b, targets_d, text_embeds_b = batch_data
        
        print(f"  Forma de images_batch: {images_b.shape if isinstance(images_b, torch.Tensor) else 'No Tensor'}")
        print(f"  Forma de text_embeddings_for_batch_classes: {text_embeds_b.shape if isinstance(text_embeds_b, torch.Tensor) else 'No Tensor'}")
        print(f"  Contenido de targets_dict:")
        if isinstance(targets_d, dict):
            for k, v in targets_d.items():
                print(f"    '{k}': shape={v.shape if isinstance(v, torch.Tensor) else type(v)}, dtype={v.dtype if isinstance(v, torch.Tensor) else 'N/A'}")
                if isinstance(v, torch.Tensor) and v.numel() > 0 and v.numel() < 10: # Imprimir algunos valores si es pequeño
                    print(f"      Valores (primeros): {v.flatten()[:5]}")
            
            # Verificar consistencias
            if targets_d['batch_idx'].numel() > 0:
                num_total_gt_objects_in_batch = targets_d['bboxes_gt'].shape[0]
                if targets_d['cls_indices_for_text_embeddings'].shape[0] != num_total_gt_objects_in_batch:
                    print(f"    WARN: Desajuste entre cls_indices ({targets_d['cls_indices_for_text_embeddings'].shape[0]}) y bboxes_gt ({num_total_gt_objects_in_batch})")
                if text_embeds_b.numel() > 0 and torch.any(targets_d['cls_indices_for_text_embeddings'] >= text_embeds_b.shape[0]):
                    print(f"    ERROR: cls_indices_for_text_embeddings fuera de rango para text_embeddings! Max índice: {targets_d['cls_indices_for_text_embeddings'].max()}, Shape embeds: {text_embeds_b.shape}")
        else:
            print("  targets_dict no es un diccionario.")
            
    print("--- FIN PRUEBA DE DATALOADER Y COLLATE FUNCTION ---")


def train_gui_heads(resume_from_checkpoint=False):
    if not all([ULTRALYTICS_AVAILABLE, MODEL_COMPONENTS_AVAILABLE]): 
        print("ERR: Faltan dependencias (Ultralytics YOLOE o Componentes del Modelo). Saliendo."); return
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Usando device para entrenamiento: {device}")
    
    os.makedirs(RUTA_BASE_GUARDADO_TAREA_ABS, exist_ok=True)

    print(f"Inicializando Text Encoder (usará YOLOE base desde {PATH_YOLOE_PT_BASE_ABS})...")
    if not initialize_text_encoder(device=device, yoloe_model_path_str=str(PATH_YOLOE_PT_BASE_ABS)):
        print("ERROR CRITICO: Fallo al inicializar el text encoder. Saliendo."); return
    
    current_embedding_dim = get_text_embedding_dim()
    if current_embedding_dim is None: # Doble chequeo
        print("ERROR CRITICO: EMBEDDING_DIM no se pudo determinar. Saliendo."); return
    print(f"Dimensión de embedding confirmada: {current_embedding_dim}")

    print(f"Cargando extractor YOLOE base: {PATH_YOLOE_PT_BASE_ABS}...");
    try: 
        yoloe_base_instance = YOLOE_Ultralytics(str(PATH_YOLOE_PT_BASE_ABS))
    except Exception as e: 
        print(f"CRITICO ERROR al cargar YOLOE base: {e}"); traceback.print_exc(); return
    
    if not (hasattr(yoloe_base_instance, 'model') and isinstance(yoloe_base_instance.model, torch.nn.Module)):
        print("CRITICO: yoloe_base_instance.model no es un nn.Module válido."); return
    yoloe_base_instance.model.to(device).eval() 
    print("Extractor YOLOE base cargado y en modo eval.")

    print(f"Instanciando ModeloYOLOE_MultiScale_PoC_Prompt (EmbDim={current_embedding_dim}, RegMax={REG_MAX_DFL})...")
    # Pasar la clase del extractor, no la instancia, si así lo espera el constructor del modelo PoC.
    # O pasar la instancia si el constructor de PoC espera la instancia.
    # En tu components_heads_prompt.py, ModeloYOLOE_MultiScale_PoC_Prompt espera la instancia.
    model = ModeloYOLOE_MultiScale_PoC_Prompt(
        yoloe_base_instance_for_extractor=yoloe_base_instance, 
        reg_max_default=REG_MAX_DFL, 
        embedding_dim_default=current_embedding_dim
    )
    model.to(device)
    print("ModeloYOLOE_MultiScale_PoC_Prompt instanciado y movido a device.")

    print("Preparando GuiPromptDataset para entrenamiento...");
    train_dataset = GuiPromptDataset(str(PATH_JSON_ANNOTATIONS_DIR_ABS), str(PATH_IMAGES_DIR_ABS), IMG_SIZE)
    if len(train_dataset) == 0: print(f"ERROR: Dataset de entrenamiento vacío."); return
    
    collate_fn_with_dev = lambda batch: collate_fn_gui_prompt(batch, device=device)
    train_dataloader = DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_dev, 
        num_workers=0, pin_memory=False, # pin_memory=True usualmente con num_workers > 0 y CUDA
        drop_last= len(train_dataset) >= BATCH_SIZE # Solo drop_last si hay más de 1 batch
    )
    print(f"Dataset de entrenamiento: {len(train_dataset)} muestras. DataLoader listo.")
    
    # (Opcional) Preparar DataLoader de Validación
    val_dataloader = None
    if PATH_VALID_JSON_ANNOTATIONS_DIR_ABS.is_dir() and PATH_VALID_IMAGES_DIR_ABS.is_dir():
        valid_dataset = GuiPromptDataset(str(PATH_VALID_JSON_ANNOTATIONS_DIR_ABS), str(PATH_VALID_IMAGES_DIR_ABS), IMG_SIZE)
        if len(valid_dataset) > 0:
            val_dataloader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn_with_dev, num_workers=0, pin_memory=False)
            print(f"Dataset de validación: {len(valid_dataset)} muestras.")
        else:
            print("INFO: Dataset de validación encontrado pero vacío.")
    else:
        print("INFO: No se encontraron directorios de validación. Se omitirá la validación.")


    print("Configurando Optimizador (AdamW) y Scheduler...");
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.detection_head.parameters()), 
                            lr=LEARNING_RATE, betas=(MOMENTUM_ADAMW_B1,0.999), weight_decay=WEIGHT_DECAY)
    
    effective_epochs_for_scheduler = max(1, EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_epochs_for_scheduler, 
                                                 eta_min=LEARNING_RATE*0.01 if effective_epochs_for_scheduler > 0 else LEARNING_RATE)
    print(f"Optimizador AdamW y Scheduler Cosine (T_max={effective_epochs_for_scheduler}) configurados.")

    print("Instanciando YOLOEPromptLoss (DUMMY para prueba de flujo)...");
    loss_fn = YOLOEPromptLoss(model, IMG_SIZE, # <--- AÑADIR IMG_SIZE AQUÍ
                          LOSS_BOX_WEIGHT, LOSS_DFL_WEIGHT, 
                          LOSS_OBJ_WEIGHT, LOSS_CLS_EMBED_WEIGHT, 
                          ASSIGNER_CFG)
    print("YOLOEPromptLoss (DUMMY) instanciada.")
    
    start_epoch=0; best_avg_loss=float('inf')
    # Lógica de reanudar checkpoint (simplificada por ahora)
    if resume_from_checkpoint and RUTA_CHECKPOINT_MULTIHEAD_LAST.exists():
        print(f"Intentando reanudar desde: {RUTA_CHECKPOINT_MULTIHEAD_LAST}")
        # ... (implementar carga de checkpoint) ...

    print(f"\n--- Iniciando Prueba de Flujo de Entrenamiento para '{NOMBRE_TAREA}' ({EPOCHS} épocas) ---")
    num_batches_train = len(train_dataloader)
    if num_batches_train == 0: print("ERROR: DataLoader de entrenamiento vacío. No se puede entrenar."); return
    
    warmup_total_iters = WARMUP_EPOCHS * num_batches_train if WARMUP_EPOCHS > 0 else 0

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss_sum = 0.0
        processed_batches_in_epoch = 0
        
        pbar = enumerate(train_dataloader)
        print(f"--- Época {epoch+1}/{EPOCHS} ---")
        
        for batch_idx, batch_content in pbar:
            if batch_content is None or batch_content[0] is None:
                print(f"  WARN (Epoch {epoch+1}): Batch {batch_idx+1}/{num_batches_train} inválido o vacío, skip."); continue
            
            images_batch, targets_dict_gpu, text_embeds_gpu = batch_content
            
            current_iter = batch_idx + num_batches_train * epoch
            if WARMUP_EPOCHS > 0 and current_iter < warmup_total_iters:
                lr_factor = float(current_iter + 1) / warmup_total_iters
                new_lr = (LEARNING_RATE - 1e-6) * lr_factor + 1e-6 # Interpolar desde casi 0
                for pg in optimizer.param_groups: pg['lr'] = new_lr
            
            # Mover al device ya se hace en collate_fn si se pasa el device
            # images_batch = images_batch.to(device)
            # targets_dict_gpu = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in targets_dict_from_collate.items()}
            # text_embeds_gpu = text_embeds_for_batch.to(device)

            if targets_dict_gpu['batch_idx'].numel() == 0 and images_batch.shape[0] > 0:
                # print(f"  INFO (Epoch {epoch+1}, Batch {batch_idx+1}): Batch sin targets, solo forward.")
                # Aún así, es bueno hacer un forward para asegurar que no crashee, pero no backward.
                 with torch.no_grad(): predictions_list = model(images_batch)
                 loss_total = torch.tensor(0.0, device=device)
                 loss_items = torch.zeros(4, device=device)
            else:
                optimizer.zero_grad()
                predictions_list = model(images_batch)
                loss_total, loss_items = loss_fn(predictions_list, targets_dict_gpu, text_embeds_gpu)

                if torch.isfinite(loss_total) and loss_total.item() > 1e-9 :
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, model.detection_head.parameters()), max_norm=10.0)
                    optimizer.step()
                elif isinstance(loss_total, torch.Tensor) and loss_total.item() <= 1e-9 and targets_dict_gpu['batch_idx'].numel() > 0 :
                     print(f"  WARN (Epoch {epoch+1}, Batch {batch_idx+1}): Ls CASI CERO ({loss_total.item():.2e}) CON TARGETS. No BWD/STEP.")
                elif not torch.isfinite(loss_total):
                     print(f"  WARN (Epoch {epoch+1}, Batch {batch_idx+1}): Pérdida NO FINITA ({loss_total.item()}). Skip BWD/STEP.")
            
            if isinstance(loss_total, torch.Tensor): epoch_loss_sum += loss_total.item()
            processed_batches_in_epoch += 1

            # Imprimir progreso
            if (batch_idx + 1) % max(1, num_batches_train // 5) == 0 or batch_idx == num_batches_train - 1:
                lrs_str = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
                loss_items_str = ", ".join([f"{it:.4f}" for it in loss_items.cpu().numpy()]) if isinstance(loss_items, torch.Tensor) else "N/A"
                print(f"  E{epoch+1}|B{batch_idx+1}/{num_batches_train}| Ls:{loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total:.4f} (Itms:[{loss_items_str}]) | LRs:[{','.join(lrs_str)}]")
        
        avg_epoch_loss = epoch_loss_sum / processed_batches_in_epoch if processed_batches_in_epoch > 0 else 0.0
        print(f"--- Fin Época {epoch+1}/{EPOCHS}. Avg Loss: {avg_epoch_loss:.4f} ---")
        
        if epoch >= WARMUP_EPOCHS -1 and effective_epochs_for_scheduler > 0: # Empezar a hacer step después de la última época de warmup
            scheduler.step()
        
        # Guardar Checkpoint (simplificado)
        if RUTA_CHECKPOINT_MULTIHEAD_LAST.parent.exists():
            ckpt_data = {'epoch': epoch, 'model_state_dict': model.state_dict()}
            torch.save(ckpt_data, RUTA_CHECKPOINT_MULTIHEAD_LAST)
            if avg_epoch_loss < best_avg_loss and avg_epoch_loss > 1e-7 : # Solo guardar best si la pérdida es significativa
                best_avg_loss = avg_epoch_loss
                shutil.copyfile(RUTA_CHECKPOINT_MULTIHEAD_LAST, RUTA_CHECKPOINT_MULTIHEAD_BEST)
                print(f"  Nueva mejor loss: {best_avg_loss:.4f}. Best ckpt actualizado.")
        else:
            print(f"WARN: Directorio de checkpoint no existe: {RUTA_CHECKPOINT_MULTIHEAD_LAST.parent}. No se guardó checkpoint.")


    print(f"\n--- Prueba de Flujo de Entrenamiento '{NOMBRE_TAREA}' Finalizada ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar/Probar Cabezas de Detección con Prompts para GUI")
    parser.add_argument('--resume', action='store_true', help='Reanudar entrenamiento desde el último checkpoint')
    parser.add_argument('--test_dataloader', action='store_true', help='Solo probar el DataLoader y CollateFn')
    args_script = parser.parse_args()

    if not all([ULTRALYTICS_AVAILABLE, MODEL_COMPONENTS_AVAILABLE]): 
        print("SALIDA __main__: Faltan dependencias críticas. Revisa los mensajes de error de importación."); exit()
    
    # Verificar paths antes de cualquier otra cosa
    paths_ok = True
    for p_name, p_val in [("YOLOE Base", PATH_YOLOE_PT_BASE_ABS), 
                          ("JSON Annotations (Train)", PATH_JSON_ANNOTATIONS_DIR_ABS),
                          ("Images (Train)", PATH_IMAGES_DIR_ABS)]:
        if (isinstance(p_val, pathlib.Path) and not p_val.exists()) or (isinstance(p_val, str) and not os.path.exists(p_val)):
            print(f"SALIDA __main__: ERROR - Ruta no encontrada para {p_name}: {p_val}"); paths_ok = False
    if not paths_ok: exit()
            
    # Determinar el device principal para la prueba/entrenamiento
    main_device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if args_script.test_dataloader:
        test_dataloader_and_collate(device_str=main_device_str)
    else:
        print("SALIDA __main__: Iniciando flujo de entrenamiento completo...")
        train_gui_heads(resume_from_checkpoint=args_script.resume)