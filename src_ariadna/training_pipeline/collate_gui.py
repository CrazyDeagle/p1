# src_ariadna/training_pipeline/collate_gui.py
from typing import Optional
import torch
# Ajusta la importación para que funcione desde aquí, asumiendo que este archivo
# está en training_pipeline y text_encoder_utils está en vision_system
try:
    from ..vision_system.text_encoder_utils import get_text_embeddings_batch, get_text_embedding_dim 
except ImportError:
    print("ERROR (collate_gui.py): No se pudo importar desde vision_system.text_encoder_utils.")
    print("  Asegúrate de que __init__.py exista en vision_system y que las rutas sean correctas.")
    # Fallbacks para que el script no crashee al importar, pero el entrenamiento fallará
    def get_text_embeddings_batch(prompt_list, device): return torch.empty(0)
    def get_text_embedding_dim(): return 512


def collate_fn_gui_prompt(batch, device: str = 'cpu') -> tuple[Optional[torch.Tensor], Optional[dict], Optional[torch.Tensor]]:
    valid_items = [item for item in batch if item[0] is not None and item[1] is not None and item[2] is not None and item[3] is not None]
    
    if not valid_items: 
        if not hasattr(collate_fn_gui_prompt, 'empty_batch_count_debug'): collate_fn_gui_prompt.empty_batch_count_debug = 0
        collate_fn_gui_prompt.empty_batch_count_debug += 1
        if collate_fn_gui_prompt.empty_batch_count_debug < 3: print("COLLATE WARN: Batch vacío o con Nones post-filtro.")
        return None, None, None 
    
    images_list, prompts_for_each_image_list, bboxes_for_each_image_list, objectness_for_each_image_list = zip(*valid_items)
    
    try:
        images_batch_tensor = torch.stack(images_list, 0)
    except RuntimeError as e_stack: 
        print(f"COLLATE ERROR: apilando imágenes: {e_stack}")
        for i_img, img_t in enumerate(images_list): print(f" Img {i_img} shape: {img_t.shape if isinstance(img_t, torch.Tensor) else 'No Tensor'}")
        return None, None, None

    all_prompts_in_batch_ordered = [] 
    unique_prompts_map = {} 
    unique_prompts_list = [] 
    
    current_unique_idx = 0
    for prompts_this_image in prompts_for_each_image_list:
        for prompt_str in prompts_this_image:
            all_prompts_in_batch_ordered.append(prompt_str)
            if prompt_str not in unique_prompts_map:
                unique_prompts_map[prompt_str] = current_unique_idx
                unique_prompts_list.append(prompt_str)
                current_unique_idx += 1
    
    # Generar embeddings para los prompts únicos del batch
    # get_text_embeddings_batch debería devolver [NumUniquePrompts, EmbDim] o [1, NumUniquePrompts, EmbDim]
    text_embeddings_for_unique_prompts = get_text_embeddings_batch(unique_prompts_list, target_device=device) if unique_prompts_list else torch.empty((0, get_text_embedding_dim()), device=device)

    # Si get_text_embeddings_batch devuelve [1, N, D], quitar la primera dimensión si N > 0
    if text_embeddings_for_unique_prompts.ndim == 3 and text_embeddings_for_unique_prompts.shape[0] == 1 and text_embeddings_for_unique_prompts.shape[1] > 0:
        text_embeddings_for_unique_prompts = text_embeddings_for_unique_prompts.squeeze(0)
    elif text_embeddings_for_unique_prompts.ndim == 3 and text_embeddings_for_unique_prompts.shape[0] > 1 :
         print(f"COLLATE WARN: text_embeddings_for_unique_prompts tiene un batch_dim > 1 ({text_embeddings_for_unique_prompts.shape}), inesperado.")


    cls_indices_for_text_embeddings = []
    if all_prompts_in_batch_ordered:
        try:
            cls_indices_for_text_embeddings = [unique_prompts_map[p_str] for p_str in all_prompts_in_batch_ordered]
        except KeyError as e_key:
            print(f"COLLATE ERROR: KeyError al mapear prompt a índice: {e_key}. Prompts: {all_prompts_in_batch_ordered}, Mapa: {unique_prompts_map}"); return None, None, None

    target_batch_indices = []
    target_bboxes_gt_list = []
    target_objectness_gt_list = []
    valid_objects_in_batch = 0

    for i_img, bboxes_this_image in enumerate(bboxes_for_each_image_list):
        num_objects_this_image = bboxes_this_image.shape[0] if isinstance(bboxes_this_image, torch.Tensor) else 0
        if num_objects_this_image > 0:
            target_batch_indices.extend([i_img] * num_objects_this_image)
            target_bboxes_gt_list.append(bboxes_this_image)
            target_objectness_gt_list.append(objectness_for_each_image_list[i_img])
            valid_objects_in_batch += num_objects_this_image

    targets_dict = {}
    emb_dim_for_empty = get_text_embedding_dim() # Asegurar que se llame una vez si es necesario
    if valid_objects_in_batch > 0:
        targets_dict['batch_idx'] = torch.tensor(target_batch_indices, dtype=torch.long, device=device)
        targets_dict['bboxes_gt'] = torch.cat(target_bboxes_gt_list, dim=0).to(device)
        targets_dict['objectness_gt'] = torch.cat(target_objectness_gt_list, dim=0).view(-1, 1).to(device)
        targets_dict['cls_indices_for_text_embeddings'] = torch.tensor(cls_indices_for_text_embeddings, dtype=torch.long, device=device)
    else: 
        targets_dict['batch_idx'] = torch.empty((0), dtype=torch.long, device=device)
        targets_dict['bboxes_gt'] = torch.empty((0, 4), dtype=torch.float32, device=device)
        targets_dict['objectness_gt'] = torch.empty((0, 1), dtype=torch.float32, device=device)
        targets_dict['cls_indices_for_text_embeddings'] = torch.empty((0), dtype=torch.long, device=device)
        text_embeddings_for_unique_prompts = torch.empty((0, emb_dim_for_empty), device=device) # Asegurar que esté vacío si no hay targets

    # Mover todos los tensores del targets_dict al device final
    for k, v_tensor in targets_dict.items():
        if isinstance(v_tensor, torch.Tensor) and v_tensor.device.type != device:
            targets_dict[k] = v_tensor.to(device)
            
    return images_batch_tensor.to(device), targets_dict, text_embeddings_for_unique_prompts.to(device)