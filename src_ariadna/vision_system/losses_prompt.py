import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import traceback

def make_anchors(feats_hw_shape: Tuple[int, int],
                 stride: int,
                 grid_cell_offset: float = 0.5,
                 device: Union[str, torch.device] = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    h_feat, w_feat = feats_hw_shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h_feat, device=device, dtype=torch.float32),
        torch.arange(w_feat, device=device, dtype=torch.float32),
        indexing='ij'
    )
    anchor_points_flat = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1) + grid_cell_offset
    stride_tensor_flat = torch.full((h_feat * w_feat, 1), float(stride), device=device, dtype=torch.float32)
    return anchor_points_flat, stride_tensor_flat

def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xyxy: bool = True) -> torch.Tensor:
    x1 = anchor_points[..., 0] - distance[..., 0]
    y1 = anchor_points[..., 1] - distance[..., 1]
    x2 = anchor_points[..., 0] + distance[..., 2]
    y2 = anchor_points[..., 1] + distance[..., 3]
    if xyxy:
        return torch.stack((x1, y1, x2, y2), dim=-1)
    else:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return torch.stack((cx, cy, w, h), dim=-1)

try:
    from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist as ultralytics_bbox2dist_tal
    from ultralytics.utils.ops import xywh2xyxy as ultralytics_xywh2xyxy_ops
    from ultralytics.utils.loss import DFLoss as UltralyticsDFLoss_real
    from ultralytics.utils.metrics import bbox_iou
    ULTRALYTICS_LOSS_COMPONENTS_AVAILABLE = True
    print("INFO (losses_prompt.py): Componentes de pérdida y TAL de Ultralytics importados.")
except ImportError as e:
    ULTRALYTICS_LOSS_COMPONENTS_AVAILABLE = False
    print(f"WARN (losses_prompt.py): No se pudieron importar componentes de Ultralytics: {e}.")
    print("  Usando versiones DUMMY para TaskAlignedAssigner, CIoULoss, DFLLoss, bbox2dist, xywh2xyxy.")

    class TaskAlignedAssigner:
        def __init__(self,topk,num_classes,alpha,beta): self.topk,self.num_classes,self.alpha,self.beta=topk,num_classes,alpha,beta; print("WARN: Usando TaskAlignedAssigner DUMMY.")
        def __call__(self,pd_scores,pd_bboxes,anc_points,gt_labels,gt_bboxes_img_scale,mask_gt):
            bs = pd_scores.shape[0]
            n_anc = pd_bboxes.shape[1]
            n_gt_total = gt_bboxes_img_scale.shape[1] if gt_bboxes_img_scale.ndim == 3 else 0


            target_labels_out=torch.full((bs,n_anc),0,dtype=torch.long,device=pd_scores.device)
            target_bboxes_iou_flat_list = []
            target_scores_obj_out=torch.zeros(bs,n_anc,device=pd_scores.device)
            fg_mask_out=torch.zeros(bs,n_anc,dtype=torch.bool,device=pd_scores.device)
            target_gt_idx_flat_pos_list = []
            
            num_pos_total_dummy=0
            current_gt_offset = 0 # Para simular el manejo de gt_idx globales en el dummy

            for b_i in range(bs):
                # Simular cuántos GTs hay para esta imagen (usando mask_gt si está disponible y es 3D)
                n_gts_this_img = gt_bboxes_img_scale.shape[1] if gt_bboxes_img_scale.ndim == 3 and mask_gt[b_i].any() else 0
                if gt_bboxes_img_scale.ndim == 2 and b_i == 0 : # Caso de gt concatenados y bs=1
                     n_gts_this_img = gt_bboxes_img_scale.shape[0]

                if n_gts_this_img > 0:
                    act_pos_this_img = min(self.topk, n_anc // 100 + 1, 3, n_gts_this_img)
                    if act_pos_this_img > 0:
                        fg_mask_out[b_i, :act_pos_this_img] = True
                        
                        # Simular asignación de etiquetas y bboxes
                        if gt_labels.ndim == 3 and gt_labels.shape[0] == bs:
                             target_labels_out[b_i, :act_pos_this_img] = gt_labels[b_i, :act_pos_this_img, 0]
                        elif gt_labels.ndim == 1 and b_i == 0: # GTs concatenados, bs=1
                             target_labels_out[b_i, :act_pos_this_img] = gt_labels[current_gt_offset : current_gt_offset + act_pos_this_img]


                        if gt_bboxes_img_scale.ndim == 3 and gt_bboxes_img_scale.shape[0] == bs:
                            target_bboxes_iou_flat_list.append(gt_bboxes_img_scale[b_i, :act_pos_this_img])
                        elif gt_bboxes_img_scale.ndim == 2 and b_i == 0: # GTs concatenados, bs=1
                            target_bboxes_iou_flat_list.append(gt_bboxes_img_scale[current_gt_offset : current_gt_offset + act_pos_this_img])


                        target_scores_obj_out[b_i, :act_pos_this_img] = 1.0
                        
                        # Simular target_gt_idx (índices dentro de los GTs de *esta* imagen)
                        # Para el dummy, podemos simplemente usar un arange, ya que no hay un mapeo real
                        # Si los GTs originales fueran concatenados, estos serían índices globales
                        target_gt_idx_flat_pos_list.append(torch.arange(current_gt_offset, current_gt_offset + act_pos_this_img, device=pd_scores.device))
                        num_pos_total_dummy += act_pos_this_img
                current_gt_offset += n_gts_this_img # Actualizar offset para el caso de GTs concatenados

            target_bboxes_iou_flat = torch.cat(target_bboxes_iou_flat_list, dim=0) if target_bboxes_iou_flat_list else torch.empty((0,4),device=pd_bboxes.device)
            target_gt_idx_flat_pos = torch.cat(target_gt_idx_flat_pos_list) if target_gt_idx_flat_pos_list else torch.empty((0,),dtype=torch.long,device=pd_scores.device)
            
            # El assigner real devuelve target_gt_idx como (bs, n_anc), no aplanado.
            # Para el dummy, vamos a simularlo llenando con -1 y luego poniendo los índices
            # Esto es solo para que la forma de salida sea más parecida, la lógica interna del dummy es simple.
            target_gt_idx_out = torch.full((bs, n_anc), -1, dtype=torch.long, device=pd_scores.device)
            if num_pos_total_dummy > 0:
                 # Esta parte es una simplificación grosera para el dummy
                 # No intenta replicar la lógica de asignación real para target_gt_idx_out
                 # Solo asegura que las anclas positivas tengan algún índice GT válido (0 a n_gts_this_img-1)
                 # El target_gt_idx_flat_pos es más representativo de lo que se usaría para la pérdida de embedding.
                 temp_offset = 0
                 for b_idx_iter in range(bs):
                    num_pos_this_img_iter = fg_mask_out[b_idx_iter].sum().item()
                    if num_pos_this_img_iter > 0:
                        # Asignar índices de GT (0 a num_gts_this_img-1) a las anclas positivas de esta imagen
                        # Esto es una simplificación, el asignador real hace un mapeo más complejo
                        # y target_gt_idx_flat_pos ya contiene los índices correctos para las anclas positivas
                        # en el formato concatenado.
                        # Para el formato (bs, n_anc) de target_gt_idx_out, necesitamos los índices *dentro de cada imagen*.
                        # El `target_gt_idx_flat_pos` ya tiene los índices correctos si los GTs originales eran concatenados.
                        # Si los GTs eran (bs, max_gts, ...), entonces target_gt_idx_flat_pos no es lo que necesitamos aquí.
                        # El assigner real de Ultralytics devuelve target_gt_idx (bs, n_anc) con índices 0..max_gts-1.
                        # Vamos a simularlo con un arange simple para las posiciones fg.
                        target_gt_idx_out[b_idx_iter, fg_mask_out[b_idx_iter]] = torch.arange(num_pos_this_img_iter, device=pd_scores.device) % (gt_bboxes_img_scale.shape[1] if gt_bboxes_img_scale.ndim == 3 else gt_bboxes_img_scale.shape[0] if b_idx_iter==0 else 1)


            return target_labels_out, target_bboxes_iou_flat, target_scores_obj_out, fg_mask_out, target_gt_idx_out


    class UltralyticsCIoULoss_real(nn.Module):
        def __init__(self,reduction='sum'):super().__init__();self.reduction=reduction;print("WARN: CIoULoss DUMMY");
        def forward(self,p,t): return torch.abs(p-t).sum() if p.numel()>0 and t.numel()>0 and p.shape==t.shape else torch.tensor(0.0,device=p.device)

    class UltralyticsDFLoss_real(nn.Module):
        def __init__(self,reg_max=16):super().__init__();self.reg_max=reg_max;print("WARN: UltralyticsDFLLoss(BFLoss) DUMMY");
        def forward(self,p,t_idx): return torch.tensor(0.0,device=p.device) if p.numel()==0 else F.cross_entropy(p.view(-1,self.reg_max),t_idx.view(-1).long(),reduction='mean')*0.01

    def ultralytics_bbox2dist_tal(anc_pts, t_bboxes, reg_max): print("WARN: bbox2dist DUMMY"); return torch.randint(0,reg_max,(t_bboxes.shape[0],4),device=t_bboxes.device,dtype=torch.long) if t_bboxes.numel()>0 else torch.empty((0,4),device=t_bboxes.device,dtype=torch.long)
    def ultralytics_xywh2xyxy_ops(x): y=x.clone();y[...,0]=x[...,0]-x[...,2]/2;y[...,1]=x[...,1]-x[...,3]/2;y[...,2]=x[...,0]+x[...,2]/2;y[...,3]=x[...,1]+x[...,3]/2;return y

if ULTRALYTICS_LOSS_COMPONENTS_AVAILABLE:
    class CIoULossReal(nn.Module):
        def __init__(self, reduction='sum'):
            super().__init__()
            self.reduction = reduction
            assert reduction in ['sum', 'mean', 'none'], "reduction must be 'sum', 'mean' or 'none'"

        def forward(self, pred_bboxes_xyxy, target_bboxes_xyxy):
            if pred_bboxes_xyxy.numel() == 0 or target_bboxes_xyxy.numel() == 0:
                return torch.tensor(0.0, device=pred_bboxes_xyxy.device)
            
            iou = bbox_iou(pred_bboxes_xyxy, target_bboxes_xyxy, xywh=False, CIoU=True)
            loss_ciou = 1.0 - iou

            if self.reduction == 'sum':
                return loss_ciou.sum()
            elif self.reduction == 'mean':
                return loss_ciou.mean()
            return loss_ciou
else:
    CIoULossReal = UltralyticsCIoULoss_real

class YOLOEPromptLoss(nn.Module):
    def __init__(self, model_ref, img_size: int,
                 box_w: float, dfl_w: float, obj_w: float, cls_embed_w: float,
                 assigner_cfg: dict = None):
        super().__init__()
        self.model_ref = model_ref
        self.img_size = img_size
        self.box_w, self.dfl_w, self.obj_w, self.cls_embed_w = box_w, dfl_w, obj_w, cls_embed_w
        self.reg_max = model_ref.reg_max
        self.embedding_dim = model_ref.embedding_dim
        self.strides = model_ref.stride.tolist()
        self.num_levels = len(self.strides)
        default_assigner_cfg = {'topk': 10, 'alpha': 0.5, 'beta': 6.0, 'num_classes': 1} # num_classes=1 para objectness
        self.assigner_config = assigner_cfg if assigner_cfg is not None else default_assigner_cfg
        print(f"INICIALIZANDO YOLOEPromptLoss con img_size={img_size}, pesos: box={box_w}, dfl={dfl_w}, obj={obj_w}, cls_embed={cls_embed_w}")

        if ULTRALYTICS_LOSS_COMPONENTS_AVAILABLE:
            print("INFO (YOLOEPromptLoss): Usando TaskAlignedAssigner y Losses REALES de Ultralytics.")
            self.assigner = TaskAlignedAssigner(topk=self.assigner_config['topk'], num_classes=self.assigner_config['num_classes'], alpha=self.assigner_config['alpha'], beta=self.assigner_config['beta'])
            self.bce_obj = nn.BCEWithLogitsLoss(reduction='sum')
            self.cls_embed_loss_fn = nn.CrossEntropyLoss(reduction='sum')
            self.iou_loss_fn = CIoULossReal(reduction='sum')
            self.dfl_loss_fn = UltralyticsDFLoss_real(reg_max=self.reg_max)
        else:
            print("WARN (YOLOEPromptLoss): Usando DUMMY Assigner y Losses.")
            self.assigner = TaskAlignedAssigner(topk=self.assigner_config['topk'], num_classes=self.assigner_config['num_classes'], alpha=self.assigner_config['alpha'], beta=self.assigner_config['beta'])
            self.bce_obj = nn.BCEWithLogitsLoss(reduction='sum')
            self.cls_embed_loss_fn = nn.CrossEntropyLoss(reduction='sum')
            self.iou_loss_fn = UltralyticsCIoULoss_real(reduction='sum')
            self.dfl_loss_fn = UltralyticsDFLoss_real(reg_max=self.reg_max)

    def forward(self, preds_list_from_model: List[torch.Tensor],
                targets_dict: dict,
                text_embeddings_for_batch_classes: torch.Tensor):
        device = preds_list_from_model[0].device
        batch_size = preds_list_from_model[0].shape[0]
        num_dfl_ch = 4 * self.reg_max
        num_obj_ch = 1
        pd_bboxes_xyxy_list, pd_obj_logits_list, pd_visual_embeds_list, pd_dfl_raw_list = [], [], [], []
        anchor_points_cat_list, stride_tensor_cat_list = [], []

        for i_lvl in range(self.num_levels):
            pred_raw = preds_list_from_model[i_lvl]
            strd = self.strides[i_lvl]
            hf, wf = pred_raw.shape[2:4]
            n_anc = hf * wf
            pred_flat = pred_raw.permute(0, 2, 3, 1).contiguous().view(batch_size, n_anc, -1)
            dflr = pred_flat[..., :num_dfl_ch]
            objl = pred_flat[..., num_dfl_ch:num_dfl_ch + num_obj_ch]
            visl = pred_flat[..., num_dfl_ch + num_obj_ch:]
            anc_feat, _ = make_anchors((hf, wf), int(strd), 0.5, device)
            dfl_rs = dflr.view(batch_size, n_anc, 4, self.reg_max)
            dfl_sm = F.softmax(dfl_rs, dim=-1)
            integ_v = torch.arange(self.reg_max, device=device, dtype=torch.float32)
            bbox_dist_f = dfl_sm @ integ_v
            anc_img = anc_feat * strd
            bbox_dist_i = bbox_dist_f * strd
            pred_bboxes_lvl = dist2bbox(bbox_dist_i, anc_img.unsqueeze(0).expand(batch_size, -1, -1), xyxy=True)
            pd_bboxes_xyxy_list.append(pred_bboxes_lvl)
            pd_obj_logits_list.append(objl)
            pd_visual_embeds_list.append(visl)
            pd_dfl_raw_list.append(dflr)
            anchor_points_cat_list.append(anc_img)
            stride_tensor_cat_list.append(torch.full((n_anc, 1), float(strd), device=device, dtype=torch.float32))

        pd_bboxes_all = torch.cat(pd_bboxes_xyxy_list, dim=1)
        pd_obj_logits_all = torch.cat(pd_obj_logits_list, dim=1)
        pd_visual_embeds_all = torch.cat(pd_visual_embeds_list, dim=1)
        pd_dfl_raw_all = torch.cat(pd_dfl_raw_list, dim=1)
        anc_points_for_assigner = torch.cat(anchor_points_cat_list, dim=0)
        stride_tensor_for_assigner = torch.cat(stride_tensor_cat_list, dim=0)

        gt_bboxes_cxcywh_norm = targets_dict['bboxes_gt'] # (N_total_gt, 4)
        batch_idx_gt = targets_dict['batch_idx'] # (N_total_gt,)
        
        if ULTRALYTICS_LOSS_COMPONENTS_AVAILABLE:
            gt_bboxes_xyxy_norm = ultralytics_xywh2xyxy_ops(gt_bboxes_cxcywh_norm)
        else:
            cx, cy, w, h = gt_bboxes_cxcywh_norm.t()
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            gt_bboxes_xyxy_norm = torch.stack((x1, y1, x2, y2), dim=1)
        
        gt_bboxes_for_assigner_scaled = gt_bboxes_xyxy_norm * self.img_size # (N_total_gt, 4)

        # Preparar GTs para el TaskAlignedAssigner de Ultralytics
        if batch_idx_gt.numel() > 0:
            counts_per_image = torch.bincount(batch_idx_gt, minlength=batch_size)
            max_gts_per_image = counts_per_image.max().item()
        else:
            max_gts_per_image = 0

        gt_labels_padded = torch.zeros((batch_size, max_gts_per_image, 1), device=device, dtype=torch.long)
        gt_bboxes_padded = torch.zeros((batch_size, max_gts_per_image, 4), device=device, dtype=gt_bboxes_for_assigner_scaled.dtype)
        mask_gt_padded = torch.zeros((batch_size, max_gts_per_image, 1), device=device, dtype=torch.bool)
        cls_indices_padded = torch.full((batch_size, max_gts_per_image), -1, device=device, dtype=torch.long)

        if max_gts_per_image > 0:
            for b_idx_loop in range(batch_size):
                is_this_image = (batch_idx_gt == b_idx_loop)
                n_gts_this_image = is_this_image.sum().item()
                if n_gts_this_image > 0:
                    gt_bboxes_padded[b_idx_loop, :n_gts_this_image] = gt_bboxes_for_assigner_scaled[is_this_image]
                    mask_gt_padded[b_idx_loop, :n_gts_this_image] = True
                    cls_indices_padded[b_idx_loop, :n_gts_this_image] = targets_dict['cls_indices_for_text_embeddings'][is_this_image]
        
        pd_obj_scores_for_assigner = pd_obj_logits_all.sigmoid().detach()  # (bs, N_anc_total, 1)

        # Llamada al asignador con GTs acolchados
        # target_labels_cls_assigner: (bs, N_anc_total)
        # target_bboxes_iou: (num_total_pos_anchors_in_batch, 4) - bboxes GT asignadas a anclas positivas
        # target_scores_obj: (bs, N_anc_total, 1) - scores de objectness GT para cada ancla
        # fg_mask: (bs, N_anc_total) - máscara booleana de anclas positivas
        # target_gt_idx_assigner: (bs, N_anc_total) - índice (0 a max_gts-1) del GT asignado a cada ancla positiva
        _, target_bboxes_iou, target_scores_obj, fg_mask, target_gt_idx_assigner = \
            self.assigner(
                pd_scores=pd_obj_scores_for_assigner,
                pd_bboxes=pd_bboxes_all.detach(),
                anc_points=anc_points_for_assigner, # El asignador lo maneja por imagen
                gt_labels=gt_labels_padded,
                gt_bboxes=gt_bboxes_padded,
                mask_gt=mask_gt_padded
            )
        num_pos = fg_mask.sum().item()

        if not hasattr(YOLOEPromptLoss, 'assigner_print_count_debug'):
            YOLOEPromptLoss.assigner_print_count_debug = 0
        if YOLOEPromptLoss.assigner_print_count_debug < 1 and batch_size > 0:
            print(f"  YOLOEPromptLoss DEBUG - Assigner Output (num_pos={num_pos}):")
            print(f"    fg_mask shape: {fg_mask.shape}, sum: {fg_mask.sum().item()}")
            print(f"    target_gt_idx_assigner shape: {target_gt_idx_assigner.shape}")
            print(f"    target_bboxes_iou shape: {target_bboxes_iou.shape}")
            print(f"    target_scores_obj shape: {target_scores_obj.shape}")
            if num_pos > 0:
                # Para imprimir un ejemplo, necesitamos el batch_idx de las anclas positivas
                # y los gt_idx dentro de la imagen para esas anclas.
                # fg_mask[0].nonzero() daría los índices de anclas positivas en la primera imagen.
                # target_gt_idx_assigner[0, fg_mask[0].nonzero().squeeze()] daría los gt_idx para esas anclas.
                # target_bboxes_iou son los GTs ya filtrados para las anclas positivas.
                print(f"      Ej. target_bboxes_iou[:5]: {target_bboxes_iou[:min(5, num_pos)]}")
            YOLOEPromptLoss.assigner_print_count_debug += 1

        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        loss_cls_embed = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)

        if pd_obj_logits_all.numel() > 0 and target_scores_obj.numel() > 0:
            loss_obj = self.bce_obj(pd_obj_logits_all.squeeze(-1), target_scores_obj.squeeze(-1))

        if num_pos > 0:
            pred_bboxes_pos = pd_bboxes_all[fg_mask]  # (num_pos, 4)
            pred_dfl_raw_pos = pd_dfl_raw_all[fg_mask]  # (num_pos, 4 * reg_max)
            pred_visual_embeds_pos_unfiltered = pd_visual_embeds_all[fg_mask]  # (num_pos, emb_dim)
            target_bboxes_pos = target_bboxes_iou[fg_mask]  # (num_pos, 4)

            if pred_bboxes_pos.numel() > 0 and target_bboxes_pos.numel() > 0:
                loss_box = self.iou_loss_fn(pred_bboxes_pos, target_bboxes_pos)

            if pred_dfl_raw_pos.numel() > 0 and target_bboxes_pos.numel() > 0:
                anc_points_pos = anc_points_for_assigner[fg_mask.view(-1)] # (num_pos, 2)
                stride_tensor_pos = stride_tensor_for_assigner[fg_mask.view(-1)] # (num_pos, 1)

                target_dist_pixels = torch.empty_like(target_bboxes_pos)  # (num_pos, 4)
                if ULTRALYTICS_LOSS_COMPONENTS_AVAILABLE:
                    try:
                        # bbox2dist espera anchor_points (escala imagen), target_bboxes (escala imagen), reg_max-1
                        target_dist_pixels = ultralytics_bbox2dist_tal(anc_points_pos, target_bboxes_pos, self.reg_max -1)
                    except Exception as e_b2d:
                        print(f"WARN: Error en ultralytics_bbox2dist_tal: {e_b2d}, usando DUMMY DFL target.")
                        target_dist_pixels = torch.randint(0, self.reg_max, (num_pos, 4), device=device, dtype=target_bboxes_pos.dtype)
                else:
                     target_dist_pixels = torch.randint(0, self.reg_max, (num_pos, 4), device=device, dtype=target_bboxes_pos.dtype)

                # Los targets para DFL deben estar en la escala de la feature map (divididos por el stride)
                target_dfl_indices_feat_scale = target_dist_pixels / stride_tensor_pos

                # DFLoss espera pred_dist (N, reg_max) y target (N)
                loss_dfl = self.dfl_loss_fn(pred_dfl_raw_pos.view(-1, self.reg_max), target_dfl_indices_feat_scale.view(-1))
                # dfl_loss_fn ya promedia, así que loss_dfl es escalar.

            # Pérdida de Embedding
            batch_idx_pos = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(fg_mask)[fg_mask]
            gt_idx_in_image_pos = target_gt_idx_assigner[fg_mask]
            gt_cls_indices_for_embed_loss_unfiltered = cls_indices_padded[batch_idx_pos, gt_idx_in_image_pos]
            
            valid_embedding_targets_mask = gt_cls_indices_for_embed_loss_unfiltered != -1
            gt_cls_indices_for_embed_loss = gt_cls_indices_for_embed_loss_unfiltered[valid_embedding_targets_mask]
            pred_visual_embeds_pos_filtered = pred_visual_embeds_pos_unfiltered[valid_embedding_targets_mask]

            if pred_visual_embeds_pos_filtered.numel() > 0 and text_embeddings_for_batch_classes.numel() > 0 and gt_cls_indices_for_embed_loss.numel() > 0:
                similarity_logits = F.normalize(pred_visual_embeds_pos_filtered) @ F.normalize(text_embeddings_for_batch_classes).T
                loss_cls_embed = self.cls_embed_loss_fn(similarity_logits, gt_cls_indices_for_embed_loss)
            else:
                loss_cls_embed = torch.tensor(0.0, device=device)
        
        num_total_anchors_batch = batch_size * (pd_obj_logits_all.shape[1] if pd_obj_logits_all.ndim > 1 and pd_obj_logits_all.shape[1] > 0 else 1)

        loss_obj_final = loss_obj / num_total_anchors_batch if num_total_anchors_batch > 0 else torch.tensor(0.0, device=device)
        loss_box_final = loss_box / num_pos if num_pos > 0 else torch.tensor(0.0, device=device)
        loss_dfl_final = loss_dfl # Ya promediado por DFLoss de Ultralytics
        loss_cls_embed_final = loss_cls_embed / pred_visual_embeds_pos_filtered.shape[0] if pred_visual_embeds_pos_filtered.numel() > 0 else torch.tensor(0.0, device=device)


        loss_total = (loss_box_final * self.box_w +
                      loss_dfl_final * self.dfl_w +
                      loss_obj_final * self.obj_w +
                      loss_cls_embed_final * self.cls_embed_w)
        loss_items = torch.tensor([
            loss_box_final.item(),
            loss_dfl_final.item(),
            loss_obj_final.item(),
            loss_cls_embed_final.item(),
        ], device=device)

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            print("ERROR: Pérdida NaN/Inf")
            loss_total = torch.zeros((), device=device, requires_grad=True)
            loss_items = torch.zeros(4, device=device)
        if loss_total.numel() > 0 and loss_total.item() > 1e-9 and not loss_total.requires_grad :
            loss_total = loss_total.clone().requires_grad_(True)
        return loss_total, loss_items