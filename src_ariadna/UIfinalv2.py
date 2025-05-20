#!/usr/bin/env python
# SSGnano.py (jarvis_ui_helper_v6.42_full_explicit_definitions.py)
# -------------------------------------------------------------------------
# • VERSIÓN COMPLETA CON TODAS LAS DEFINICIONES DE ImageTab EXPLÍCITAS.
# • Integra todas las funcionalidades discutidas.
# -------------------------------------------------------------------------
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import json
import pathlib
import os
import re
import base64
import io
from typing import List, Tuple, Any, Dict, Callable, Optional 
import numpy as np
import cv2
import ttkbootstrap as tb
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, font as tkFont, filedialog, simpledialog
import threading
import queue
import math
import shutil
import yaml 
from collections import defaultdict
import traceback 
import time
import sys
import os
import pathlib
import copy
import sys
import os
import pathlib

# Obtener la ruta al directorio donde está UIfinalv2.py
SCRIPT_DIR_UI = pathlib.Path(__file__).parent.resolve()

# Construir la ruta a la carpeta 'yoloe_official_code'
YOLOE_OFFICIAL_CODE_ROOT = SCRIPT_DIR_UI / "yoloe_official_code" 

# Añadir la carpeta que contiene el paquete 'ultralytics' de THU-MIG al inicio de sys.path
# Esto hará que 'import ultralytics.data.augment' busque primero dentro de YOLOE_OFFICIAL_CODE_ROOT
if os.path.isdir(YOLOE_OFFICIAL_CODE_ROOT):
    if str(YOLOE_OFFICIAL_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLOE_OFFICIAL_CODE_ROOT))
        print(f"INFO: Añadido al sys.path para priorizar módulos: {YOLOE_OFFICIAL_CODE_ROOT}")
else:
    print(f"WARN: Carpeta 'yoloe_official_code' no encontrada en {SCRIPT_DIR_UI}. Las importaciones podrían usar la librería instalada.")

# Ahora las importaciones deberían tomar los módulos de yoloe_official_code si existen allí
try:
    from yoloe_official_code.ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor, YOLOEVPDetectPredictor
    # Esta importación ahora buscará en YOLOE_OFFICIAL_CODE_ROOT/ultralytics/models/yolo/yoloe/predict_vp.py
    
    # Y predict_vp.py, cuando haga:
    # from ultralytics.data.augment import LetterBox, LoadVisualPrompt
    # también buscará primero en YOLOE_OFFICIAL_CODE_ROOT/ultralytics/data/augment.py
    
    YOLOVP_PREDICTORS_AVAILABLE = True
    print("DEBUG: Predictors YOLOE VP importados (esperemos desde yoloe_official_code).")
except ImportError as e_vp_import:
    YOLOVP_PREDICTORS_AVAILABLE = False
    YOLOEVPSegPredictor = None 
    YOLOEVPDetectPredictor = None
    print(f"WARN: No se pudieron importar los predictores de YOLOE VP desde yoloe_official_code: {e_vp_import}")
    print(f"       Asegúrate de que 'predict_vp.py' y 'augment.py' de YOLOE (THU-MIG) estén en la estructura correcta dentro de 'yoloe_official_code'.")
    print(f"       PYTHONPATH actual (primeros elementos): {sys.path[:5]}")
    traceback.print_exc()

def calculate_iou(box1_xyxy: List[int], box2_xyxy: List[int]) -> float:
    x1_i = max(box1_xyxy[0], box2_xyxy[0])
    y1_i = max(box1_xyxy[1], box2_xyxy[1])
    x2_i = min(box1_xyxy[2], box2_xyxy[2])
    y2_i = min(box1_xyxy[3], box2_xyxy[3])

    inter_width = max(0, x2_i - x1_i)
    inter_height = max(0, y2_i - y1_i)
    inter_area = inter_width * inter_height

    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
YOLOE_OFFICIAL_CODE_ROOT = SCRIPT_DIR / "yoloe_official_code" 


OFFICIAL_ULTRALYTICS_PATH = YOLOE_OFFICIAL_CODE_ROOT / "ultralytics" 

if str(YOLOE_OFFICIAL_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOE_OFFICIAL_CODE_ROOT))
    print(f"DEBUG: Añadido al sys.path: {YOLOE_OFFICIAL_CODE_ROOT}")

try:
    from yoloe_official_code.ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor, YOLOEVPDetectPredictor
    print("DEBUG: Importado predict_vp desde la ruta ajustada (esperemos).")
except ImportError as e:
    print(f"ERROR: No se pudo importar predict_vp después de ajustar sys.path: {e}")
    print("       Asegúrate de que predict_vp.py y sus dependencias (como augment.py)")
    print(f"       estén en la estructura correcta dentro de: {YOLOE_OFFICIAL_CODE_ROOT}")
    YOLOEVPSegPredictor = None 
try:
    from ultralytics import YOLO
    from ultralytics.engine.predictor import BasePredictor 
    from ultralytics.models.yolo.model import YOLO as YOLOModelClass
    try:
        from predict_vp import YOLOEVPSegPredictor, YOLOEVPDetectPredictor 
        YOLOVP_PREDICTORS_AVAILABLE = True
        print("DEBUG: Predictors YOLOE VP importados desde predict_vp.py.")
    except ImportError as e_vp_import:
        YOLOVP_PREDICTORS_AVAILABLE = False
        YOLOEVPSegPredictor = None 
        YOLOEVPDetectPredictor = None
        print(f"WARN: No se pudieron importar los predictores de YOLOE VP desde predict_vp.py: {e_vp_import}")
        print("       La funcionalidad de 'Sugerir con YOLOE (Visual)' NO ESTARÁ DISPONIBLE.")


    import torch
    import torchvision.transforms as T
    ULTRALYTICS_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo para Modelos de Visión: {DEVICE}")
    if DEVICE == 'cuda':
        try:
            print(f"  Nombre GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"  WARN: No se pudo obtener nombre de GPU: {e}")
            DEVICE = 'cpu'
            print(f"  WARN: Cambiando a dispositivo CPU.")
    YOLO_CONF = 0.25 
    YOLO_IMGSZ = 512
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLOVP_PREDICTORS_AVAILABLE = False 
    YOLOEVPSegPredictor = None
    YOLOEVPDetectPredictor = None  
    DEVICE = 'cpu'
    YOLO_CONF = 0.25
    YOLO_IMGSZ = 512
    YOLO = None
    T = None 
    print("ERROR: Librerías 'ultralytics' o 'torch' no instaladas. Funcionalidad Pipeline YOLO, AutoTrain y YOLOE Visual Prompting desactivadas.")

try:
    import requests
    REQUESTS_AVAILABLE = True
    print("DEBUG: Librería 'requests' importada correctamente.")
except ImportError:
    REQUESTS_AVAILABLE = False
    print("ERROR: La librería 'requests' no está instalada. Funcionalidad Gemini desactivada.")

try:
    import faiss 
    import open_clip 
    OPENCLIP_AVAILABLE = True
except ImportError:
     print("ERROR: Librerías 'faiss' u 'open_clip_torch' no instaladas. Memoria Visual desactivada.")
     OPENCLIP_AVAILABLE = False
     MemoryManager = None

try:
    from groundingdino.util.inference import load_model as gd_load_model
    import supervision as sv
    GROUNDINGDINO_AVAILABLE = True
    print("DEBUG: Librería 'groundingdino' y 'supervision' importadas correctamente.")
except ImportError:
    GROUNDINGDINO_AVAILABLE = False
    sv = None
    print("ERROR: Librerías 'groundingdino-py' o 'supervision' no instaladas o no se pudieron importar. Funcionalidad Grounding DINO desactivada.")

DEFAULT_LABEL_FALLBACK = "objeto_desconocido" 
def normalize_label_fallback(label_text: str) -> str:
    if not isinstance(label_text, str): return DEFAULT_LABEL_FALLBACK
    normalized = re.sub(r"^\d+:\s*", "", label_text.strip()).lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip('_')
    return normalized if normalized else DEFAULT_LABEL_FALLBACK

try:
    from utils_dataset import json_to_yolo, normalize_label, DEFAULT_LABEL
    from training_manager import TrainingManager
    if OPENCLIP_AVAILABLE:
         from memory_manager import MemoryManager
    LOCAL_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: No se pudieron importar módulos locales (utils_dataset.py, training_manager.py, memory_manager.py): {e}")
    LOCAL_MODULES_AVAILABLE = False
    if not OPENCLIP_AVAILABLE: MemoryManager = None
    else: MemoryManager = None
    TrainingManager = None
    DEFAULT_LABEL = DEFAULT_LABEL_FALLBACK
    normalize_label = normalize_label_fallback

API_KEY = "API_KEY"
GEMINI_MODEL_NAME = 'gemini-2.5-flash-preview-04-17'
GEMINI_BATCH_SIZE = 16
GEMINI_THINKING_BUDGET = 1024
BOX_COLOR_DEFAULT = "#00e676"
BOX_COLOR_ACTIVE = "#1565C0"
BOX_COLOR_CROSSHAIR = "#FFD700"
BOX_COLOR_SUGGESTION = "#FF8C00"
BOX_COLOR_SUGGESTION_EDIT = "#FFFF00"  # Amarillo para la sugerencia en edición
BOX_W_NORMAL = 2
BOX_W_ACTIVE = 3
MIN_BOX_SIZE = 5
CONFIG_FILE_NAME = ".jarvis_ui_helper_config.json"
DEFAULT_CLASSES_FILENAME = "YOLOE_DESCRIPTIONS.txt"
MEMORY_SIMILARITY_THRESHOLD = 0.30

DEFAULT_GROUNDINGDINO_CONFIG_FILENAME = "GroundingDINO_SwinB_COGCOOR.py"
DEFAULT_GROUNDINGDINO_CHECKPOINT_FILENAME = "groundingdino_swinb_cogcoor.pth"
GROUNDINGDINO_BOX_THRESHOLD = 0.30
GROUNDINGDINO_TEXT_THRESHOLD = 0.25

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
LABELS_JSON_DIR = DATA_DIR / "labels_json"
LABELS_YOLO_DIR = DATA_DIR / "labels_yolo"
MODELS_DIR = SCRIPT_DIR / "models"
FAISS_INDEX_PATH = MODELS_DIR / "visual_memory_yoloe.index"
FAISS_LABELS_PATH = MODELS_DIR / "visual_memory_labels_yoloe.json"

def normalize_description_text(desc: str) -> str:
    if not isinstance(desc, str): return ""
    return desc.strip().lower()

class EditDescriptionsWindow(tk.Toplevel):
    def __init__(self, parent, main_app_ref):
        super().__init__(parent)
        self.main_app = main_app_ref; self.transient(parent); self.grab_set(); self.title("Editar Descripciones"); self.geometry("500x450")
        main_frame = tb.Frame(self, padding=10); main_frame.pack(expand=True, fill="both")
        tk.Label(main_frame, text="Descripciones Actuales:").pack(anchor="w")
        self.listbox_frame = tb.Frame(main_frame); self.listbox_frame.pack(fill="both", expand=True, pady=5)
        self.scrollbar = tb.Scrollbar(self.listbox_frame, orient="vertical")
        self.descriptions_listbox = tk.Listbox(self.listbox_frame, yscrollcommand=self.scrollbar.set, exportselection=False)
        self.scrollbar.config(command=self.descriptions_listbox.yview); self.scrollbar.pack(side="right", fill="y"); self.descriptions_listbox.pack(side="left", fill="both", expand=True)
        self.populate_list()
        button_frame = tb.Frame(main_frame); button_frame.pack(fill="x", pady=(10,0))
        self.btn_rename = tb.Button(button_frame, text="Renombrar", command=self.rename_selected, bootstyle="info-outline"); self.btn_rename.pack(side="left", expand=True, padx=2)
        self.btn_delete = tb.Button(button_frame, text="Borrar", command=self.delete_selected, bootstyle="danger-outline"); self.btn_delete.pack(side="left", expand=True, padx=2)
        global_actions_frame = tb.Frame(main_frame); global_actions_frame.pack(fill="x", pady=(10,0))
        self.btn_delete_all = tb.Button(global_actions_frame, text="Borrar Todas", command=self.delete_all, bootstyle="danger"); self.btn_delete_all.pack(side="left", expand=True, padx=2)
        self.btn_close = tb.Button(global_actions_frame, text="Cerrar", command=self.destroy, bootstyle="secondary"); self.btn_close.pack(side="right", expand=True, padx=2)
        self.protocol("WM_DELETE_WINDOW", self.destroy); self.update_idletasks(); x=parent.winfo_x()+(parent.winfo_width()//2)-(self.winfo_width()//2); y=parent.winfo_y()+(parent.winfo_height()//2)-(self.winfo_height()//2); self.geometry(f"+{x}+{y}")
    def populate_list(self): self.descriptions_listbox.delete(0,tk.END); [self.descriptions_listbox.insert(tk.END,d) for d in sorted(self.main_app.get_class_names())]
    def delete_selected(self):
        sel_idx=self.descriptions_listbox.curselection()
        if not sel_idx: messagebox.showwarning("Borrar","Selecciona una descripción.",parent=self); return
        sel_desc=self.descriptions_listbox.get(sel_idx[0])
        if messagebox.askyesno("Confirmar","Borrar '{}'?".format(sel_desc),parent=self) and sel_desc in self.main_app.class_names:
            self.main_app.class_names.remove(sel_desc); self._update_boxes_and_main_app("delete",sel_desc,None); self.populate_list()
    def rename_selected(self):
        sel_idx=self.descriptions_listbox.curselection()
        if not sel_idx: messagebox.showwarning("Renombrar","Selecciona una descripción.",parent=self); return
        old_desc=self.descriptions_listbox.get(sel_idx[0])
        new_raw=simpledialog.askstring("Renombrar",f"Nuevo nombre para '{old_desc}':",initialvalue=old_desc,parent=self)
        if new_raw and (new_norm:=normalize_description_text(new_raw)):
            if new_norm==old_desc: return
            if new_norm in self.main_app.class_names: messagebox.showerror("Error",f"'{new_norm}' ya existe.",parent=self); return
            if old_desc in self.main_app.class_names:
                self.main_app.class_names.remove(old_desc); self.main_app.class_names.append(new_norm); self.main_app.class_names.sort()
                self._update_boxes_and_main_app("rename",old_desc,new_norm); self.populate_list()
        elif new_raw is not None: messagebox.showerror("Error","Nombre no válido.",parent=self)
    def delete_all(self):
        if messagebox.askyesno("Confirmar","Borrar TODAS las descripciones?",parent=self):
            self.main_app.class_names.clear(); self._update_boxes_and_main_app("delete_all",None,None); self.populate_list()
    def _update_boxes_and_main_app(self,action:str,old_val:Optional[str],new_val:Optional[str]):
        for tab in self.main_app.tabs.values():
            if tab and tab.winfo_exists():
                changed_tab=False
                for i,box_data in enumerate(tab.boxes):
                    x1,y1,x2,y2,descs,conf = box_data; upd_descs=list(descs or [DEFAULT_LABEL]); modified=False
                    if action=="delete" and old_val in upd_descs: upd_descs.remove(old_val); modified=True; upd_descs=upd_descs or [DEFAULT_LABEL]
                    elif action=="rename" and old_val in upd_descs: 
                        try: upd_descs[upd_descs.index(old_val)]=new_val; modified=True
                        except ValueError: pass
                    elif action=="delete_all": upd_descs=[DEFAULT_LABEL]; modified=True
                    if modified: tab.boxes[i]=(x1,y1,x2,y2,sorted(list(set(upd_descs))),conf); changed_tab=True
                if changed_tab: tab._redraw(); tab._update_cls_panel()
        self.main_app.save_class_names(); self.main_app.refresh_class_lists_in_tabs()
        messagebox.showinfo("Actualizado","Descripciones actualizadas.",parent=self)

class DetectorElementosUI:
    def __init__(self, min_area: int = 30, max_area_ratio: float = 0.7, iou_threshold: float = 0.1,
                 canny_dilate_iterations: int = 2, close_kernel_size: int = 5, close_iterations: int = 1):
        self.min_area=min_area; self.max_area_ratio=max_area_ratio; self.iou_threshold=iou_threshold; self.canny_dilate_iterations=canny_dilate_iterations
        self.close_kernel_size=close_kernel_size; self.close_iterations=close_iterations
    def _fusionar_regiones_nms(self, regiones: List[Tuple[int,int,int,int]], thr: float) -> List[Tuple[int,int,int,int]]:
        if not regiones: return []
        cajas=np.array(regiones); areas=cajas[:,2]*cajas[:,3]; idxs=np.argsort(areas)[::-1]; out=[]
        while idxs.size > 0:
            i=idxs[0]; out.append(tuple(cajas[i])); rem=[]
            for j_idx in range(1, idxs.size):
                j=idxs[j_idx]; x1,y1,w1,h1=cajas[i]; x2,y2,w2,h2=cajas[j]; xa,ya=max(x1,x2),max(y1,y2)
                xb,yb=min(x1+w1,x2+w2),min(y1+h1,y2+h2); inter_area=max(0,xb-xa)*max(0,yb-ya)
                union_area=float(w1*h1+w2*h2-inter_area+1e-6); iou=inter_area/union_area
                if iou < thr: rem.append(j)
            idxs=np.array(rem)
        return out
    def _find_contours_and_filter(self,binary_image:np.ndarray,max_area:float)->List[Tuple[int,int,int,int]]:
        cnts,_=cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); regs=[]
        for c in cnts:
            area=cv2.contourArea(c)
            if self.min_area<area<max_area:
                x,y,wc,hc=cv2.boundingRect(c)
                if wc>MIN_BOX_SIZE and hc>MIN_BOX_SIZE and 0.05 < (wc/(hc+1e-6)) < 20.0: regs.append((x,y,wc,hc))
        return regs
    def detectar(self,img:np.ndarray)->List[Tuple[int,int,int,int]]:
        if img is None or img.size==0: return []
        h,w=img.shape[:2]; max_area=h*w*self.max_area_ratio; gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur=cv2.bilateralFilter(gray,9,50,50); edges=cv2.Canny(blur,50,150)
        dilated_edges=cv2.dilate(edges,np.ones((3,3),np.uint8),iterations=self.canny_dilate_iterations)
        canny_regions=self._find_contours_and_filter(dilated_edges,max_area)
        blur_thresh=cv2.GaussianBlur(gray,(5,5),0)
        thresh=cv2.adaptiveThreshold(blur_thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,4)
        closed_thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(self.close_kernel_size,self.close_kernel_size)),iterations=self.close_iterations)
        thresh_regions=self._find_contours_and_filter(closed_thresh,max_area)
        all_regions=canny_regions+thresh_regions
        return self._fusionar_regiones_nms(all_regions,self.iou_threshold) if all_regions else []

class ImageTab(tb.Frame):
    BOX_ENLARGE_INCREMENT = 10; RESIZE_HANDLE_VISUAL_SIZE = 6; RESIZE_HANDLE_SENSITIVITY_PIXELS = 8 
    def __init__(self, parent_notebook: tb.Notebook, main_app_ref, image_path: pathlib.Path, output_dir: pathlib.Path, api_key: Optional[str], yolo_model: Optional[YOLO], memory_manager: Optional[MemoryManager]):
        super().__init__(parent_notebook)
        self.notebook = parent_notebook; self.main_app = main_app_ref; self.image_path = image_path
        self.labels_json_dir = output_dir; self.api_key = api_key
        self.class_names: List[str] = list(self.main_app.get_class_names())
        self.class2id: Dict[str, int] = {name: i for i, name in enumerate(self.class_names)}
        self.yolo_model = yolo_model; self.memory_manager = memory_manager
        self.boxes: List[Tuple[int, int, int, int, List[str], int]] = []
        self.active_ids: set[int] = set(); self.undo_stack: List[Tuple[str, Any]] = []
        self.redo_stack: List[Tuple[str, Any]] = []
        self.suggested_boxes_data: List[Tuple[List[int], str, float]] = []
        self.gemini_request_count = 0; self.gemini_processing = False; self.gemini_queue = queue.Queue()
        self.panning_mode = False; self.pan_start: Optional[Tuple[int, int]] = None
        self.draw_start: Optional[Tuple[int, int]] = None; self.tmp_rect_id: Optional[int] = None
        self.box_visible = True; self.hover_idx: Optional[int] = None
        self.sel_start: Optional[Tuple[int, int]] = None; self.sel_rect_id: Optional[int] = None; self.sel_prev_active: set[int] = set()
        self.is_moving_boxes=False; self.move_start_img_coords:Optional[Tuple[int,int]]=None; self.moved_boxes_initial_state:Dict[int,Tuple[int,int,int,int,List[str],int]]={}
        self.is_resizing_box=False; self.resize_box_idx:Optional[int]=None; self.resize_handle_type:Optional[str]=None; self.resize_start_img_coords:Optional[Tuple[int,int]]=None; self.resize_box_original_state:Optional[Tuple[int,int,int,int,List[str],int]]=None
        self.current_hover_resize_handle:Optional[str]=None; self.current_hover_resize_box_idx:Optional[int]=None
        self.alt_l_pressed=False; self.is_alt_adjusting_edges=False; self.alt_adjust_click_img_coords:Optional[Tuple[int,int]]=None; self.alt_adjust_original_boxes_state:Dict[int,Tuple[int,int,int,int,List[str],int]]={}
        self.detector=DetectorElementosUI()
        self.crosshair_h_line=None; self.crosshair_v_line=None; self.show_crosshair=False
        
        self.editing_suggestion_idx: Optional[int] = None
        self.editing_suggestion_original_data: Optional[Tuple[List[int], str, float]] = None
        self.is_moving_suggestion: bool = False
        self.is_resizing_suggestion: bool = False
        self.resize_suggestion_handle_type: Optional[str] = None
        self.resize_suggestion_start_img_coords: Optional[Tuple[int, int]] = None
        self.resize_suggestion_original_state_during_op: Optional[Tuple[List[int], str, float]] = None # State of suggestion at start of current move/resize op
        self.current_hover_resize_suggestion_handle: Optional[str] = None
        self.current_hover_resize_suggestion_idx: Optional[int] = None


        try:
            self.base_img=Image.open(self.image_path).convert("RGB"); self.original_image_width=self.base_img.width; self.original_image_height=self.base_img.height
        except Exception as e: self._build_error_ui(f"Error al abrir imagen:\n{e}"); return
        self.scale=1.0; self.offset=[0,0]; self.gemini_enabled=REQUESTS_AVAILABLE and self.api_key
        self.rowconfigure(0,weight=1); self.columnconfigure(0,weight=1); self.columnconfigure(1,weight=0)
        self._build_canvas(); self._build_side_panel(); self.bind_events()
        self._refresh_class_list(init=True); self._load_prev_labels()
        run_initial_cv = not self.boxes and not (self.main_app.active_pipeline_detector=="yolo" and self.yolo_model) and not (self.main_app.active_pipeline_detector=="groundingdino" and self.main_app.loaded_groundingdino_model)
        if run_initial_cv: self._run_initial_detection()
        self._update_box_count_display(); self._update_gemini_count_display(); self._redraw(); self._update_group_button_state(); self.update_pipeline_button_state(); self.canvas.focus_set()
        self._update_cursor_and_crosshair()
        self._update_suggestion_buttons_state()

    def _add_to_undo(self, action: str, data: Any):
        self.undo_stack.append((action, data))
        if self.redo_stack: self.redo_stack.clear()
    def _image_to_base64(self, img: Image.Image, format="JPEG") -> str:
        buffered = io.BytesIO(); quality = 85 if max(img.size) > 1024 else 95
        img.save(buffered, format=format, quality=quality)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    def _build_error_ui(self, error_message: str):
        tb.Label(self,text=error_message,justify="left",anchor="center").pack(expand=True,fill="both",padx=20,pady=20)
    def _build_canvas(self):
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        try:
            self.tk_img = ImageTk.PhotoImage(self.base_img)
            self.img_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        except Exception as e: self.canvas.create_text(10,10,anchor="nw",text=f"Error PhotoImage:{e}",fill="red")
    
    def _cancel_edit_suggestion(self):
        if self.editing_suggestion_idx is not None and \
           self.editing_suggestion_original_data is not None and \
           0 <= self.editing_suggestion_idx < len(self.suggested_boxes_data):
            self.suggested_boxes_data[self.editing_suggestion_idx] = copy.deepcopy(self.editing_suggestion_original_data)
            
        self.editing_suggestion_idx = None
        self.editing_suggestion_original_data = None
        self.is_moving_suggestion = False
        self.is_resizing_suggestion = False
        self._redraw()
        self._update_cls_panel() 
        self._update_suggestion_buttons_state()
        self._update_cursor_and_crosshair()


    def _approve_edited_suggestion(self):
        if self.editing_suggestion_idx is None or \
           not (0 <= self.editing_suggestion_idx < len(self.suggested_boxes_data)):
            return

        edited_xyxy, prompt_text, original_score = self.suggested_boxes_data[self.editing_suggestion_idx]
        normalized_desc = self.main_app.add_description_if_new(prompt_text)
        
        new_box_data = (edited_xyxy[0], edited_xyxy[1], edited_xyxy[2], edited_xyxy[3], 
                        [normalized_desc], float(original_score))
        self.boxes.append(new_box_data)
        new_box_internal_idx = len(self.boxes) - 1

        original_suggestion_data_for_undo = copy.deepcopy(self.editing_suggestion_original_data)
        original_suggestion_list_idx_for_undo = self.editing_suggestion_idx

        self.suggested_boxes_data.pop(self.editing_suggestion_idx)

        self._add_to_undo("approve_single_suggestion", {
            "added_box_idx": new_box_internal_idx,
            "original_suggestion_data_tuple": original_suggestion_data_for_undo, 
            "original_suggestion_list_idx": original_suggestion_list_idx_for_undo
        })
        
        self.editing_suggestion_idx = None
        self.editing_suggestion_original_data = None
        self.is_moving_suggestion = False
        self.is_resizing_suggestion = False
        self.active_ids = {new_box_internal_idx} 

        self._update_box_count_display()
        self._redraw()
        self._update_cls_panel()
        self._update_suggestion_buttons_state()
        self._update_cursor_and_crosshair()
        print(f"INFO: Sugerencia aprobada y añadida como caja: {new_box_data[4][0]}")

    def _gemini_sel(self):
        if self.gemini_processing:
            messagebox.showwarning("En Progreso", "Gemini ya está en ejecución.", parent=self)
            return
        if not self.active_ids:
            messagebox.showinfo("Gemini Sel", "Selecciona una o más cajas primero.", parent=self)
            return
        
        indices_to_process = list(self.active_ids)
        if not indices_to_process:
            messagebox.showinfo("Gemini Sel", "No hay cajas válidas seleccionadas.", parent=self)
            return

        print(f"INFO ({self.image_path.name}): Iniciando Gemini para {len(indices_to_process)} caja(s) seleccionada(s)...")
        self._start_gemini_thread(indices_to_process)

    def _gemini_all(self):
        if self.gemini_processing:
            messagebox.showwarning("En Progreso", "Gemini ya está en ejecución.", parent=self)
            return
        if not self.boxes:
            messagebox.showinfo("Gemini All", "No hay cajas en la imagen.", parent=self)
            return
        
        indices_to_process = list(range(len(self.boxes)))
        num_boxes = len(indices_to_process)
        
        if num_boxes > 50: 
            if not messagebox.askyesno("Confirmar Gemini All", 
                                       f"Procesar {num_boxes} cajas con Gemini podría tomar tiempo y consumir API cuota.\n¿Continuar?", 
                                       parent=self):
                return
        
        print(f"INFO ({self.image_path.name}): Iniciando Gemini para TODAS las {num_boxes} cajas...")
        self._start_gemini_thread(indices_to_process)

    def _start_gemini_thread(self, indices_to_process: List[int]):
        if not self.gemini_enabled:
            messagebox.showerror("Gemini Deshabilitado", "Revisa tu API Key o la librería 'requests'.", parent=self)
            return
        
        self.gemini_processing = True
        self.btn_gemini_sel.config(state="disabled")
        self.btn_gemini_all.config(state="disabled")
        
        self.progress_frame.pack(fill="x", pady=(5,0), before=self.gemini_control_frame) 
        self.progress_bar['value'] = 0
        num_batches = (len(indices_to_process) + GEMINI_BATCH_SIZE - 1) // GEMINI_BATCH_SIZE
        self.progress_label.config(text=f"Iniciando {num_batches} lotes...")
        self.progress_bar['maximum'] = num_batches if num_batches > 0 else 1
        
        while not self.gemini_queue.empty():
            try:
                self.gemini_queue.get_nowait()
            except queue.Empty:
                break

        thread = threading.Thread(
            target=self._gemini_worker,
            args=(indices_to_process, num_batches, self._call_gemini_api_batch, self.gemini_queue),
            daemon=True
        )
        thread.start()
        self.after(100, self._check_gemini_queue)

    def _gemini_worker(self, indices:List[int], num_batches:int, api_call_func:Callable, out_q:queue.Queue):
        try:
            old_map={idx:list(self.boxes[idx][4]) for idx in indices if 0<=idx<len(self.boxes)}
            all_res:Dict[int,Tuple[str,float]]={}; failed_batches=0; total_processed_boxes=0
            
            for i in range(0, len(indices), GEMINI_BATCH_SIZE):
                current_batch_num = (i // GEMINI_BATCH_SIZE) + 1
                out_q.put({'type': 'progress', 'current': current_batch_num, 'total': num_batches})
                
                batch_indices = indices[i : i + GEMINI_BATCH_SIZE]
                batch_crops = []
                valid_indices_in_batch = []
                
                for box_idx in batch_indices:
                    if not (0 <= box_idx < len(self.boxes)):
                        print(f"WARN (Gemini Worker): Índice de caja {box_idx} inválido, saltando.")
                        continue
                    x1, y1, x2, y2, _, _ = self.boxes[box_idx]
                    img_w, img_h = self.base_img.size
                    crop_x1, crop_y1, crop_x2, crop_y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
                    
                    if crop_x1 < crop_x2 and crop_y1 < crop_y2: 
                        try:
                            crop_image = self.base_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                            batch_crops.append(crop_image)
                            valid_indices_in_batch.append(box_idx)
                        except Exception as e_crop:
                            print(f"ERROR (Gemini Worker): No se pudo recortar la imagen para la caja {box_idx}: {e_crop}")
                    else:
                        print(f"WARN (Gemini Worker): Caja {box_idx} con dimensiones inválidas ({crop_x1},{crop_y1},{crop_x2},{crop_y2}), saltando.")

                if batch_crops: 
                    batch_results = api_call_func(self.base_img, batch_crops, valid_indices_in_batch, self.class_names)
                    if batch_results:
                        all_res.update(batch_results)
                        total_processed_boxes += len(batch_results)
                    else:
                        failed_batches += 1 
                elif valid_indices_in_batch: 
                     failed_batches +=1 
                     print(f"WARN (Gemini Worker): Lote {current_batch_num} no tuvo imágenes válidas para procesar.")


            results_data = {
                'results_dict': all_res,
                'old_labels_map': old_map,
                'failed_batches': failed_batches,
                'total_processed': total_processed_boxes 
            }
            out_q.put({'type': 'results', 'data': results_data})

        except Exception as e_worker:
            print(f"ERROR CRÍTICO en Hilo Gemini (_gemini_worker): {e_worker}")
            traceback.print_exc()
            out_q.put({'type': 'error', 'message': f"Error interno en el hilo de Gemini:\n{e_worker}\n\n{traceback.format_exc()}"})
        finally:
            out_q.put({'type': 'done'})

    def _call_gemini_api_batch(self,full_img:Image.Image,batch_crops:List[Image.Image],batch_idxs:List[int],current_descriptions:List[str])->Dict[int,Tuple[str,float]]:
        if not self.gemini_enabled or not batch_crops or not REQUESTS_AVAILABLE or not self.api_key: return {}
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={self.api_key}"; headers={"Content-Type":"application/json"}
        
        num_current_batch_crops = len(batch_crops)

        prompt_header = (
            "**Role:** You are an expert UI/GUI Analyst. Your mission is to generate highly descriptive and semantically rich textual prompts for specific visual elements within a User Interface. These descriptions will be used to train an advanced AI vision model to accurately identify, differentiate, and understand these UI elements for subsequent interaction.\n\n"
            "**Input Context:** You will first be provided with a `FULL IMAGE` screenshot of a complete user interface. This full image is crucial for understanding the overall layout, context, and potential function of the elements. Following the full image, you will receive one or more `CROPPED IMAGE` segments, each isolating a specific UI element from the `FULL IMAGE` that requires a detailed textual description.\n\n"
            "**Detailed Task for EACH `CROPPED IMAGE`:**\n"
            "For every `CROPPED IMAGE` you analyze, meticulously perform the following steps and synthesize the information into a coherent, distinguishing textual description. The goal is to capture the visual and functional essence of the element in a way that is unambiguous for an AI.\n\n"
            "1.  **Precise Element Type Identification:**\n"
            "    *   Determine the functional category of the element. Common examples include: 'action button', 'radio button', 'checkbox', 'toggle switch', 'text input field', 'descriptive text label', 'section title', 'functional icon', 'illustrative image', 'dropdown menu item', 'slider control', 'progress bar', 'hyperlink', 'application logo', 'company logo', 'video thumbnail', 'image thumbnail', 'navigation tab', 'user avatar icon', 'menu bar item', 'scrollbar element'.\n"
            "    *   Be as specific as your analysis allows. Instead of just 'button', specify 'selected radio button with blue dot', 'primary call-to-action button', 'close button with X icon'.\n\n"
            "2.  **Verbatim Extraction of Visible Text Content:**\n"
            "    *   Transcribe any and all legible text found directly on or within the UI element. This is critical for elements like buttons, labels, placeholders in input fields, titles, menu items, etc.\n"
            "    *   Example: If a button displays 'Save Changes', your description must include 'text Save Changes'. Quote a_i_generated_text directly.\n\n"
            "3.  **Description of Key Visual Attributes and Observable State:**\n"
            "    *   **Color:** Specify dominant or functionally significant colors (e.g., 'blue primary action button', 'red warning text', 'light gray background panel').\n"
            "    *   **Iconography:** If an icon is present, describe its visual appearance and, if obvious, its meaning (e.g., 'button with a magnifying glass icon for search functionality', 'folder icon indicating open file operation', 'red and white YouTube play button logo').\n"
            "    *   **Shape & Style:** Detail the general shape if it's a distinguishing feature (e.g., 'button with rounded corners', 'text input field with a thin bottom border and subtle inner shadow', 'iOS-style toggle switch').\n"
            "    *   **State (if visually apparent):** Indicate if the element appears to be in a particular state, such as 'enabled', 'disabled/dimmed/grayed-out', 'selected', 'checked', 'unchecked', 'toggled on', 'toggled off', 'hovered (if inferable from visual cues like highlighting)'. Example: 'checkbox успішно позначено синьою галочкою', 'disabled send button in light gray'.\n\n"
            "4.  **Functional Inference from Context (Leverage the `FULL IMAGE`):**\n"
            "    *   By observing the `FULL IMAGE` and the element itself, if you can clearly infer its primary purpose or role within the UI, briefly include this functional context. This adds significant semantic value.\n"
            "    *   Example: 'text input field for entering the email address in a user registration form', 'main video playback button for the featured content', 'bell-shaped notification icon in the top status bar'.\n\n"
            "5.  **Construction of the Final, Discriminative Description:**\n"
            "    *   Synthesize the information from points 1-4 into a natural language description that is precise, comprehensive, and maximizes semantic distinctiveness without excessive verbosity. The description should be a phrase or a short sentence.\n"
            "    *   **Prioritization Order:** Element Type, then Visible Text (if any), then the most salient Visual Attributes (color, icon, shape, state), and finally Functional Context (if clear and concise).\n"
            "    *   **Strive for Excellence - Examples:**\n"
            "        *   POOR: 'blue square'\n"
            "        *   BETTER: 'blue button with OK text'\n"
            "        *   EXCELLENT: 'rounded primary blue action button with white uppercase text OK and a subtle drop shadow'\n"
            "        *   POOR: 'picture'\n"
            "        *   BETTER: 'user icon'\n"
            "        *   EXCELLENT: 'circular user avatar icon with a gray person silhouette placeholder and a green online status indicator dot'\n"
            "        *   POOR: 'text field'\n"
            "        *   BETTER: 'search bar with magnifying glass'\n"
            "        *   EXCELLENT: 'text input field for site-wide search with placeholder text Search... and a gray magnifying glass icon on the left'\n\n"
            "6.  **Confidence Score Estimation:**\n"
            "    *   Provide a confidence score between 0.0 and 1.0 (e.g., 0.95) reflecting how certain you are that your generated description is accurate, complete, and will be highly effective for the AI vision model to uniquely identify this specific UI element.\n"
        )

        prompt_footer = (
            "**Output Instructions (Strict Adherence Required - ONE LINE PER CROP - NO ADDITIONAL EXPLANATIONS - Field order must be exact and case-sensitive):**\n"
            "For each `CROPPED IMAGE` provided in this batch, respond on a new line using the following precise format. Do not include any introductory or concluding remarks outside of this specified format for each crop.\n"
            "`Crop: [Integer number of the crop within this batch, starting from 1] Description: [Your detailed, semantic, English-language description of the UI element] Confidence: [Your confidence score as a float, e.g., 0.85]`\n\n"
            f"**Begin processing. Please provide descriptions for all {num_current_batch_crops} cropped images in this batch according to these detailed instructions:**\n"
        )
        parts = [{"text": prompt_header}];
        try: parts.append({"text":"\nFULL IMAGE (for context):"}); parts.append({"inline_data":{"mime_type":"image/jpeg","data":self._image_to_base64(full_img.convert("RGB"))}})
        except Exception as e: return {}
        parts.append({"text":"\nCROPPED IMAGES to describe (one by one):"})
        for c_idx, c_img in enumerate(batch_crops):
            try: parts.append({"text":f"\n--- Analyzing CROPPED IMAGE {c_idx + 1} of {len(batch_crops)} ---"}); parts.append({"inline_data":{"mime_type":"image/jpeg","data":self._image_to_base64(c_img.convert("RGB"))}})
            except Exception as e: return {}
        parts.append({"text":prompt_footer})
        payload={"contents":[{"parts":parts}],"generationConfig":{"temperature":0.2,"topK":1,"maxOutputTokens":8192},"safetySettings":[{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]]}
        if GEMINI_THINKING_BUDGET and GEMINI_THINKING_BUDGET>0: payload["generationConfig"]["thinkingConfig"]={"thinkingBudget":GEMINI_THINKING_BUDGET}
        try:
            self.gemini_request_count+=1; resp=requests.post(url,headers=headers,json=payload,timeout=300); resp.raise_for_status() 
            resp_data=resp.json(); results:Dict[int,Tuple[str,float]]={}; gen_text=""
            try:
                if 'candidates' in resp_data and resp_data['candidates']:
                    cand=resp_data['candidates'][0]; reason=cand.get('finishReason')
                    if reason=='SAFETY': print(f"ERROR: Gemini bloqueada. Ratings: {cand.get('safetyRatings',[])}"); return {} 
                    elif reason!='STOP' and reason is not None and reason!='MAX_TOKENS': print(f"WARN: Finish reason: {reason}")
                    if 'content' in cand and 'parts' in cand['content']: gen_text="\n".join(p['text'] for p in cand['content']['parts'] if 'text' in p).strip()
                if not gen_text: print("WARN: Gemini no devolvió texto.")
            except Exception as e_parse: print(f"WARN: Error parseando Gemini: {e_parse}")
            parsed_count=0; line_pattern=re.compile(r"Crop\s*:\s*(\d+)\s*Description\s*:(.*?)(Confidence\s*:\s*[0-9.]+|\Z)", re.I|re.S); confidence_pattern=re.compile(r"Confidence\s*:\s*([0-9.]+)")
            for match in line_pattern.finditer(gen_text):
                try:
                    crop_num_str=match.group(1).strip(); desc_raw=match.group(2).strip(); conf_str="0.0"; conf_match=confidence_pattern.search(match.group(3))
                    if conf_match: conf_str=conf_match.group(1).strip()
                    elif (m_conf:=confidence_pattern.search(desc_raw)): conf_str=m_conf.group(1).strip(); desc_raw=desc_raw[:m_conf.start()].strip()
                    crop_num=int(crop_num_str); conf=float(conf_str); desc_txt=desc_raw if desc_raw else DEFAULT_LABEL; target_idx=crop_num-1
                    if 0<=target_idx<len(batch_idxs): results[batch_idxs[target_idx]]=(desc_txt,conf); parsed_count+=1
                except Exception as e_line: print(f"ERROR Parse Line: '{match.group(0)}' -> {e_line}")
            if parsed_count==0 and len(batch_crops)>0 : print("WARN Parse: No descripciones parseadas.")
            return results
        except requests.exceptions.HTTPError as http_err: err_msg=f"Error HTTP Gemini: {http_err}"; print(f"[ERROR Worker] {err_msg}"); return {}
        except requests.exceptions.RequestException as req_err: print(f"[ERROR Worker] Error Red/Conexión Gemini: {req_err}"); return {}
        except Exception as e: print(f"[ERROR Worker] Error inesperado Gemini: {type(e).__name__}: {e}"); traceback.print_exc(); return {}

    def _apply_gemini_results(self, results_data: Dict):
        results = results_data.get('results_dict',{}); old_map = results_data.get('old_labels_map',{})
        failed_batches = results_data.get('failed_batches',0); total_processed_from_gemini = results_data.get('total_processed',0)
        changed_anything = False; changed_indices = []; current_labels_for_undo = {}
        for box_idx,(new_desc_raw,confidence) in results.items():
            if not(0<=box_idx<len(self.boxes)): continue
            current_labels_for_undo[box_idx]=list(self.boxes[box_idx][4])
            normalized_new_desc=self.main_app.add_description_if_new(new_desc_raw)
            if not normalized_new_desc or (normalized_new_desc==DEFAULT_LABEL and confidence<0.5): normalized_new_desc=DEFAULT_LABEL
            current_desc=current_labels_for_undo[box_idx][0] if current_labels_for_undo[box_idx] else DEFAULT_LABEL
            if current_desc!=normalized_new_desc:
                x1,y1,x2,y2,_,old_conf=self.boxes[box_idx]; self.boxes[box_idx]=(x1,y1,x2,y2,[normalized_new_desc],float(confidence))
                changed_anything=True; changed_indices.append(box_idx)
        if changed_anything:
            self._add_to_undo("gemini",{"indices":changed_indices,"old_labels":current_labels_for_undo})
            self.active_ids=set(changed_indices); self._update_cls_panel(); self._redraw()
            msg=f"Gemini: {len(changed_indices)}/{total_processed_from_gemini} caja(s) actualizada(s)."
            if failed_batches>0: msg+=f"\n{failed_batches} lote(s) fallaron."
            messagebox.showinfo("Gemini Completado",msg,parent=self)
        else:
            msg="Gemini no realizó cambios."; 
            if total_processed_from_gemini==0 and failed_batches>0: msg=f"Gemini falló ({failed_batches} lotes)."
            elif failed_batches>0: msg+=f"\n{failed_batches} lote(s) fallaron."
            messagebox.showinfo("Gemini Completado",msg,parent=self)
        self._update_gemini_count_display()


    def _check_gemini_queue(self):
        try:
            msg=self.gemini_queue.get_nowait(); type=msg.get('type')
            if type=='progress': cur,tot=msg.get('current',0),msg.get('total',1); self.progress_bar['value']=cur; self.progress_bar['maximum']=tot; self.progress_label.config(text=f"Lote {cur}/{tot}...")
            elif type=='results': self._apply_gemini_results(msg.get('data',{}))
            elif type=='error': messagebox.showerror(f"Error Gemini ({self.image_path.name})",msg.get('message','Error.'),parent=self)
            elif type=='done': self.gemini_processing=False; self.progress_frame.pack_forget()
            self.gemini_queue.task_done()
        except queue.Empty: pass 
        except Exception as e: self.gemini_processing=False; self.progress_frame.pack_forget(); traceback.print_exc()
        if self.gemini_processing: self.after(100,self._check_gemini_queue)
        elif self.gemini_enabled: self.btn_gemini_sel.config(state="normal"); self.btn_gemini_all.config(state="normal")

    def _update_suggestion_buttons_state(self):
        has_suggestions = bool(self.suggested_boxes_data)
        is_editing_suggestion = self.editing_suggestion_idx is not None

        if hasattr(self, 'btn_approve_suggestions'):
            self.btn_approve_suggestions.config(state="normal" if has_suggestions and not is_editing_suggestion else "disabled")
        if hasattr(self, 'btn_clear_suggestions'):
            self.btn_clear_suggestions.config(state="normal" if has_suggestions and not is_editing_suggestion else "disabled")
        
        if hasattr(self, 'btn_approve_edited_suggestion'):
            self.btn_approve_edited_suggestion.config(state="normal" if is_editing_suggestion else "disabled")
        if hasattr(self, 'btn_cancel_edit_suggestion'):
            self.btn_cancel_edit_suggestion.config(state="normal" if is_editing_suggestion else "disabled")


    def _build_side_panel(self):
        self.side_panel = tb.Frame(self, width=280, padding=6); self.side_panel.grid(row=0, column=1, sticky="ns"); self.side_panel.grid_propagate(False)
        tk.Label(self.side_panel, text="Buscar descripción:", anchor="w").pack(fill="x")
        self.search_var = tk.StringVar(); search_entry = tk.Entry(self.side_panel, textvariable=self.search_var); search_entry.pack(fill="x", pady=(0,4)); search_entry.bind("<KeyRelease>", lambda *_: self._refresh_class_list())
        self.lst_cls = tk.Listbox(self.side_panel, selectmode=tk.SINGLE, exportselection=False, height=12); self.lst_cls.pack(fill="both", expand=True); self.lst_cls.bind("<<ListboxSelect>>", self._on_cls_change)
        tb.Button(self.side_panel, text="Añadir Descripción Manual...", command=self._add_manual_description).pack(fill="x", pady=(4,0))
        count_frame = tb.Frame(self.side_panel); count_frame.pack(fill="x", pady=(6,0))
        self.lbl_box_count=tb.Label(count_frame,text="Cajas: 0",anchor="w"); self.lbl_box_count.pack(side="left",fill="x",expand=True)
        self.lbl_gemini_req_count=tb.Label(count_frame,text="API: 0",anchor="e"); self.lbl_gemini_req_count.pack(side="right",fill="x",expand=True)
        self.lbl_info=tk.Label(self.side_panel,text="Sin selección",anchor="w",wraplength=260,justify="left"); self.lbl_info.pack(fill="x",pady=(4,6))
        self.gemini_control_frame=tb.LabelFrame(self.side_panel,text="Gemini (Descripciones de Texto)",padding=5); self.gemini_control_frame.pack(fill="x",pady=(6,0))
        gemini_btn_state="normal" if self.gemini_enabled else "disabled"
        self.btn_gemini_sel=tb.Button(self.gemini_control_frame,text="⚡ Sel.",bootstyle="primary-outline",command=self._gemini_sel,state=gemini_btn_state); self.btn_gemini_sel.pack(side="left",expand=True,fill="x",padx=2)
        self.btn_gemini_all=tb.Button(self.gemini_control_frame,text="⚡ Todas",bootstyle="info-outline",command=self._gemini_all,state=gemini_btn_state); self.btn_gemini_all.pack(side="left",expand=True,fill="x",padx=2)
        
        yoloe_vp_frame=tb.LabelFrame(self.side_panel,text="YOLOE (Sugerencias Visuales)",padding=5); yoloe_vp_frame.pack(fill="x",pady=(6,0))
        yoloe_vp_btn_state="normal" if ULTRALYTICS_AVAILABLE and self.main_app.loaded_yolo_model and YOLOVP_PREDICTORS_AVAILABLE else "disabled"
        self.btn_yoloe_visual_prompt=tb.Button(yoloe_vp_frame,text="💡 Sugerir (Sel. Cajas)",command=self._yoloe_visual_prompt,state=yoloe_vp_btn_state); self.btn_yoloe_visual_prompt.pack(fill="x", pady=(0,2))
        
        sugg_actions_frame = tb.Frame(yoloe_vp_frame); sugg_actions_frame.pack(fill="x")
        self.btn_approve_suggestions=tb.Button(sugg_actions_frame,text="✅ Todas",command=self._approve_all_suggestions,state="disabled"); self.btn_approve_suggestions.pack(side="left",expand=True,fill="x",padx=2)
        self.btn_clear_suggestions=tb.Button(sugg_actions_frame,text="❌ Todas",command=self._clear_suggestions,state="disabled"); self.btn_clear_suggestions.pack(side="left",expand=True,fill="x",padx=2)
        
        edit_sugg_frame = tb.LabelFrame(self.side_panel, text="Editar Sugerencia", padding=5); edit_sugg_frame.pack(fill="x", pady=(6,0))
        self.btn_approve_edited_suggestion = tb.Button(edit_sugg_frame, text="✅ Aprobar Editada", command=self._approve_edited_suggestion, state="disabled"); self.btn_approve_edited_suggestion.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_cancel_edit_suggestion = tb.Button(edit_sugg_frame, text="❌ Cancelar Edición", command=self._cancel_edit_suggestion, state="disabled"); self.btn_cancel_edit_suggestion.pack(side="left", expand=True, fill="x", padx=2)

        self.progress_frame=tb.Frame(self.side_panel); self.progress_label=tb.Label(self.progress_frame,text="Procesando Lote 0 / 0..."); self.progress_label.pack(pady=(5,0)); self.progress_bar=tb.Progressbar(self.progress_frame,mode='determinate',length=240); self.progress_bar.pack(pady=5,fill="x",expand=True)
        action_btns_frame=tb.Frame(self.side_panel); action_btns_frame.pack(side="bottom",fill="x",pady=(6,0))
        undo_redo_frame=tb.Frame(action_btns_frame); undo_redo_frame.pack(side="left",expand=True,fill="x",padx=2)
        tb.Button(undo_redo_frame,text="↩ Undo",bootstyle="warning-outline",command=self._undo).pack(side="left",expand=True,fill="x")
        tb.Button(undo_redo_frame,text="↪ Redo",bootstyle="info-outline",command=self._redo).pack(side="left",expand=True,fill="x",padx=2)
        tb.Button(action_btns_frame,text="🗑 Borrar",bootstyle="danger-outline",command=self._delete_sel).pack(side="left",expand=True,fill="x",padx=2)
        self.btn_group=tb.Button(action_btns_frame,text="➕ Agrupar",bootstyle="info-outline",command=self._group_selected_boxes,state="disabled"); self.btn_group.pack(side="left",expand=True,fill="x",padx=2)
        self.btn_enlarge=tb.Button(action_btns_frame,text="↔️ Agrandar",bootstyle="secondary-outline",command=self._enlarge_selected_boxes); self.btn_enlarge.pack(side="left",expand=True,fill="x",padx=2)
        self.btn_pipeline=tb.Button(action_btns_frame,text="🔄 Pipeline",bootstyle="success-outline",command=self._run_detection_pipeline,state="disabled"); self.btn_pipeline.pack(side="left",expand=True,fill="x",padx=2)
        tb.Button(self.side_panel,text="👁 Mostrar/Ocultar",bootstyle="secondary-outline",command=self._toggle_boxes).pack(fill="x",pady=(4,0),side="bottom")

    def bind_events(self):
        c=self.canvas; c.bind("<MouseWheel>",self._on_zoom); c.bind("<Button-4>",self._on_zoom); c.bind("<Button-5>",self._on_zoom)
        c.bind("<ButtonPress-1>",self._on_press); c.bind("<B1-Motion>",self._on_drag); c.bind("<ButtonRelease-1>",self._on_release)
        c.bind("<ButtonPress-3>",self._sel_start); c.bind("<B3-Motion>",self._sel_drag); c.bind("<ButtonRelease-3>",self._sel_end)
        c.bind("<Motion>",self._on_motion); c.bind("<KeyPress-space>",self._on_space_press); c.bind("<KeyRelease-space>",self._on_space_release)
        c.bind("<KeyPress-Alt_L>",self._on_alt_l_press); c.bind("<KeyRelease-Alt_L>",self._on_alt_l_release)
        self.canvas.bind("<Delete>", self._delete_sel_event) 
    def _delete_sel_event(self, event=None): self._delete_sel(); return "break"
    def _undo_if_active_tab(self, event=None):
        if self.main_app.get_current_tab() is self: self._undo()
        return "break" 
    def _redo_if_active_tab(self, event=None):
        if self.main_app.get_current_tab() is self: self._redo()
        return "break"
    def _on_alt_l_press(self,event=None): self.after(10,lambda: setattr(self,'alt_l_pressed',True))
    def _on_alt_l_release(self,event=None):
        self.alt_l_pressed=False
        if self.is_alt_adjusting_edges:
            if self.alt_adjust_original_boxes_state:
                changed=[idx for idx,initial in self.alt_adjust_original_boxes_state.items() if 0<=idx<len(self.boxes) and self.boxes[idx][:4]!=initial[:4]]
                if changed: self._add_to_undo("alt_adjust_edges",{idx:data for idx,data in self.alt_adjust_original_boxes_state.items() if idx in changed})
            self.is_alt_adjusting_edges=False; self.alt_adjust_click_img_coords=None; self.alt_adjust_original_boxes_state={}
            self._redraw(); self._update_cls_panel()
    def _to_canvas(self,x,y): return int((x-self.offset[0])/self.scale),int((y-self.offset[1])/self.scale)
    def _from_canvas(self,x,y): return int(x*self.scale+self.offset[0]),int(y*self.scale+self.offset[1])
    def _on_zoom(self,ev):
        factor=1.0
        if sys.platform=="darwin": factor=1.0+ev.delta*0.01
        elif hasattr(ev,'num') and ev.num==4: factor=1.1
        elif hasattr(ev,'num') and ev.num==5: factor=0.9
        elif hasattr(ev,'delta') and ev.delta>0: factor=1.1
        elif hasattr(ev,'delta') and ev.delta<0: factor=0.9
        if factor==1.0: return
        img_x,img_y=self._to_canvas(ev.x,ev.y); new_scale=max(0.1,min(10.0,self.scale*factor))
        if new_scale==self.scale: return
        self.offset[0]=ev.x-img_x*new_scale; self.offset[1]=ev.y-img_y*new_scale; self.scale=new_scale
        self._redraw()
    def _on_space_press(self,event=None):
        if not self.panning_mode: self.panning_mode=True; self.canvas.config(cursor="fleur"); self.is_moving_boxes=False; self.is_resizing_box=False; self.is_alt_adjusting_edges=False; self.draw_start=None; self.show_crosshair=False; self._delete_crosshair()
        self.is_moving_suggestion = False; self.is_resizing_suggestion = False # Cancel suggestion ops
    def _on_space_release(self,event=None):
        if self.panning_mode: self.panning_mode=False; self.pan_start=None; self._update_cursor_and_crosshair(event); self._redraw()

    def _get_resize_handle_at_pos(self,box_idx,canvas_x,canvas_y):
        if not(0<=box_idx<len(self.boxes)): return None
        x1i,y1i,x2i,y2i,_,_=self.boxes[box_idx]; cx1,cy1=self._from_canvas(x1i,y1i); cx2,cy2=self._from_canvas(x2i,y2i); sens=self.RESIZE_HANDLE_SENSITIVITY_PIXELS/2
        handles={'nw':(cx1,cy1),'n':((cx1+cx2)/2,cy1),'ne':(cx2,cy1),'w':(cx1,(cy1+cy2)/2),'e':(cx2,(cy1+cy2)/2),'sw':(cx1,cy2),'s':((cx1+cx2)/2,cy2),'se':(cx2,cy2)}
        for ht,(hx,hy) in handles.items():
            if abs(canvas_x-hx)<=sens and abs(canvas_y-hy)<=sens: return ht
        return None
    
    def _get_resize_handle_at_pos_for_suggestion(self, suggestion_idx, canvas_x, canvas_y):
        if not (0 <= suggestion_idx < len(self.suggested_boxes_data)): return None
        sug_xyxy, _, _ = self.suggested_boxes_data[suggestion_idx]
        x1i, y1i, x2i, y2i = sug_xyxy
        cx1, cy1 = self._from_canvas(x1i, y1i); cx2, cy2 = self._from_canvas(x2i, y2i)
        sens = self.RESIZE_HANDLE_SENSITIVITY_PIXELS / 2
        handles = {'nw': (cx1, cy1), 'n': ((cx1 + cx2) / 2, cy1), 'ne': (cx2, cy1),
                   'w': (cx1, (cy1 + cy2) / 2), 'e': (cx2, (cy1 + cy2) / 2),
                   'sw': (cx1, cy2), 's': ((cx1 + cx2) / 2, cy2), 'se': (cx2, cy2)}
        for ht, (hx, hy) in handles.items():
            if abs(canvas_x - hx) <= sens and abs(canvas_y - hy) <= sens: return ht
        return None


    def _perform_alt_edge_adjust(self,click_x_img,click_y_img):
        if not self.alt_adjust_original_boxes_state: return []
        img_w,img_h=self.base_img.size; changed=[]
        for box_idx,orig_data in self.alt_adjust_original_boxes_state.items():
            if not(0<=box_idx<len(self.boxes)): continue
            ox1,oy1,ox2,oy2,lbls,cnf=orig_data; dl,dr,dt,db=abs(click_x_img-ox1),abs(click_x_img-ox2),abs(click_y_img-oy1),abs(click_y_img-oy2)
            min_d=min(dl,dr,dt,db); nx1,ny1,nx2,ny2=ox1,oy1,ox2,oy2; adj=False
            if min_d==dl:nx1=click_x_img;adj=True; 
            elif min_d==dr:nx2=click_x_img;adj=True; 
            elif min_d==dt:ny1=click_y_img;adj=True; 
            elif min_d==db:ny2=click_y_img;adj=True
            if adj:
                fx1,fx2=min(nx1,nx2),max(nx1,nx2); fy1,fy2=min(ny1,ny2),max(ny1,ny2)
                if(fx2-fx1)<MIN_BOX_SIZE:fx1,fx2=(fx2-MIN_BOX_SIZE,fx2) if nx1==fx1 else (fx1,fx1+MIN_BOX_SIZE)
                if(fy2-fy1)<MIN_BOX_SIZE:fy1,fy2=(fy2-MIN_BOX_SIZE,fy2) if ny1==fy1 else (fy1,fy1+MIN_BOX_SIZE)
                fx1,fy1,fx2,fy2=max(0,int(fx1)),max(0,int(fy1)),min(img_w,int(fx2)),min(img_h,int(fy2))
                if(fx2-fx1)>=MIN_BOX_SIZE and(fy2-fy1)>=MIN_BOX_SIZE and(fx1,fy1,fx2,fy2)!=(ox1,oy1,ox2,oy2): self.boxes[box_idx]=(fx1,fy1,fx2,fy2,lbls,cnf);changed.append(box_idx)
                else: self.boxes[box_idx]=orig_data
            else: self.boxes[box_idx]=orig_data
        if changed: self._redraw(); self._update_cls_panel()
        return changed
    def _on_press(self,ev):
        self.canvas.focus_set(); img_x,img_y=self._to_canvas(ev.x,ev.y); can_x,can_y=ev.x,ev.y; is_sh=(ev.state&1)!=0; is_ct=(ev.state&4)!=0
        
        # Clear any active operation flags
        self.is_moving_boxes=False; self.is_resizing_box=False; self.is_alt_adjusting_edges=False
        self.is_moving_suggestion = False; self.is_resizing_suggestion = False
        self.draw_start = None; self.pan_start = None
        
        if self.panning_mode: 
            self.pan_start=(can_x,can_y); return

        if is_ct: 
            self.draw_start=(img_x,img_y); self.show_crosshair=False; self._delete_crosshair(); return

        if self.alt_l_pressed and len(self.active_ids)>=2: 
            self.alt_adjust_original_boxes_state={i:self.boxes[i] for i in self.active_ids if 0<=i<len(self.boxes)}; self.alt_adjust_click_img_coords=(img_x,img_y); self.is_alt_adjusting_edges=True; self._perform_alt_edge_adjust(img_x,img_y); self.show_crosshair=False; self._delete_crosshair(); return
        
        # Interaction with the suggestion being edited (if any)
        if self.editing_suggestion_idx is not None:
            sug_handle = self._get_resize_handle_at_pos_for_suggestion(self.editing_suggestion_idx, can_x, can_y)
            if sug_handle:
                self.is_resizing_suggestion = True
                self.resize_suggestion_handle_type = sug_handle
                self.resize_suggestion_start_img_coords = (img_x, img_y)
                self.resize_suggestion_original_state_during_op = copy.deepcopy(self.suggested_boxes_data[self.editing_suggestion_idx])
                self.show_crosshair=False; self._delete_crosshair(); return
            else: # Check if click is on the body of the suggestion being edited
                sug_xyxy, _, _ = self.suggested_boxes_data[self.editing_suggestion_idx]
                if sug_xyxy[0] <= img_x <= sug_xyxy[2] and sug_xyxy[1] <= img_y <= sug_xyxy[3]:
                    self.is_moving_suggestion = True
                    self.move_start_img_coords = (img_x, img_y)
                    self.resize_suggestion_original_state_during_op = copy.deepcopy(self.suggested_boxes_data[self.editing_suggestion_idx])
                    self.show_crosshair=False; self._delete_crosshair(); return
        
        # Interaction with regular boxes (resize/move)
        if len(self.active_ids)==1:
            act_idx=next(iter(self.active_ids)); hndl=self._get_resize_handle_at_pos(act_idx,can_x,can_y)
            if hndl: 
                self.is_resizing_box=True; self.resize_box_idx=act_idx; self.resize_handle_type=hndl; self.resize_start_img_coords=(img_x,img_y); self.resize_box_original_state=self.boxes[act_idx]; self.show_crosshair=False; self._delete_crosshair(); return
        
        clk_idx_box = -1
        for i in reversed(range(len(self.boxes))):
            x1, y1, x2, y2, _, _ = self.boxes[i]
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                clk_idx_box = i; break
        
        if clk_idx_box != -1 and clk_idx_box in self.active_ids and not is_sh: 
            self.is_moving_boxes=True; self.move_start_img_coords=(img_x,img_y); self.moved_boxes_initial_state={i_m:self.boxes[i_m] for i_m in self.active_ids if 0<=i_m<len(self.boxes)}; self.show_crosshair=False; self._delete_crosshair(); return

        # Interaction with suggestions (selecting for edit)
        clicked_sug_idx = -1
        for i, (s_xyxy, _, _) in enumerate(self.suggested_boxes_data):
            if s_xyxy[0] <= img_x <= s_xyxy[2] and s_xyxy[1] <= img_y <= s_xyxy[3]:
                clicked_sug_idx = i; break
        
        if clicked_sug_idx != -1:
            if self.editing_suggestion_idx is not None and self.editing_suggestion_idx != clicked_sug_idx:
                self._cancel_edit_suggestion() # Cancel previous before starting new
            
            self.editing_suggestion_idx = clicked_sug_idx
            self.editing_suggestion_original_data = copy.deepcopy(self.suggested_boxes_data[clicked_sug_idx])
            self.active_ids.clear() # Deselect normal boxes
            self._update_cls_panel()
            self._update_suggestion_buttons_state()
            self._redraw()
            self.show_crosshair=False; self._delete_crosshair(); return

        # Select/deselect regular boxes or start drawing new box
        sel_chg=False
        if clk_idx_box != -1: # Click was on a box, but not for moving active ones
            idx=clk_idx_box; old_s=set(self.active_ids)
            if is_sh: self.active_ids.symmetric_difference_update({idx})
            elif not(len(self.active_ids)==1 and idx in self.active_ids): self.active_ids={idx}
            if self.active_ids!=old_s: sel_chg=True; self._add_to_undo("select",old_s)
            self.show_crosshair=False; self._delete_crosshair()
        else: # Click was in empty space
            old_s=set(self.active_ids)
            if old_s: sel_chg=True; self._add_to_undo("select",old_s); self.active_ids.clear()
            self.draw_start=(img_x,img_y); self.show_crosshair=True; self.canvas.config(cursor="none"); self._update_crosshair(ev.x,ev.y)
        
        if sel_chg or clk_idx_box != -1:
            if self.editing_suggestion_idx is not None: # If a box was selected, cancel suggestion editing
                self._cancel_edit_suggestion()
            self._update_cls_panel(); self._redraw()

        self._update_group_button_state(); self._update_cursor_and_crosshair(ev)
        self._update_suggestion_buttons_state()


    def _on_drag(self,ev):
        img_x,img_y=self._to_canvas(ev.x,ev.y); can_x,can_y=ev.x,ev.y
        if self.panning_mode and self.pan_start: dx=can_x-self.pan_start[0]; dy=can_y-self.pan_start[1]; self.canvas.move(self.img_id,dx,dy); self.canvas.move("box",dx,dy); self.canvas.move("box_label",dx,dy); self.offset[0]+=dx; self.offset[1]+=dy; self.pan_start=(can_x,can_y); return
        
        if self.is_resizing_suggestion and self.editing_suggestion_idx is not None and self.resize_suggestion_original_state_during_op:
            orig_sug_xyxy, _, _ = self.resize_suggestion_original_state_during_op
            nx1,ny1,nx2,ny2 = list(orig_sug_xyxy) # Make a mutable copy
            ht=self.resize_suggestion_handle_type
            if 'n' in ht:ny1=img_y; 
            elif 's' in ht:ny2=img_y
            if 'w' in ht:nx1=img_x; 
            elif 'e' in ht:nx2=img_x
            fx1,fx2=min(nx1,nx2),max(nx1,nx2); fy1,fy2=min(ny1,ny2),max(ny1,ny2)
            if(fx2-fx1)<MIN_BOX_SIZE:fx1,fx2=(fx2-MIN_BOX_SIZE,fx2) if nx1==fx1 else(fx1,fx1+MIN_BOX_SIZE)
            if(fy2-fy1)<MIN_BOX_SIZE:fy1,fy2=(fy2-MIN_BOX_SIZE,fy2) if ny1==fy1 else(fy1,fy1+MIN_BOX_SIZE)
            im_w,im_h=self.base_img.size;fx1,fy1,fx2,fy2=max(0,int(fx1)),max(0,int(fy1)),min(im_w,int(fx2)),min(im_h,int(fy2))
            if(fx2-fx1)>=MIN_BOX_SIZE and(fy2-fy1)>=MIN_BOX_SIZE: 
                self.suggested_boxes_data[self.editing_suggestion_idx] = ([fx1,fy1,fx2,fy2], self.suggested_boxes_data[self.editing_suggestion_idx][1], self.suggested_boxes_data[self.editing_suggestion_idx][2])
            self._redraw(); return

        if self.is_moving_suggestion and self.editing_suggestion_idx is not None and self.move_start_img_coords and self.resize_suggestion_original_state_during_op:
            dx_m,dy_m=img_x-self.move_start_img_coords[0],img_y-self.move_start_img_coords[1]
            orig_sug_xyxy, sug_prompt, sug_score = self.resize_suggestion_original_state_during_op
            ox1,oy1,ox2,oy2 = orig_sug_xyxy
            bw,bh=ox2-ox1,oy2-oy1; nxm,nym=ox1+dx_m,oy1+dy_m
            w_i,h_i=self.base_img.size
            nxm,nym=max(0,min(nxm,w_i-bw)),max(0,min(nym,h_i-bh))
            self.suggested_boxes_data[self.editing_suggestion_idx] = ([int(nxm),int(nym),int(nxm+bw),int(nym+bh)], sug_prompt, sug_score)
            self._redraw(); return

        if self.is_alt_adjusting_edges and self.alt_l_pressed and self.alt_adjust_click_img_coords: [self.boxes.__setitem__(i,d) for i,d in self.alt_adjust_original_boxes_state.items() if 0<=i<len(self.boxes)]; self._perform_alt_edge_adjust(img_x,img_y); return
        if self.is_resizing_box and self.resize_box_idx is not None:
            ox1,oy1,ox2,oy2,lbls,cnf=self.resize_box_original_state; nx1,ny1,nx2,ny2=ox1,oy1,ox2,oy2; ht=self.resize_handle_type
            if 'n' in ht:ny1=img_y; 
            elif 's' in ht:ny2=img_y
            if 'w' in ht:nx1=img_x; 
            elif 'e' in ht:nx2=img_x
            fx1,fx2=min(nx1,nx2),max(nx1,nx2); fy1,fy2=min(ny1,ny2),max(ny1,ny2)
            if(fx2-fx1)<MIN_BOX_SIZE:fx1,fx2=(fx2-MIN_BOX_SIZE,fx2) if nx1==fx1 else(fx1,fx1+MIN_BOX_SIZE)
            if(fy2-fy1)<MIN_BOX_SIZE:fy1,fy2=(fy2-MIN_BOX_SIZE,fy2) if ny1==fy1 else(fy1,fy1+MIN_BOX_SIZE)
            im_w,im_h=self.base_img.size;fx1,fy1,fx2,fy2=max(0,int(fx1)),max(0,int(fy1)),min(im_w,int(fx2)),min(im_h,int(fy2))
            if(fx2-fx1)>=MIN_BOX_SIZE and(fy2-fy1)>=MIN_BOX_SIZE: self.boxes[self.resize_box_idx]=(fx1,fy1,fx2,fy2,lbls,cnf)
            self._redraw(); return
        if self.is_moving_boxes and self.move_start_img_coords:
            dx_m,dy_m=img_x-self.move_start_img_coords[0],img_y-self.move_start_img_coords[1]; w_i,h_i=self.base_img.size
            for i,orig_s in self.moved_boxes_initial_state.items():
                if not(0<=i<len(self.boxes)):continue
                ox1,oy1,ox2,oy2,lbls,cnf=orig_s; bw,bh=ox2-ox1,oy2-oy1; nxm,nym=ox1+dx_m,oy1+dy_m
                nxm,nym=max(0,min(nxm,w_i-bw)),max(0,min(nym,h_i-bh)); self.boxes[i]=(int(nxm),int(nym),int(nxm+bw),int(nym+bh),lbls,cnf)
            self._redraw(); return
        if self.draw_start:
            self.show_crosshair=True; self._update_crosshair(ev.x,ev.y)
            x0c,y0c=self._from_canvas(*self.draw_start); x1c,y1c=can_x,can_y
            if self.tmp_rect_id: self.canvas.delete(self.tmp_rect_id)
            self.tmp_rect_id=self.canvas.create_rectangle(x0c,y0c,x1c,y1c,outline=BOX_COLOR_DEFAULT,width=BOX_W_NORMAL,dash=(4,2),tags="temp_box"); return
    def _on_release(self,ev):
        if self.panning_mode and self.pan_start: self.pan_start=None; self._redraw(); self._update_cursor_and_crosshair(ev); return
        
        if self.is_resizing_suggestion:
            self.is_resizing_suggestion = False; self.resize_suggestion_handle_type = None
            self.resize_suggestion_start_img_coords = None; self.resize_suggestion_original_state_during_op = None
            self._redraw(); self._update_cursor_and_crosshair(ev); return
        
        if self.is_moving_suggestion:
            self.is_moving_suggestion = False; self.move_start_img_coords = None
            self.resize_suggestion_original_state_during_op = None # Reset this, as it was used for original state
            self._redraw(); self._update_cursor_and_crosshair(ev); return

        if self.is_alt_adjusting_edges:
            if self.alt_adjust_original_boxes_state:
                changed=[i for i,s in self.alt_adjust_original_boxes_state.items() if 0<=i<len(self.boxes) and self.boxes[i][:4]!=s[:4]]
                if changed: self._add_to_undo("alt_adjust_edges",{i:d for i,d in self.alt_adjust_original_boxes_state.items() if i in changed})
            self.is_alt_adjusting_edges=False; self.alt_adjust_click_img_coords=None; self.alt_adjust_original_boxes_state={}; self._redraw(); self._update_cls_panel(); self._update_cursor_and_crosshair(ev); return
        if self.is_resizing_box and self.resize_box_idx is not None:
            if self.resize_box_original_state and self.boxes[self.resize_box_idx][:4]!=self.resize_box_original_state[:4]: self._add_to_undo("resize_single",{"idx":self.resize_box_idx,"old_box_data":self.resize_box_original_state})
            self.is_resizing_box=False; self.resize_box_idx=None; self.resize_handle_type=None; self.resize_start_img_coords=None; self.resize_box_original_state=None; self.canvas.config(cursor=""); self._redraw(); self._update_cls_panel(); self._update_cursor_and_crosshair(ev); return
        if self.is_moving_boxes:
            if self.moved_boxes_initial_state and any(0<=i<len(self.boxes) and self.boxes[i][:4]!=s[:4] for i,s in self.moved_boxes_initial_state.items()): self._add_to_undo("move_multiple",dict(self.moved_boxes_initial_state))
            self.is_moving_boxes=False; self.move_start_img_coords=None; self.moved_boxes_initial_state={}; self._redraw(); self._update_cls_panel(); self._update_cursor_and_crosshair(ev); return
        if self.draw_start:
            if self.tmp_rect_id: self.canvas.delete(self.tmp_rect_id); self.tmp_rect_id=None
            x0,y0=self.draw_start; x1,y1=self._to_canvas(ev.x,ev.y); xmin,xmax=sorted((x0,x1)); ymin,ymax=sorted((y0,y1))
            w_i,h_i=self.base_img.size; xmin,ymin,xmax,ymax=max(0,int(xmin)),max(0,int(ymin)),min(w_i,int(xmax)),min(h_i,int(ymax))
            if (xmax-xmin)>=MIN_BOX_SIZE and (ymax-ymin)>=MIN_BOX_SIZE and math.sqrt((x1-x0)**2+(y1-y0)**2)>MIN_BOX_SIZE:
                new_idx=len(self.boxes); desc=self.class_names[0] if self.class_names else DEFAULT_LABEL
                self.boxes.append((xmin,ymin,xmax,ymax,[desc],-1)); self._add_to_undo("add",new_idx)
                self.active_ids={new_idx}; self._update_cls_panel(); self._update_box_count_display(); self._redraw()
                if self.editing_suggestion_idx is not None: self._cancel_edit_suggestion() # Cancel suggestion edit if new box drawn
            elif not self.active_ids: self._update_cls_panel(); self._redraw()
            self.draw_start=None; self._update_cursor_and_crosshair(ev); return
        self._update_group_button_state(); self._update_cursor_and_crosshair(ev)

    def _on_motion(self,ev):
        if self.show_crosshair and not self.draw_start: self._update_crosshair(ev.x,ev.y)
        
        # If any operation is active, clear hover states and return
        if self.panning_mode or self.draw_start or self.is_moving_boxes or self.is_resizing_box or \
           self.is_alt_adjusting_edges or self.is_moving_suggestion or self.is_resizing_suggestion:
            if self.hover_idx is not None: self.hover_idx=None; self._redraw()
            if self.current_hover_resize_handle: self.canvas.config(cursor=""); self.current_hover_resize_handle=None; self.current_hover_resize_box_idx=None
            if self.current_hover_resize_suggestion_handle: self.canvas.config(cursor=""); self.current_hover_resize_suggestion_handle=None; self.current_hover_resize_suggestion_idx=None
            if not self.draw_start and self.show_crosshair: self.show_crosshair=False; self._delete_crosshair(); self.canvas.config(cursor="")
            return

        can_x,can_y=ev.x,ev.y; img_x,img_y=self._to_canvas(can_x,can_y); new_cursor=""; 
        self.current_hover_resize_handle=None; self.current_hover_resize_box_idx=None
        self.current_hover_resize_suggestion_handle = None; self.current_hover_resize_suggestion_idx = None
        redraw_needed=False

        # Check for resize handles on the suggestion being edited
        if self.editing_suggestion_idx is not None:
            sug_handle = self._get_resize_handle_at_pos_for_suggestion(self.editing_suggestion_idx, can_x, can_y)
            if sug_handle:
                self.current_hover_resize_suggestion_handle = sug_handle
                self.current_hover_resize_suggestion_idx = self.editing_suggestion_idx
                if sug_handle in['n','s']:new_cursor="sb_v_double_arrow"; 
                elif sug_handle in['e','w']:new_cursor="sb_h_double_arrow"
                elif sug_handle in['nw','se']:new_cursor="size_nw_se"; 
                elif sug_handle in['ne','sw']:new_cursor="size_ne_sw"
                else: new_cursor="crosshair" # Default for unrecognized, though all should be covered
                redraw_needed = True
        
        # Check for resize handles on active regular box (only if not interacting with suggestion)
        if not new_cursor and len(self.active_ids)==1:
            active_idx=next(iter(self.active_ids)); handle=self._get_resize_handle_at_pos(active_idx,can_x,can_y)
            if handle:
                self.current_hover_resize_handle=handle; self.current_hover_resize_box_idx=active_idx
                if handle in['n','s']:new_cursor="sb_v_double_arrow"; 
                elif handle in['e','w']:new_cursor="sb_h_double_arrow"
                elif handle in['nw','se']:new_cursor="size_nw_se"; 
                elif handle in['ne','sw']:new_cursor="size_ne_sw"
                else: new_cursor="crosshair"
                redraw_needed = True
        
        found_hover_box=None
        if not new_cursor: # If no resize handle is hovered
            idxs_check=set(range(len(self.boxes))) if self.box_visible else self.active_ids
            if self.hover_idx is not None and 0<=self.hover_idx<len(self.boxes): idxs_check.add(self.hover_idx)
            for i in reversed(list(idxs_check)):
                if i>=len(self.boxes): continue
                x1b,y1b,x2b,y2b,_,_=self.boxes[i]
                if x1b<=img_x<=x2b and y1b<=img_y<=y2b: found_hover_box=i; break
        
        cur_sys_cursor=self.canvas.cget("cursor")
        if new_cursor: # A resize handle (box or suggestion) is hovered
            if cur_sys_cursor!=new_cursor: self.canvas.config(cursor=new_cursor)
            if self.hover_idx is not None: self.hover_idx=None; redraw_needed=True # Clear box hover if resizing
            if self.show_crosshair: self.show_crosshair=False; self._delete_crosshair()
        else: # No resize handle hovered
            if self._should_show_crosshair():
                if not self.show_crosshair or cur_sys_cursor!="none": self.show_crosshair=True; self.canvas.config(cursor="none")
                self._update_crosshair(ev.x,ev.y)
            else:
                if self.show_crosshair: self.show_crosshair=False; self._delete_crosshair()
                if cur_sys_cursor!="": self.canvas.config(cursor="") # Default cursor
            
            if found_hover_box!=self.hover_idx: self.hover_idx=found_hover_box; redraw_needed=True
        
        if redraw_needed: self._redraw()


    def _should_show_crosshair(self) -> bool:
        if self.panning_mode or self.is_resizing_box or self.is_moving_boxes or \
           self.is_alt_adjusting_edges or self.draw_start or \
           self.is_resizing_suggestion or self.is_moving_suggestion: return False
        
        if self.editing_suggestion_idx is not None:
            cur_x=self.canvas.winfo_pointerx()-self.canvas.winfo_rootx(); cur_y=self.canvas.winfo_pointery()-self.canvas.winfo_rooty()
            if self._get_resize_handle_at_pos_for_suggestion(self.editing_suggestion_idx, cur_x, cur_y): return False
            
        if len(self.active_ids)==1:
            active_idx=next(iter(self.active_ids)); cur_x=self.canvas.winfo_pointerx()-self.canvas.winfo_rootx(); cur_y=self.canvas.winfo_pointery()-self.canvas.winfo_rooty()
            if self._get_resize_handle_at_pos(active_idx, cur_x, cur_y): return False
        return True

    def _update_cursor_and_crosshair(self, event=None):
        if self.panning_mode or self.is_resizing_box or self.is_moving_boxes or self.is_alt_adjusting_edges or self.draw_start or self.is_resizing_suggestion or self.is_moving_suggestion:
            if self.show_crosshair: self.show_crosshair=False; self._delete_crosshair()
            # Specific cursors for these ops are set during the op or in _on_motion
            return
        
        current_cursor_set = False
        # Suggestion resize cursor
        if self.editing_suggestion_idx is not None and self.current_hover_resize_suggestion_handle:
            handle = self.current_hover_resize_suggestion_handle
            if handle in['n','s']: self.canvas.config(cursor="sb_v_double_arrow"); current_cursor_set = True
            elif handle in['e','w']: self.canvas.config(cursor="sb_h_double_arrow"); current_cursor_set = True
            elif handle in['nw','se']: self.canvas.config(cursor="size_nw_se"); current_cursor_set = True
            elif handle in['ne','sw']: self.canvas.config(cursor="size_ne_sw"); current_cursor_set = True
        
        # Box resize cursor (only if not suggestion resize)
        if not current_cursor_set and len(self.active_ids)==1 and self.current_hover_resize_handle:
            handle = self.current_hover_resize_handle
            if handle in['n','s']: self.canvas.config(cursor="sb_v_double_arrow"); current_cursor_set = True
            elif handle in['e','w']: self.canvas.config(cursor="sb_h_double_arrow"); current_cursor_set = True
            elif handle in['nw','se']: self.canvas.config(cursor="size_nw_se"); current_cursor_set = True
            elif handle in['ne','sw']: self.canvas.config(cursor="size_ne_sw"); current_cursor_set = True

        if self._should_show_crosshair():
            if not self.show_crosshair: self.show_crosshair=True
            if not current_cursor_set: self.canvas.config(cursor="none") # Crosshair cursor overrides default
            current_cursor_set = True # Consider crosshair as a "set" cursor
            if event: self._update_crosshair(event.x,event.y)
            else: 
                try:
                    cur_x=self.canvas.winfo_pointerx()-self.canvas.winfo_rootx(); cur_y=self.canvas.winfo_pointery()-self.canvas.winfo_rooty(); self._update_crosshair(cur_x,cur_y)
                except tk.TclError: pass # Widget might not be fully ready
        else:
            if self.show_crosshair: self.show_crosshair=False; self._delete_crosshair()
            if not current_cursor_set: self.canvas.config(cursor="") # Default arrow if no other cursor applies

    def _update_crosshair(self, canvas_x:int, canvas_y:int):
        if not self.show_crosshair: self._delete_crosshair(); return
        self.canvas.delete("crosshair"); cw=self.canvas.winfo_width(); ch=self.canvas.winfo_height()
        if cw==1 and ch==1: return 
        self.crosshair_h_line=self.canvas.create_line(0,canvas_y,cw,canvas_y,fill=BOX_COLOR_CROSSHAIR,width=1,dash=(2,4),tags="crosshair")
        self.crosshair_v_line=self.canvas.create_line(canvas_x,0,canvas_x,ch,fill=BOX_COLOR_CROSSHAIR,width=1,dash=(2,4),tags="crosshair")
        self.canvas.tag_lower("crosshair", "box") # Ensure crosshair is behind boxes and suggestions
    def _delete_crosshair(self): self.canvas.delete("crosshair"); self.crosshair_h_line=None; self.crosshair_v_line=None
    def _sel_start(self, ev):
        self.sel_start=self._to_canvas(ev.x,ev.y); self.sel_rect_id=None; self.sel_prev_active=set(self.active_ids)
        self.show_crosshair=False; self._delete_crosshair(); self.canvas.config(cursor="")
        if self.editing_suggestion_idx is not None: # Cancel suggestion editing on RMB click
            self._cancel_edit_suggestion()
    def _sel_drag(self, ev):
        if not hasattr(self,'sel_start') or self.sel_start is None: return
        x0c,y0c=self._from_canvas(*self.sel_start); x1c,y1c=ev.x,ev.y
        if self.sel_rect_id: self.canvas.delete(self.sel_rect_id)
        self.sel_rect_id=self.canvas.create_rectangle(x0c,y0c,x1c,y1c,outline="yellow",dash=(4,2),tags="selrect")
    def _sel_end(self, ev):
        if not hasattr(self,'sel_start') or self.sel_start is None: return
        if self.sel_rect_id: self.canvas.delete(self.sel_rect_id); self.sel_rect_id=None
        x0i,y0i=self.sel_start; x1i,y1i=self._to_canvas(ev.x,ev.y); xmin,xmax=sorted((x0i,x1i)); ymin,ymax=sorted((y0i,y1i)); new_sel=set()
        for i,(bx1,by1,bx2,by2,_,_) in enumerate(self.boxes):
            cx,cy=(bx1+bx2)/2,(by1+by2)/2
            if xmin<=cx<=xmax and ymin<=cy<=ymax: new_sel.add(i)
        if new_sel!=self.sel_prev_active: self._add_to_undo("select",self.sel_prev_active); self.active_ids=new_sel; self._update_cls_panel(); self._redraw() 
        elif abs(x0i-x1i)>MIN_BOX_SIZE or abs(y0i-y1i)>MIN_BOX_SIZE: self._update_cls_panel(); self._redraw()
        self.sel_start=None; self._update_group_button_state(); self._update_cursor_and_crosshair(ev)

    def _draw_resize_handles(self, box_idx: int):
        if not (0<=box_idx<len(self.boxes)): return
        x1i,y1i,x2i,y2i,_,_=self.boxes[box_idx]; cx1,cy1=self._from_canvas(x1i,y1i); cx2,cy2=self._from_canvas(x2i,y2i)
        hs=self.RESIZE_HANDLE_VISUAL_SIZE; hh=hs/2
        h_coords={'nw':(cx1,cy1),'n':(cx1+(cx2-cx1)/2,cy1),'ne':(cx2,cy1),'w':(cx1,cy1+(cy2-cy1)/2),'e':(cx2,cy1+(cy2-cy1)/2),'sw':(cx1,cy2),'s':(cx1+(cx2-cx1)/2,cy2),'se':(cx2,cy2)}
        for ht,(hx,hy) in h_coords.items(): 
            fillc="yellow" if self.current_hover_resize_handle==ht and self.current_hover_resize_box_idx==box_idx else "white"
            self.canvas.create_rectangle(hx-hh,hy-hh,hx+hh,hy+hh,fill=fillc,outline="black",tags=("box_label",f"handle_box_{box_idx}_{ht}"))

    def _draw_resize_handles_for_suggestion(self, suggestion_idx: int):
        if not (0 <= suggestion_idx < len(self.suggested_boxes_data)): return
        sug_xyxy, _, _ = self.suggested_boxes_data[suggestion_idx]
        x1i, y1i, x2i, y2i = sug_xyxy
        cx1, cy1 = self._from_canvas(x1i, y1i); cx2, cy2 = self._from_canvas(x2i, y2i)
        hs = self.RESIZE_HANDLE_VISUAL_SIZE; hh = hs / 2
        h_coords = {'nw': (cx1, cy1), 'n': (cx1 + (cx2 - cx1) / 2, cy1), 'ne': (cx2, cy1),
                    'w': (cx1, cy1 + (cy2 - cy1) / 2), 'e': (cx2, cy1 + (cy2 - cy1) / 2),
                    'sw': (cx1, cy2), 's': (cx1 + (cx2 - cx1) / 2, cy2), 'se': (cx2, cy2)}
        for ht, (hx, hy) in h_coords.items():
            fillc = "yellow" if self.current_hover_resize_suggestion_handle == ht and self.current_hover_resize_suggestion_idx == suggestion_idx else "white"
            self.canvas.create_rectangle(hx - hh, hy - hh, hx + hh, hy + hh, fill=fillc, outline="black", tags=("box_label", f"handle_sug_{suggestion_idx}_{ht}"))


    def _redraw(self):
        if not hasattr(self,'base_img') or not self.base_img: return
        try: 
            w_s,h_s=max(1,int(self.base_img.width*self.scale)),max(1,int(self.base_img.height*self.scale))
            rs_m=Image.Resampling.NEAREST if self.scale>1.5 else Image.Resampling.LANCZOS
            disp_img=self.base_img.resize((w_s,h_s),rs_m); self.tk_img=ImageTk.PhotoImage(disp_img)
        except Exception as e: print(f"Error redraw resize: {e}"); return
        self.canvas.itemconfigure(self.img_id,image=self.tk_img); self.canvas.coords(self.img_id,int(self.offset[0]),int(self.offset[1]))
        self.canvas.delete("box"); self.canvas.delete("box_label") 
        show_idxs=set(range(len(self.boxes))) if self.box_visible else self.active_ids
        if not self.box_visible and self.hover_idx is not None and 0<=self.hover_idx<len(self.boxes): show_idxs.add(self.hover_idx)
        for i in show_idxs:
            if i>=len(self.boxes): continue
            x1,y1,x2,y2,descs,_=self.boxes[i]; cx1,cy1=self._from_canvas(x1,y1); cx2,cy2=self._from_canvas(x2,y2)
            is_act=i in self.active_ids; is_hov=i==self.hover_idx; clr=BOX_COLOR_ACTIVE if is_act else BOX_COLOR_DEFAULT; wid=BOX_W_ACTIVE if is_act or is_hov else BOX_W_NORMAL
            try:
                if cx1>=cx2 or cy1>=cy2: continue
                self.canvas.create_rectangle(cx1,cy1,cx2,cy2,outline=clr,width=wid,tags=("box",f"box_{i}"))
                if descs and isinstance(descs,list) and descs[0] and (cx2-cx1)>10 and (cy2-cy1)>10:
                    d_txt=str(descs[0]); tx,ty=cx1+3,cy1+3; fs,fst,ff=8,"bold","Segoe UI"; fnt=tkFont.Font(family=ff,size=fs,weight=fst)
                    tw=fnt.measure(d_txt); th=fnt.metrics("linespace"); max_dw=(cx2-cx1)-6
                    if tw>max_dw: avg_cw=fnt.measure("a") if fnt.measure("a")>0 else fs/1.8; max_c=int(max_dw/avg_cw) if avg_cw>0 else 10; d_txt=d_txt[:max_c-3]+"..." if max_c>3 else d_txt[:max_c]; tw=fnt.measure(d_txt)
                    self.canvas.create_rectangle(tx-2,ty-1,tx+tw+2,ty+th,fill="black",outline="",tags=("box_label",f"label_box_{i}"))
                    self.canvas.create_text(tx,ty,text=d_txt,anchor="nw",fill=clr,font=fnt,tags=("box_label",f"label_box_{i}"))
            except Exception as e: print(f"Error dibujando caja {i}: {e}")
        
        for i,(xyxy_sug,prompt_sug,score_sug) in enumerate(self.suggested_boxes_data):
            x1s,y1s,x2s,y2s=xyxy_sug; cx1s,cy1s=self._from_canvas(x1s,y1s); cx2s,cy2s=self._from_canvas(x2s,y2s)
            is_editing_this_sug = (i == self.editing_suggestion_idx)
            sug_color = BOX_COLOR_SUGGESTION_EDIT if is_editing_this_sug else BOX_COLOR_SUGGESTION
            sug_dash = () if is_editing_this_sug else (4,4) # Solid if editing, dashed otherwise
            sug_width = BOX_W_ACTIVE if is_editing_this_sug else BOX_W_NORMAL

            try:
                if cx1s>=cx2s or cy1s>=cy2s: continue
                self.canvas.create_rectangle(cx1s,cy1s,cx2s,cy2s,outline=sug_color,width=sug_width,dash=sug_dash,tags=("box",f"suggestion_box_{i}"))
                if prompt_sug and(cx2s-cx1s)>10 and(cy2s-cy1s)>10:
                    sug_txt=f"S: {prompt_sug[:15]} ({score_sug:.2f})"; txs,tys=cx1s+3,cy1s-12; 
                    if tys<0: tys=cy1s+3
                    fs,fst,ff=7,"normal","Segoe UI"; fnt_sug=tkFont.Font(family=ff,size=fs,weight=fst); tws=fnt_sug.measure(sug_txt); ths=fnt_sug.metrics("linespace")
                    self.canvas.create_rectangle(txs-2,tys-1,txs+tws+2,tys+ths,fill="black",outline="",tags=("box_label",f"suggestion_label_{i}"))
                    self.canvas.create_text(txs,tys,text=sug_txt,anchor="nw",fill=sug_color,font=fnt_sug,tags=("box_label",f"suggestion_label_{i}"))
                if is_editing_this_sug:
                    self._draw_resize_handles_for_suggestion(i)
            except Exception as e: print(f"Error dibujando sugerencia {i}: {e}")

        if len(self.active_ids)==1 and self.editing_suggestion_idx is None: # Only draw box handles if not editing a suggestion
            self._draw_resize_handles(next(iter(self.active_ids)))

    def _refresh_class_list(self,init=False):
        self.class_names=list(self.main_app.get_class_names()); self.class2id={name:i for i,name in enumerate(self.class_names)}
        cur_sel_idx=self.lst_cls.curselection(); cur_sel_val=self.lst_cls.get(cur_sel_idx[0]) if cur_sel_idx else None
        self.lst_cls.delete(0,"end"); search_term=self.search_var.get().lower(); new_sel_idx=-1
        for desc_name in self.class_names:
            if search_term in desc_name.lower(): self.lst_cls.insert("end",desc_name)
            if desc_name==cur_sel_val: new_sel_idx=self.lst_cls.size()-1
        if not init and new_sel_idx!=-1: self.lst_cls.selection_set(new_sel_idx); self.lst_cls.activate(new_sel_idx)
    def _on_cls_change(self, *_):
        sel_idx=self.lst_cls.curselection()
        if not sel_idx or not self.active_ids: return
        if self.editing_suggestion_idx is not None: # Cannot change class while editing suggestion
             messagebox.showinfo("Info", "Finaliza la edición de la sugerencia antes de cambiar la descripción de la caja.", parent=self)
             self.lst_cls.selection_clear(0, tk.END) # Deselect
             return

        sel_desc=self.lst_cls.get(sel_idx[0])
        if sel_desc not in self.class_names: return
        new_descs_list=[sel_desc]; old_descs_map={}; changed=False
        valid_ids=list(self.active_ids)
        for idx in valid_ids:
            if 0<=idx<len(self.boxes):
                cur_box_descs=self.boxes[idx][4]; old_descs_map[idx]=list(cur_box_descs)
                if not cur_box_descs or cur_box_descs[0]!=sel_desc: changed=True
            else: self.active_ids.discard(idx)
        if changed:
            self._add_to_undo("cls",old_descs_map)
            for idx in self.active_ids:
                if 0<=idx<len(self.boxes): x1,y1,x2,y2,_,cnf=self.boxes[idx]; self.boxes[idx]=(x1,y1,x2,y2,list(new_descs_list),cnf)
            self._redraw(); self._update_cls_panel()
    def _add_manual_description(self):
        desc=simpledialog.askstring("Nueva Descripción","Introduce la descripción:",parent=self)
        if desc and desc.strip():
            norm_desc=self.main_app.add_description_if_new(desc)
            if norm_desc and self.active_ids and self.editing_suggestion_idx is None and \
               messagebox.askyesno("Aplicar",f"Aplicar '{norm_desc}' a {len(self.active_ids)} caja(s)?",parent=self):
                try:
                    idx_in_list=self.lst_cls.get(0,"end").index(norm_desc)
                    self.lst_cls.selection_clear(0,"end"); self.lst_cls.selection_set(idx_in_list); self.lst_cls.activate(idx_in_list); self._on_cls_change()
                except ValueError: pass
            elif norm_desc: messagebox.showinfo("Añadida",f"Descripción '{norm_desc}' añadida.",parent=self)
    def _save_class_names(self): self.main_app.save_class_names()
    def _update_cls_panel(self):
        if self.editing_suggestion_idx is not None:
            sug_xyxy, sug_prompt, sug_score = self.suggested_boxes_data[self.editing_suggestion_idx]
            self.lbl_info.config(text=f"Editando Sugerencia:\n{sug_prompt} ({sug_score:.2f})\nCaja: {sug_xyxy}")
            self.lst_cls.selection_clear(0,"end") # No selection for normal boxes
            self._update_group_button_state()
            return

        valid_ids={idx for idx in self.active_ids if 0<=idx<len(self.boxes)}; num_sel=len(valid_ids); common_descs=set()
        if valid_ids:
            first_idx=next(iter(valid_ids))
            if 0<=first_idx<len(self.boxes):
                first_ds=self.boxes[first_idx][4]; common_descs=set(d for d in first_ds if d) if first_ds else set()
                for idx in valid_ids:
                    if idx==first_idx: continue
                    if 0<=idx<len(self.boxes):
                        box_ds=self.boxes[idx][4]
                        if isinstance(box_ds,list): common_descs.intersection_update(d for d in box_ds if d)
                        else: common_descs=set(); break
                    else: common_descs=set(); break
            else: common_descs=set()
        self.lst_cls.selection_clear(0,"end"); listbox_items=list(self.lst_cls.get(0,"end"))
        if len(common_descs)==1:
            the_desc=list(common_descs)[0]
            try: idx_in_list=listbox_items.index(the_desc); self.lst_cls.selection_set(idx_in_list); self.lst_cls.activate(idx_in_list); self.lst_cls.see(idx_in_list)
            except ValueError: pass
        if not valid_ids: self.lbl_info.config(text="Sin selección")
        else:
            desc_txt_disp="Descripciones Mixtas"
            if len(common_descs)==1: desc_txt_disp=f"Desc: {list(common_descs)[0]}"
            elif not common_descs and num_sel==1:
                the_id=next(iter(valid_ids))
                if 0<=the_id<len(self.boxes):
                    cur_descs=self.boxes[the_id][4]
                    if not cur_descs or not cur_descs[0] or cur_descs[0]==DEFAULT_LABEL: desc_txt_disp=f"Sin descripción/{DEFAULT_LABEL}"
            elif not common_descs: desc_txt_disp="Sin descripciones comunes"
            self.lbl_info.config(text=f"{num_sel} caja(s) sel.\n{desc_txt_disp}")
        if self.active_ids!=valid_ids: self.active_ids=valid_ids
        self._update_group_button_state()
    def _delete_sel(self,event=None):
        if self.editing_suggestion_idx is not None: # Cannot delete normal boxes if editing suggestion
            messagebox.showinfo("Info", "Finaliza la edición de la sugerencia antes de borrar cajas.", parent=self)
            return
        if not self.active_ids: return
        removed=[]; indices_to_del=sorted([idx for idx in self.active_ids if 0<=idx<len(self.boxes)],reverse=True)
        if not indices_to_del: return
        for idx in indices_to_del:
            try: removed.append((idx,self.boxes.pop(idx)))
            except IndexError: pass
        if removed: self._add_to_undo("del",list(reversed(removed)))
        self.active_ids.clear(); self._update_cls_panel(); self._update_box_count_display(); self._redraw(); self._update_group_button_state()
    def _update_group_button_state(self):
        can_group = len(self.active_ids)>1 and self.editing_suggestion_idx is None
        if hasattr(self,'btn_group'): self.btn_group.config(state="normal" if can_group else "disabled")
    def _group_selected_boxes(self,event=None):
        if self.editing_suggestion_idx is not None: return
        if len(self.active_ids)<=1: messagebox.showinfo("Agrupar","Selecciona 2+ cajas."); return
        min_x1,min_y1,max_x2,max_y2=float('inf'),float('inf'),float('-inf'),float('-inf')
        old_boxes=[]; valid_indices=sorted([i for i in self.active_ids if 0<=i<len(self.boxes)],reverse=True)
        first_desc=[DEFAULT_LABEL]
        if valid_indices and self.boxes[valid_indices[-1]][4]: first_desc=[self.boxes[valid_indices[-1]][4][0]]
        for idx in valid_indices:
            try: x1,y1,x2,y2,_,_=self.boxes[idx]; old_boxes.append((idx,self.boxes[idx])); min_x1=min(min_x1,x1); min_y1=min(min_y1,y1); max_x2=max(max_x2,x2); max_y2=max(max_y2,y2)
            except IndexError: continue
        if not old_boxes: return
        for idx in valid_indices: 
            try: del self.boxes[idx]
            except IndexError: pass
        new_box=(min_x1,min_y1,max_x2,max_y2,first_desc,-1); self.boxes.append(new_box); new_idx=len(self.boxes)-1
        self._add_to_undo("group",{'new_box_idx':new_idx,'old_boxes_data':sorted(old_boxes,key=lambda item:item[0])})
        self.active_ids={new_idx}; self._update_cls_panel(); self._update_box_count_display(); self._redraw(); self._update_group_button_state()
    def _enlarge_selected_boxes(self,increment=BOX_ENLARGE_INCREMENT):
        if self.editing_suggestion_idx is not None: return
        if not self.active_ids or not self.boxes: return
        old_map={}; img_w,img_h=self.base_img.size; changed=[]
        for idx in self.active_ids:
            if 0<=idx<len(self.boxes):
                old_map[idx]=self.boxes[idx]; x1,y1,x2,y2,lbls,cnf=self.boxes[idx]; cw,ch=x2-x1,y2-y1; cx,cy=x1+cw/2,y1+ch/2
                nw,nh=max(MIN_BOX_SIZE,cw+increment),max(MIN_BOX_SIZE,ch+increment)
                nx1,ny1,nx2,ny2=int(cx-nw/2),int(cy-nh/2),int(cx+nw/2),int(cy+nh/2)
                nx1,ny1,nx2,ny2=max(0,nx1),max(0,ny1),min(img_w,nx2),min(img_h,ny2)
                if(nx2-nx1)>=MIN_BOX_SIZE and(ny2-ny1)>=MIN_BOX_SIZE and(nx1,ny1,nx2,ny2)!=(x1,y1,x2,y2): self.boxes[idx]=(nx1,ny1,nx2,ny2,lbls,cnf); changed.append(idx)
        if changed: undo_data={i:d for i,d in old_map.items() if i in changed};
        if changed and undo_data: self._add_to_undo("resize_multiple",undo_data)
        self._redraw(); self._update_cls_panel()

    def _handle_undo_approve_single_suggestion(self, data: Dict[str, Any]) -> bool:
        added_box_idx = data['added_box_idx']
        original_suggestion_data_tuple = data['original_suggestion_data_tuple']
        original_list_idx = data['original_suggestion_list_idx']

        box_removed_for_redo = None
        if 0 <= added_box_idx < len(self.boxes):
            box_removed_for_redo = self.boxes.pop(added_box_idx)
        
        self.suggested_boxes_data.insert(original_list_idx, original_suggestion_data_tuple)
        
        self.active_ids.clear() # Clear selection of normal boxes
        # Optionally, re-select the suggestion for editing:
        # self.editing_suggestion_idx = original_list_idx
        # self.editing_suggestion_original_data = copy.deepcopy(original_suggestion_data_tuple)
        # For simplicity, just restore lists for now. User can re-select if needed.
        self.editing_suggestion_idx = None
        self.editing_suggestion_original_data = None


        if box_removed_for_redo:
            self.redo_stack.append(("approve_single_suggestion", {
                "box_to_re_add_tuple": box_removed_for_redo,
                "suggestion_to_remove_again_original_data": original_suggestion_data_tuple, # This is the data as it was BEFORE edit
                "suggestion_list_idx_to_remove_from": original_list_idx
            }))
        self._update_suggestion_buttons_state()
        return False # No global class list refresh needed

    def _handle_redo_approve_single_suggestion(self, data: Dict[str, Any]) -> bool:
        box_to_re_add_tuple = data['box_to_re_add_tuple'] # This is (x1,y1,x2,y2, [desc], score)
        suggestion_data_to_remove_tuple = data['suggestion_to_remove_again_original_data'] # This is (xyxy, prompt, score) of the original suggestion
        sug_list_idx_to_remove_from = data['suggestion_list_idx_to_remove_from']

        self.boxes.append(box_to_re_add_tuple)
        new_box_idx = len(self.boxes) - 1
        
        if 0 <= sug_list_idx_to_remove_from < len(self.suggested_boxes_data):
             # Check if the item to remove is indeed what we expect, to avoid errors if list changed unexpectedly
            if self.suggested_boxes_data[sug_list_idx_to_remove_from] == suggestion_data_to_remove_tuple:
                 self.suggested_boxes_data.pop(sug_list_idx_to_remove_from)
            else:
                # Fallback: try to find and remove by value if index is wrong but value exists
                try:
                    self.suggested_boxes_data.remove(suggestion_data_to_remove_tuple)
                except ValueError:
                    print(f"WARN (Redo Approve Single): Suggestion at index {sug_list_idx_to_remove_from} changed or not found by value.")


        self.active_ids = {new_box_idx}
        self.editing_suggestion_idx = None
        self.editing_suggestion_original_data = None
        
        self._add_to_undo("approve_single_suggestion", {
            "added_box_idx": new_box_idx,
            "original_suggestion_data_tuple": suggestion_data_to_remove_tuple,
            "original_suggestion_list_idx": sug_list_idx_to_remove_from 
        })
        self._update_suggestion_buttons_state()
        return False # No global class list refresh

    def _handle_undo_group(self, data: Dict[str, Any]) -> bool:
        new_idx = data['new_box_idx']; old_data = data['old_boxes_data']; current_box_for_redo = None
        if 0 <= new_idx < len(self.boxes):
            current_box_for_redo = self.boxes[new_idx]
            try:
                del self.boxes[new_idx]
            except IndexError:
                pass
        restored = set(); [self.boxes.insert(orig_idx, box_data) or restored.add(orig_idx) for orig_idx, box_data in old_data]
        self.active_ids = restored; self.redo_stack.append(("group", {'new_box_idx': new_idx, 'old_boxes_data': old_data, 'current_grouped_box': current_box_for_redo})); return False
    def _handle_undo_del(self, data: List[Tuple[int, Any]]) -> bool:
        restored=set(); [self.boxes.insert(idx, box_data) or restored.add(idx) for idx, box_data in data]
        self.active_ids=restored; self.redo_stack.append(("del", data)); return False
    def _handle_undo_add(self, data: int) -> bool:
        idx_rm=data; box_removed_for_redo = None
        if 0 <= idx_rm < len(self.boxes): box_removed_for_redo = self.boxes.pop(idx_rm)
        self.active_ids={i if i < idx_rm else i-1 for i in self.active_ids if i!=idx_rm}
        if box_removed_for_redo: self.redo_stack.append(("add", {'idx': idx_rm, 'box_data': box_removed_for_redo})); return False
    def _handle_undo_cls(self, data: Dict[int, List[str]]) -> bool:
        current_labels_for_redo = {}; restored_ids = set(); max_idx=len(self.boxes)-1
        for idx, old_descriptions in data.items():
             if 0 <= idx <= max_idx: 
                 x1,y1,x2,y2,current_descs,conf=self.boxes[idx]; current_labels_for_redo[idx] = list(current_descs)
                 self.boxes[idx]=(x1,y1,x2,y2,list(old_descriptions),conf); restored_ids.add(idx)
        self.active_ids=restored_ids; self.redo_stack.append(("cls", current_labels_for_redo)); return False
    def _handle_undo_select(self, data: set[int]) -> bool:
        current_selection_for_redo = set(self.active_ids); max_idx=len(self.boxes)-1; self.active_ids={idx for idx in data if 0<=idx<=max_idx}
        self.redo_stack.append(("select", current_selection_for_redo)); return False
    def _handle_undo_add_cls(self, _: Dict[str, Any]) -> bool: return True 
    def _handle_undo_gemini(self, data: Dict[str, Any]) -> bool:
        indices=data["indices"]; old_map=data["old_labels"]; current_labels_for_redo = {}
        restored=set(); max_idx=len(self.boxes)-1
        for idx in indices:
            if idx in old_map and 0 <= idx <= max_idx: 
                x1,y1,x2,y2,current_descs,conf=self.boxes[idx]; current_labels_for_redo[idx] = list(current_descs)
                self.boxes[idx]=(x1,y1,x2,y2,list(old_map[idx]),conf); restored.add(idx)
        self.active_ids=restored; self.redo_stack.append(("gemini", {"indices": indices, "old_labels": current_labels_for_redo})); return False
    def _handle_undo_pipeline_replace(self, data: List[Tuple[int, Any]]) -> bool:
        current_boxes_for_redo = [(i,b) for i,b in enumerate(self.boxes)]
        self.boxes=[bd for _,bd in sorted(data,key=lambda x: x[0])]; self.active_ids.clear()
        self.redo_stack.append(("pipeline_replace", current_boxes_for_redo)); return False
    def _handle_undo_resize_multiple(self, data: Dict[int, Tuple[int,int,int,int,List[str],int]]) -> bool:
        current_boxes_for_redo = {}; restored=set(); max_idx=len(self.boxes)-1
        for idx, old_data in data.items():
            if 0<=idx<=max_idx: current_boxes_for_redo[idx] = self.boxes[idx]; self.boxes[idx]=old_data; restored.add(idx)
        self.active_ids=restored; self.redo_stack.append(("resize_multiple", current_boxes_for_redo)); return False
    def _handle_undo_move_multiple(self, data: Dict[int, Tuple[int,int,int,int,List[str],int]]) -> bool:
        current_boxes_for_redo = {}; restored=set(); max_idx=len(self.boxes)-1
        for idx, old_data in data.items():
            if 0<=idx<=max_idx: current_boxes_for_redo[idx] = self.boxes[idx]; self.boxes[idx]=old_data; restored.add(idx)
        self.active_ids=restored; self.redo_stack.append(("move_multiple", current_boxes_for_redo)); return False
    def _handle_undo_resize_single(self, data: Dict[str, Any]) -> bool:
        idx=data["idx"]; old_data=data["old_box_data"]; current_box_for_redo = None
        if 0<=idx<len(self.boxes): current_box_for_redo = self.boxes[idx]; self.boxes[idx]=old_data; self.active_ids={idx}
        else: self.active_ids.clear()
        if current_box_for_redo: self.redo_stack.append(("resize_single", {"idx": idx, "old_box_data": current_box_for_redo})); return False
    def _handle_undo_alt_adjust_edges(self, data: Dict[int, Tuple[int,int,int,int,List[str],int]]) -> bool:
        current_boxes_for_redo = {}; restored=set(); max_idx=len(self.boxes)-1
        for idx, old_data in data.items():
            if 0<=idx<=max_idx: current_boxes_for_redo[idx] = self.boxes[idx]; self.boxes[idx]=old_data; restored.add(idx)
        self.active_ids=restored; self.redo_stack.append(("alt_adjust_edges", current_boxes_for_redo)); return False
    def _handle_undo_approve_suggestions(self, data: Dict[str, Any]) -> bool:
        num_to_remove = data['num_added']; original_len_before_approve = data['original_len']
        removed_boxes_for_redo_data = []
        if num_to_remove > 0 and len(self.boxes) >= num_to_remove:
            start_index_of_approved = original_len_before_approve
            for _ in range(num_to_remove):
                if start_index_of_approved < len(self.boxes):
                    box_data_to_save = self.boxes[start_index_of_approved]
                    removed_boxes_for_redo_data.append((list(box_data_to_save[:4]), box_data_to_save[4][0], box_data_to_save[5]))
                    del self.boxes[start_index_of_approved]
                else: break
        self.active_ids.clear(); self.redo_stack.append(("approve_suggestions", {"approved_boxes_data": removed_boxes_for_redo_data})); return False
    def _undo(self):
        if not self.undo_stack: messagebox.showinfo("Undo", "Nada que deshacer.", parent=self); return
        action, data = self.undo_stack.pop(); needs_redraw=True; needs_update_panel=True; needs_refresh_list=False
        undo_handlers={
            "del":self._handle_undo_del, "add":self._handle_undo_add, 
            "cls":self._handle_undo_cls, "select":self._handle_undo_select, 
            "add_cls":self._handle_undo_add_cls, "gemini":self._handle_undo_gemini, 
            "group":self._handle_undo_group, "pipeline_replace":self._handle_undo_pipeline_replace, 
            "resize_multiple":self._handle_undo_resize_multiple, 
            "move_multiple":self._handle_undo_move_multiple, 
            "resize_single":self._handle_undo_resize_single, 
            "alt_adjust_edges":self._handle_undo_alt_adjust_edges, 
            "approve_suggestions": self._handle_undo_approve_suggestions,
            "approve_single_suggestion": self._handle_undo_approve_single_suggestion
        }
        try:
            handler=undo_handlers.get(action)
            if handler: needs_refresh_list = handler(data)
            else: self.redo_stack.append((action,data))
        except Exception as e: self.redo_stack.append((action,data)); traceback.print_exc()
        if needs_refresh_list: self.main_app.refresh_class_lists_in_tabs()
        else: self._refresh_class_list() 
        if needs_update_panel: self._update_cls_panel() 
        self._update_box_count_display();
        self._update_suggestion_buttons_state()
        if needs_redraw: self._redraw()
    def _handle_redo_group(self, data: Dict[str, Any]) -> bool:
        new_idx_to_restore=data['new_box_idx']; old_boxes_to_remove_again=data['old_boxes_data']; grouped_box_to_re_add=data['current_grouped_box']
        indices_to_delete=sorted([idx for idx,_ in old_boxes_to_remove_again],reverse=True)
        for idx in indices_to_delete:
            if 0<=idx<len(self.boxes): 
                try: del self.boxes[idx]
                except IndexError: pass
        if grouped_box_to_re_add: self.boxes.append(grouped_box_to_re_add); self.active_ids={len(self.boxes)-1}
        self._add_to_undo("group",data); return False
    def _handle_redo_del(self, data: List[Tuple[int, Any]]) -> bool:
        indices_to_delete = sorted([idx for idx, _ in data], reverse=True); removed_for_next_undo = []
        for idx in indices_to_delete:
            if 0 <= idx < len(self.boxes): removed_for_next_undo.append((idx, self.boxes.pop(idx)))
        self.active_ids.clear(); self._add_to_undo("del", list(reversed(removed_for_next_undo))); return False
    def _handle_redo_add(self, data: Dict[str, Any]) -> bool:
        idx_insert = data['idx']; box_to_re_add = data['box_data']
        self.boxes.insert(idx_insert, box_to_re_add); self.active_ids = {idx_insert}
        self._add_to_undo("add", idx_insert); return False
    def _handle_redo_cls(self, data: Dict[int, List[str]]) -> bool:
        old_labels_for_next_undo = {}; restored_ids = set(); max_idx = len(self.boxes) - 1
        for idx, new_labels in data.items():
            if 0 <= idx <= max_idx:
                x1,y1,x2,y2,current_labels,conf = self.boxes[idx]; old_labels_for_next_undo[idx] = list(current_labels)
                self.boxes[idx] = (x1,y1,x2,y2,list(new_labels),conf); restored_ids.add(idx)
        self.active_ids = restored_ids; self._add_to_undo("cls", old_labels_for_next_undo); return False
    def _handle_redo_select(self, data: set[int]) -> bool:
        old_selection_for_next_undo = set(self.active_ids); max_idx = len(self.boxes) - 1
        self.active_ids = {idx for idx in data if 0 <= idx <= max_idx}
        self._add_to_undo("select", old_selection_for_next_undo); return False
    def _handle_redo_gemini(self, data: Dict[str, Any]) -> bool:
        indices=data["indices"]; labels_to_restore=data["old_labels"]; old_labels_for_next_undo = {}
        restored=set(); max_idx=len(self.boxes)-1
        for idx in indices:
            if idx in labels_to_restore and 0 <= idx <= max_idx: 
                x1,y1,x2,y2,current_descs,conf=self.boxes[idx]; old_labels_for_next_undo[idx] = list(current_descs)
                self.boxes[idx]=(x1,y1,x2,y2,list(labels_to_restore[idx]),conf); restored.add(idx)
        self.active_ids=restored; self._add_to_undo("gemini", {"indices": indices, "old_labels": old_labels_for_next_undo}); return False
    def _handle_redo_pipeline_replace(self, data: List[Tuple[int, Any]]) -> bool:
        old_boxes_for_next_undo = [(i,b) for i,b in enumerate(self.boxes)]
        self.boxes=[bd for _,bd in sorted(data,key=lambda x: x[0])]; self.active_ids.clear()
        self._add_to_undo("pipeline_replace", old_boxes_for_next_undo); return False
    def _handle_redo_resize_multiple(self, data: Dict[int, Tuple[int,int,int,int,List[str],int]]) -> bool:
        old_boxes_for_next_undo = {}; restored=set(); max_idx=len(self.boxes)-1
        for idx, new_data in data.items():
            if 0<=idx<=max_idx: old_boxes_for_next_undo[idx] = self.boxes[idx]; self.boxes[idx]=new_data; restored.add(idx)
        self.active_ids=restored; self._add_to_undo("resize_multiple", old_boxes_for_next_undo); return False
    def _handle_redo_move_multiple(self, data: Dict[int, Tuple[int,int,int,int,List[str],int]]) -> bool:
        old_boxes_for_next_undo = {}; restored=set(); max_idx=len(self.boxes)-1
        for idx, new_data in data.items():
            if 0<=idx<=max_idx: old_boxes_for_next_undo[idx] = self.boxes[idx]; self.boxes[idx]=new_data; restored.add(idx)
        self.active_ids=restored; self._add_to_undo("move_multiple", old_boxes_for_next_undo); return False
    def _handle_redo_resize_single(self, data: Dict[str, Any]) -> bool:
        idx=data["idx"]; new_data=data["old_box_data"]; old_box_for_next_undo = None
        if 0<=idx<len(self.boxes): old_box_for_next_undo = self.boxes[idx]; self.boxes[idx]=new_data; self.active_ids={idx}
        else: self.active_ids.clear()
        if old_box_for_next_undo: self._add_to_undo("resize_single", {"idx": idx, "old_box_data": old_box_for_next_undo}); return False
    def _handle_redo_alt_adjust_edges(self, data: Dict[int, Tuple[int,int,int,int,List[str],int]]) -> bool:
        old_boxes_for_next_undo = {}; restored=set(); max_idx=len(self.boxes)-1
        for idx, new_data in data.items():
            if 0<=idx<=max_idx: old_boxes_for_next_undo[idx] = self.boxes[idx]; self.boxes[idx]=new_data; restored.add(idx)
        self.active_ids=restored; self._add_to_undo("alt_adjust_edges", old_boxes_for_next_undo); return False
    def _handle_redo_approve_suggestions(self, data: Dict[str, Any]) -> bool:
        boxes_to_re_add = data.get("approved_boxes_data", []); num_re_added = 0; original_len_for_next_undo = len(self.boxes)
        for (x1,y1,x2,y2),prompt_text,score in boxes_to_re_add: self.boxes.append((x1,y1,x2,y2,[prompt_text],score)); num_re_added+=1
        if num_re_added > 0: self._add_to_undo("approve_suggestions", {'num_added': num_re_added, 'original_len': original_len_for_next_undo})
        self.active_ids.clear(); return False
    def _redo(self):
        if not self.redo_stack: messagebox.showinfo("Redo", "Nada que rehacer.", parent=self); return
        action, data = self.redo_stack.pop(); needs_redraw=True; needs_update_panel=True; needs_refresh_list=False
        redo_handlers={
            "del":self._handle_redo_del, "add":self._handle_redo_add, 
            "cls":self._handle_redo_cls, "select":self._handle_redo_select, 
            "gemini":self._handle_redo_gemini, "group":self._handle_redo_group, 
            "pipeline_replace":self._handle_redo_pipeline_replace, 
            "resize_multiple":self._handle_redo_resize_multiple, 
            "move_multiple":self._handle_redo_move_multiple, 
            "resize_single":self._handle_redo_resize_single, 
            "alt_adjust_edges":self._handle_redo_alt_adjust_edges, 
            "approve_suggestions":self._handle_redo_approve_suggestions,
            "approve_single_suggestion": self._handle_redo_approve_single_suggestion
        }
        try:
            handler=redo_handlers.get(action)
            if handler: needs_refresh_list = handler(data)
            else: self.undo_stack.append((action,data))
        except Exception as e: self.undo_stack.append((action,data)); traceback.print_exc()
        if needs_refresh_list: self.main_app.refresh_class_lists_in_tabs()
        else: self._refresh_class_list()
        if needs_update_panel: self._update_cls_panel()
        self._update_box_count_display();
        self._update_suggestion_buttons_state()
        if needs_redraw: self._redraw()
    def _toggle_boxes(self): self.box_visible = not self.box_visible; self.hover_idx = None; self._redraw()
    
    def _yoloe_visual_prompt(self):
        if self.editing_suggestion_idx is not None:
            messagebox.showinfo("YOLOE", "Finaliza la edición de la sugerencia actual antes de generar nuevas.", parent=self)
            return
        if not self.active_ids: 
            messagebox.showinfo("Sugerencias YOLOE", "Selecciona una o más cajas de ejemplo (prompts visuales) primero.", parent=self)
            return
        
        yoloe_model = self.main_app.loaded_yolo_model
        if not yoloe_model or not ULTRALYTICS_AVAILABLE: 
            messagebox.showerror("Error YOLOE", "Modelo YOLOE (principal) no cargado o librerías no disponibles.", parent=self)
            return

        if not YOLOVP_PREDICTORS_AVAILABLE or YOLOEVPSegPredictor is None: 
            messagebox.showerror("Error YOLOE VP","Predictor YOLOE VP no disponible. Revisa la importación de predict_vp.py.",parent=self)
            if hasattr(self, 'btn_yoloe_visual_prompt'): self.btn_yoloe_visual_prompt.config(state="disabled")
            return

        selected_predictor_class = YOLOEVPSegPredictor 
        if hasattr(yoloe_model, 'task') and 'detect' in str(getattr(yoloe_model, 'task')).lower() and YOLOEVPDetectPredictor is not None:
            selected_predictor_class = YOLOEVPDetectPredictor 

        prompt_bboxes_xyxy = []
        prompt_texts_for_cls = [] 
        
        for box_idx in self.active_ids:
            if 0 <= box_idx < len(self.boxes):
                x1, y1, x2, y2, descs, _ = self.boxes[box_idx]
                prompt_bboxes_xyxy.append([x1, y1, x2, y2])
                prompt_texts_for_cls.append(descs[0] if descs and descs[0] else DEFAULT_LABEL) 
        
        if not prompt_bboxes_xyxy: 
            messagebox.showinfo("Sugerencias YOLOE", "No hay cajas válidas seleccionadas para usar como prompt.", parent=self)
            return

        unique_prompt_texts = sorted(list(set(prompt_texts_for_cls)))
        text_to_cls_id_map = {text: i for i, text in enumerate(unique_prompt_texts)}
        prompt_cls_ids = [text_to_cls_id_map[text] for text in prompt_texts_for_cls]
        
        visual_prompts_data = dict(bboxes=np.array(prompt_bboxes_xyxy, dtype=np.int32), 
                                   cls=np.array(prompt_cls_ids, dtype=np.int32))

        original_predictor = None
        if hasattr(yoloe_model, 'predictor'):
            original_predictor = yoloe_model.predictor
        
        self.suggested_boxes_data.clear()
        self._update_suggestion_buttons_state()
        self._redraw() 

        try:
            print(f"DEBUG VP: Llamando a yoloe_model.predict con {len(prompt_bboxes_xyxy)} prompts visuales.")
            print(f"  DEBUG VP: visual_prompts_data shapes: bboxes={visual_prompts_data['bboxes'].shape}, cls={visual_prompts_data['cls'].shape}")
            print(f"  DEBUG VP: unique_prompt_texts para VP: {unique_prompt_texts}")

            results = yoloe_model.predict(
                self.base_img, 
                prompts=visual_prompts_data, 
                predictor=selected_predictor_class, 
                verbose=False, 
                conf=0.10 
            )

            raw_suggestions = []
            if results and results[0]:
                res0 = results[0]
                boxes_data_source = None
                conf_data_source = None
                cls_data_source = None

                if hasattr(res0, 'boxes') and res0.boxes is not None and res0.boxes.xyxy.numel() > 0:
                    print("DEBUG VP: Usando res0.boxes para xyxy, conf, cls.")
                    boxes_data_source = res0.boxes.xyxy
                    conf_data_source = res0.boxes.conf
                    cls_data_source = res0.boxes.cls

                if boxes_data_source is not None and conf_data_source is not None and cls_data_source is not None:
                    for i in range(boxes_data_source.shape[0]):
                        xyxy = list(map(int, boxes_data_source[i].tolist()))
                        score = float(conf_data_source[i])
                        
                        prompt_cls_id_returned = int(cls_data_source[i])
                        prompt_text_for_suggestion = DEFAULT_LABEL
                        if 0 <= prompt_cls_id_returned < len(unique_prompt_texts):
                            prompt_text_for_suggestion = unique_prompt_texts[prompt_cls_id_returned]
                        
                        raw_suggestions.append((xyxy, prompt_text_for_suggestion, score))
            
            final_suggestions_to_show = []
            iou_threshold_for_filtering = 0.6 
            
            existing_bboxes_for_iou = [b[:4] for b in self.boxes] 

            print(f"DEBUG VP: {len(raw_suggestions)} sugerencias crudas obtenidas. Filtrando contra {len(existing_bboxes_for_iou)} cajas existentes.")

            for sug_xyxy, sug_prompt, sug_score in raw_suggestions:
                is_redundant = False
                if existing_bboxes_for_iou: 
                    for ex_box_xyxy in existing_bboxes_for_iou:
                        iou = calculate_iou(sug_xyxy, ex_box_xyxy) 
                        if iou > iou_threshold_for_filtering:
                            is_redundant = True
                            break
                
                if not is_redundant:
                    final_suggestions_to_show.append((sug_xyxy, sug_prompt, sug_score))
            
            self.suggested_boxes_data = final_suggestions_to_show
            
            if self.suggested_boxes_data:
                print(f"DEBUG VP: {len(self.suggested_boxes_data)} sugerencias finales para mostrar.")
            else:
                messagebox.showinfo("Sugerencias YOLOE","No se generaron nuevas sugerencias (o todas eran redundantes).",parent=self)
            
            self._update_suggestion_buttons_state()
            self._redraw()

        except Exception as e:
            messagebox.showerror("Error Sugerencias YOLOE",f"Error generando sugerencias con YOLOE VP:\n{type(e).__name__}: {e}",parent=self)
            traceback.print_exc()
            self.suggested_boxes_data.clear()
            self._update_suggestion_buttons_state()
            self._redraw()
        finally:
            if hasattr(yoloe_model, 'predictor') and original_predictor is not None:
                yoloe_model.predictor = original_predictor
            pass 

    def _approve_all_suggestions(self):
        if self.editing_suggestion_idx is not None:
            if messagebox.askyesno("Aprobar Todas", "Hay una sugerencia en edición. ¿Cancelar la edición y aprobar el resto?", parent=self):
                self._cancel_edit_suggestion() # Esto llama a _redraw y _update_suggestion_buttons_state
            else:
                return

        if not self.suggested_boxes_data: messagebox.showinfo("Aprobar Sugerencias","No hay sugerencias.",parent=self); return
        num_approved=0; original_len_for_undo = len(self.boxes)
        approved_boxes_data_for_undo = []

        for (x1,y1,x2,y2),prompt_text,score in self.suggested_boxes_data:
            norm_desc = self.main_app.add_description_if_new(prompt_text)
            self.boxes.append((x1,y1,x2,y2,[norm_desc],score))
            approved_boxes_data_for_undo.append( ([x1,y1,x2,y2], norm_desc, score) ) # For redo
            num_approved+=1

        if num_approved > 0: 
            # For undo, we need to know how many were added and what was the original length of self.boxes
            # For redo, we need the actual data of the boxes that were added.
            self._add_to_undo("approve_suggestions", {
                'num_added': num_approved, 
                'original_len': original_len_for_undo,
                # 'approved_boxes_data': approved_boxes_data_for_undo # This was for redo, let redo handler reconstruct
            })

        self.suggested_boxes_data.clear()
        self._update_suggestion_buttons_state()
        self._update_box_count_display(); self._redraw(); self._update_cls_panel()

    def _clear_suggestions(self):
        if self.editing_suggestion_idx is not None:
            if messagebox.askyesno("Limpiar Sugerencias", "Hay una sugerencia en edición. ¿Cancelar la edición y limpiar todas?", parent=self):
                self._cancel_edit_suggestion() # Esto llama a _redraw y _update_suggestion_buttons_state
            else:
                return
        if not self.suggested_boxes_data: return
        self.suggested_boxes_data.clear()
        self._update_suggestion_buttons_state()
        self._redraw(); messagebox.showinfo("Sugerencias","Sugerencias limpiadas.",parent=self)

    def _save(self):
        if not self.boxes:
            json_path = self.labels_json_dir / (self.image_path.stem + ".json")
            if json_path.exists(): 
                try: json_path.unlink()
                except OSError: pass
            return
        if not self.labels_json_dir or not self.labels_json_dir.is_dir(): return
        output_annotations=[]; mem_adds:List[Tuple[Image.Image,str]]=[]
        for idx,(x1,y1,x2,y2,descs,_) in enumerate(self.boxes):
            if x1>=x2 or y1>=y2 or x1<0 or y1<0 or x2>self.original_image_width or y2>self.original_image_height: continue
            prompt_text=DEFAULT_LABEL
            if descs and isinstance(descs,list) and descs[0]: prompt_text=str(descs[0])
            output_annotations.append({"prompt_text":prompt_text,"bbox_xyxy_pixel":[int(x1),int(y1),int(x2),int(y2)],"objectness_target":1.0})
            if self.memory_manager and prompt_text!=DEFAULT_LABEL:
                try: crop=self.base_img.crop((x1,y1,x2,y2)); mem_adds.append((crop,prompt_text))
                except Exception: pass
        output_json_data={"image_filename":self.image_path.name,"image_width":self.original_image_width,"image_height":self.original_image_height,"annotations":output_annotations}
        json_path=self.labels_json_dir/(self.image_path.stem+".json")
        try:
            self.labels_json_dir.mkdir(parents=True,exist_ok=True)
            with open(json_path,"w",encoding="utf-8") as f: json.dump(output_json_data,f,indent=2,ensure_ascii=False)
            num_saved=len(output_annotations)
            if LOCAL_MODULES_AVAILABLE and self.memory_manager and mem_adds:
                for crop_img,desc_text in mem_adds: self.memory_manager.add_entry(crop_img,desc_text)
            messagebox.showinfo("Guardado",f"Guardadas {num_saved} anotaciones para '{self.image_path.name}'.",parent=self)
        except Exception as e: messagebox.showerror("Error Guardar",f"No se pudo guardar:\n{e}",parent=self); traceback.print_exc()
    def _load_prev_labels(self):
        if not self.labels_json_dir or not self.labels_json_dir.is_dir(): return
        json_path=self.labels_json_dir/(self.image_path.stem+".json")
        if not json_path.exists(): return
        try:
            with open(json_path,"r",encoding="utf-8") as f: data=json.load(f)
            loaded_boxes=[]
            if isinstance(data,dict) and "annotations" in data and "image_filename" in data:
                annotations_list=data.get("annotations",[])
                for i,ann_entry in enumerate(annotations_list):
                    if not isinstance(ann_entry,dict) or "prompt_text" not in ann_entry or "bbox_xyxy_pixel" not in ann_entry: continue
                    prompt_raw=ann_entry["prompt_text"]; bbox_px=ann_entry["bbox_xyxy_pixel"]
                    if not isinstance(bbox_px,list) or len(bbox_px)!=4: continue
                    try: x1,y1,x2,y2=map(int,bbox_px)
                    except ValueError: continue
                    if x1>=x2 or y1>=y2 or x1<0 or y1<0 or x2>self.original_image_width or y2>self.original_image_height: continue
                    norm_desc=self.main_app.add_description_if_new(str(prompt_raw))
                    loaded_boxes.append((x1,y1,x2,y2,[norm_desc],-1))
            elif isinstance(data,list):
                img_w,img_h=self.original_image_width,self.original_image_height
                if not(img_w>0 and img_h>0): return
                for i,old_ann in enumerate(data):
                    if not isinstance(old_ann,dict) or "bbox" not in old_ann or "labels" not in old_ann: continue
                    bbox_norm=old_ann["bbox"]
                    if not(isinstance(bbox_norm,list) and len(bbox_norm)==4): continue
                    nx1,ny1,nx2,ny2=bbox_norm; x1,y1,x2,y2=int(nx1*img_w),int(ny1*img_h),int(nx2*img_w),int(ny2*img_h)
                    if x1>=x2 or y1>=y2 or x1<0 or y1<0 or x2>img_w or y2>img_h: continue
                    old_lbls=old_ann["labels"]; prompt_txt=DEFAULT_LABEL
                    if isinstance(old_lbls,list) and old_lbls: prompt_txt=str(old_lbls[0])
                    norm_desc=self.main_app.add_description_if_new(prompt_txt)
                    loaded_boxes.append((x1,y1,x2,y2,[norm_desc],-1))
                if loaded_boxes and messagebox.askyesno("Convertir Formato","Anotaciones en formato antiguo.\n¿Guardar en nuevo formato YOLOE?",parent=self):
                    self.boxes=loaded_boxes; self._save()
            else: messagebox.showerror("Error Carga",f"Formato JSON desconocido: {json_path.name}",parent=self); return
            self.boxes=loaded_boxes; self._refresh_class_list()
        except json.JSONDecodeError: messagebox.showerror("Error Carga",f"Error decodificando JSON: {json_path.name}",parent=self)
        except Exception as e: messagebox.showerror("Error Carga",f"Error cargando: {json_path.name}:\n{e}",parent=self)
        finally: self._update_box_count_display()
    def _run_initial_detection(self):
        data_np=np.array(self.base_img); initial_desc=self.class_names[0] if self.class_names else DEFAULT_LABEL
        try:
            regions=self.detector.detectar(data_np)
            for(x,y,w_reg,h_reg) in regions: self.boxes.append((x,y,x+w_reg,y+h_reg,[initial_desc],-1))
        except Exception as e: messagebox.showerror("Error Detección CV",f"Fallo detección CV:\n{e}",parent=self)
    def _update_box_count_display(self):
        if hasattr(self,'lbl_box_count') and self.lbl_box_count.winfo_exists(): self.lbl_box_count.config(text=f"Cajas: {len(self.boxes)}")
    def _update_gemini_count_display(self):
        if hasattr(self,'lbl_gemini_req_count') and self.lbl_gemini_req_count.winfo_exists(): self.lbl_gemini_req_count.config(text=f"API: {self.gemini_request_count}")
    def request_focus(self): self.canvas.focus_set()
    def get_state_for_save(self) -> bool: return bool(self.undo_stack)
    def _run_detection_pipeline(self):
        if self.editing_suggestion_idx is not None:
            messagebox.showinfo("Pipeline", "Finaliza la edición de la sugerencia actual antes de ejecutar el pipeline.", parent=self)
            return
        det_type=self.main_app.active_pipeline_detector; model_name="?"
        if det_type=="yolo":
            if not self.main_app.loaded_yolo_model or not ULTRALYTICS_AVAILABLE: messagebox.showwarning("Pipeline","Modelo YOLO (aux) no disponible.",parent=self); return 
            model_name="YOLO (Aux)"
        elif det_type=="groundingdino":
            if not self.main_app.loaded_groundingdino_model or not GROUNDINGDINO_AVAILABLE: messagebox.showwarning("Pipeline","Modelo GroundingDINO no disponible.",parent=self); return
            model_name="GroundingDINO (Aux)"
        else: messagebox.showerror("Pipeline",f"Detector desconocido: {det_type}",parent=self); return
        if not self.memory_manager and OPENCLIP_AVAILABLE: messagebox.showwarning("Pipeline","Gestor Memoria Visual no disponible.",parent=self)
        elif not OPENCLIP_AVAILABLE: messagebox.showerror("Pipeline","Librería 'open_clip_torch' no disponible.",parent=self)
        old_boxes_data=[(i,b) for i,b in enumerate(self.boxes)]; self._add_to_undo("pipeline_replace",old_boxes_data)
        detected_boxes_from_model:List[Tuple[Tuple[int,int,int,int],str,float]]=[]
        current_yolo_pipeline_model = self.main_app.loaded_yolo_model if det_type == "yolo" else None 
        
        if det_type=="yolo" and current_yolo_pipeline_model:
            try:
                results=current_yolo_pipeline_model.predict(source=self.base_img,conf=YOLO_CONF,imgsz=YOLO_IMGSZ,device=DEVICE,stream=False,verbose=False)
                if results and results[0] and hasattr(results[0],'boxes') and results[0].boxes:
                    for box_data in results[0].boxes:
                        coords=box_data.xyxy[0].tolist(); x1,y1,x2,y2=map(int,coords); conf=float(box_data.conf[0]); cid=int(box_data.cls[0])
                        lbl_prop=DEFAULT_LABEL; yolo_compat_cls=self.main_app.get_class_names_yolo_compat() 
                        if 0<=cid<len(yolo_compat_cls): lbl_prop=yolo_compat_cls[cid]
                        elif yolo_compat_cls: lbl_prop=yolo_compat_cls[0]
                        detected_boxes_from_model.append(((x1,y1,x2,y2),lbl_prop,conf))
            except Exception as e: messagebox.showerror("Error YOLO",f"Fallo predicción YOLO (Aux):\n{e}",parent=self); self.undo_stack.pop(); self._redraw(); return
        elif det_type=="groundingdino":
            try:
                gd_model=self.main_app.loaded_groundingdino_model; gd_prompt_classes=self.main_app.get_class_names()
                if not gd_prompt_classes: gd_prompt_classes=[DEFAULT_LABEL]
                text_prompt=" . ".join(gd_prompt_classes)+" ."; img_pil_rgb=self.base_img.convert("RGB")
                if T is None: raise ImportError("Torchvision Transforms (T) no disponible.")
                transform=T.Compose([T.Resize(800,max_size=1333),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
                image_tensor=transform(img_pil_rgb).to(DEVICE).unsqueeze(0)
                with torch.no_grad(): outputs=gd_model(image_tensor,captions=[text_prompt])
                logits=outputs["pred_logits"][0]; boxes_cxcywh=outputs["pred_boxes"][0]
                if logits.numel()>0 and boxes_cxcywh.numel()>0:
                    img_w,img_h=img_pil_rgb.size; scores_all,class_indices_all=logits.sigmoid().max(dim=1)
                    for i in range(boxes_cxcywh.shape[0]):
                        score=scores_all[i].item()
                        if score < self.main_app.groundingdino_box_threshold: continue
                        box_norm=boxes_cxcywh[i].cpu().numpy(); x_c,y_c,w_n,h_n=box_norm
                        x1,y1,x2,y2=int((x_c-w_n/2)*img_w),int((y_c-h_n/2)*img_h),int((x_c+w_n/2)*img_w),int((y_c+h_n/2)*img_h)
                        x1,y1,x2,y2=max(0,x1),max(0,y1),min(img_w,x2),min(img_h,y2)
                        if x1>=x2 or y1>=y2: continue
                        class_idx=class_indices_all[i].item(); lbl_gd=DEFAULT_LABEL
                        if logits.shape[1]==len(gd_prompt_classes):
                            if 0<=class_idx<len(gd_prompt_classes): lbl_gd=gd_prompt_classes[class_idx]
                        elif logits.shape[1]>1:
                            if 0<=class_idx<len(gd_prompt_classes): lbl_gd=gd_prompt_classes[class_idx]
                            elif gd_prompt_classes: lbl_gd=gd_prompt_classes[0] 
                        detected_boxes_from_model.append(((x1,y1,x2,y2),lbl_gd,score))
            except Exception as e: messagebox.showerror("Error GDINO",f"Fallo predicción GDINO (Aux):\n{e}",parent=self); self.undo_stack.pop(); self._redraw(); return
        new_boxes_final=[]; count_final=0; mem_hits=0; det_fallback=0
        if detected_boxes_from_model:
            for i,((x1_d,y1_d,x2_d,y2_d),det_lbl,det_conf) in enumerate(detected_boxes_from_model):
                img_w_main,img_h_main=self.base_img.size; x1_c,y1_c,x2_c,y2_c=max(0,x1_d),max(0,y1_d),min(img_w_main,x2_d),min(img_h_main,y2_d)
                if x1_c>=x2_c or y1_c>=y2_c: continue
                try:
                    crop_img=self.base_img.crop((x1_c,y1_c,x2_c,y2_c)); final_desc=DEFAULT_LABEL; norm_det_lbl=normalize_description_text(det_lbl)
                    if not self.memory_manager:
                        if norm_det_lbl and norm_det_lbl!=DEFAULT_LABEL and det_conf>0.5: final_desc=self.main_app.add_description_if_new(norm_det_lbl); det_fallback+=1
                    else:
                        mem_res=self.memory_manager.search_similar(crop_img,k=1,use_tta=True)
                        if mem_res:
                            sim_s,_,mem_ls=mem_res; best_mem_lbl=mem_ls[0] if mem_ls else None; best_mem_sim=sim_s[0] if sim_s else 0.0
                            if best_mem_lbl and best_mem_sim>=MEMORY_SIMILARITY_THRESHOLD: final_desc=best_mem_lbl; mem_hits+=1
                            elif norm_det_lbl and norm_det_lbl!=DEFAULT_LABEL and det_conf>0.5: final_desc=self.main_app.add_description_if_new(norm_det_lbl); det_fallback+=1
                        elif norm_det_lbl and norm_det_lbl!=DEFAULT_LABEL and det_conf>0.5: final_desc=self.main_app.add_description_if_new(norm_det_lbl); det_fallback+=1
                    new_boxes_final.append((x1_c,y1_c,x2_c,y2_c,[final_desc],float(det_conf))); count_final+=1
                except Exception as e_mem: print(f"ERROR Mem/Decisión caja {i}: {e_mem}"); traceback.print_exc()
        self.boxes=new_boxes_final; self.active_ids.clear(); self._update_box_count_display(); self._refresh_class_list(); self._update_cls_panel(); self._redraw()
        messagebox.showinfo(f"Pipeline {model_name}",f"Completado.\n{count_final} cajas.\n({mem_hits} Mem, {det_fallback} Fallback)",parent=self)
    def update_pipeline_button_state(self):
        state="disabled"
        if self.main_app.active_pipeline_detector=="yolo" and ULTRALYTICS_AVAILABLE and self.main_app.loaded_yolo_model: state="normal"
        elif self.main_app.active_pipeline_detector=="groundingdino" and GROUNDINGDINO_AVAILABLE and self.main_app.loaded_groundingdino_model: state="normal"
        if hasattr(self,'btn_pipeline') and self.btn_pipeline.winfo_exists():
            try: self.btn_pipeline.config(state=state)
            except tk.TclError: pass

class MainApp(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Jarvis UI Helper - YOLOE Visual Prompt Mode") 
        self.geometry("1400x900"); self.minsize(800, 600)
        self.output_dir = LABELS_JSON_DIR; self.classes_file_path: Optional[pathlib.Path] = None
        self.class_names: List[str] = []; self.class_names_yolo_compat: List[str] = []
        self.yolo_model_path:Optional[pathlib.Path]=None; self.loaded_yolo_model:Optional[YOLO]=None
        self.groundingdino_config_path:Optional[pathlib.Path]=None; self.groundingdino_checkpoint_path:Optional[pathlib.Path]=None
        self.loaded_groundingdino_model:Optional[Any]=None; self.active_pipeline_detector:str="groundingdino"
        self.groundingdino_box_threshold:float=GROUNDINGDINO_BOX_THRESHOLD; self.groundingdino_text_threshold:float=GROUNDINGDINO_TEXT_THRESHOLD
        self.memory_manager:Optional[MemoryManager]=None; self.training_manager:Optional[TrainingManager]=None
        self.api_key:Optional[str]=os.getenv("GEMINI_API_KEY",API_KEY); self.config_file_path=pathlib.Path.home()/CONFIG_FILE_NAME
        if not self.api_key or self.api_key=="TU_API_KEY_AQUI": self.api_key=None
        elif self.api_key==API_KEY and API_KEY!="TU_API_KEY_AQUI": print("ALERTA: Usando API Key hardcodeada.")
        self.tabs: Dict[str,ImageTab] = {}
        self._ensure_base_dirs_exist(); self._load_config(); self._load_class_names()
        if OPENCLIP_AVAILABLE and MemoryManager:
             try: self.memory_manager = MemoryManager(device=DEVICE,index_save_path=FAISS_INDEX_PATH,labels_save_path=FAISS_LABELS_PATH,class_names=self.class_names)
             except Exception as e: print(f"ERROR MemoryManager: {e}"); self.memory_manager = None
        self._try_load_yolo_on_startup(); self._try_load_groundingdino_on_startup()
        self._build_menu(); self._build_notebook(); self._bind_global_shortcuts()
        if LOCAL_MODULES_AVAILABLE and ULTRALYTICS_AVAILABLE and TrainingManager:
            try: self.training_manager = TrainingManager(app_instance=self,classes=self.class_names,data_base_path=DATA_DIR,model_dir=MODELS_DIR,device=DEVICE)
            except Exception as e: print(f"ERROR TrainingManager: {e}"); self.training_manager = None
    def _ensure_base_dirs_exist(self): [d.mkdir(parents=True,exist_ok=True) for d in [IMAGE_DIR,LABELS_JSON_DIR,LABELS_YOLO_DIR,MODELS_DIR]]
    def _load_config(self):
        cfg_data={}; cfg_p=self.config_file_path
        if cfg_p.exists():
            try: cfg_data=json.loads(cfg_p.read_text(encoding='utf-8'))
            except Exception: pass
        desc_p_str=cfg_data.get('descriptions_file_path',cfg_data.get('classes_file_path'))
        self.classes_file_path = pathlib.Path(desc_p_str) if desc_p_str and (pathlib.Path(desc_p_str).is_file() or pathlib.Path(desc_p_str).parent.is_dir()) else SCRIPT_DIR/DEFAULT_CLASSES_FILENAME
        yolo_s=cfg_data.get('yolo_model_path'); self.yolo_model_path = pathlib.Path(yolo_s) if yolo_s and pathlib.Path(yolo_s).is_file() else (MODELS_DIR/"yoloe.pt" if (MODELS_DIR/"yoloe.pt").is_file() else None)
        gd_cfg_s=cfg_data.get('groundingdino_config_path'); self.groundingdino_config_path = pathlib.Path(gd_cfg_s) if gd_cfg_s and pathlib.Path(gd_cfg_s).is_file() else (MODELS_DIR/DEFAULT_GROUNDINGDINO_CONFIG_FILENAME if (MODELS_DIR/DEFAULT_GROUNDINGDINO_CONFIG_FILENAME).is_file() else None)
        gd_ckpt_s=cfg_data.get('groundingdino_checkpoint_path'); self.groundingdino_checkpoint_path = pathlib.Path(gd_ckpt_s) if gd_ckpt_s and pathlib.Path(gd_ckpt_s).is_file() else (MODELS_DIR/DEFAULT_GROUNDINGDINO_CHECKPOINT_FILENAME if (MODELS_DIR/DEFAULT_GROUNDINGDINO_CHECKPOINT_FILENAME).is_file() else None)
        self.active_pipeline_detector=cfg_data.get('active_pipeline_detector','groundingdino')
    def _save_config(self):
        cfg={'active_pipeline_detector':self.active_pipeline_detector}
        if self.classes_file_path: cfg['descriptions_file_path']=str(self.classes_file_path)
        if self.yolo_model_path: cfg['yolo_model_path']=str(self.yolo_model_path)
        if self.groundingdino_config_path: cfg['groundingdino_config_path']=str(self.groundingdino_config_path)
        if self.groundingdino_checkpoint_path: cfg['groundingdino_checkpoint_path']=str(self.groundingdino_checkpoint_path)

        try: self.config_file_path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        except OSError: pass

    def _ensure_savpe_methods(self, yolo_obj):
        try:
            if not hasattr(yolo_obj, 'model') or yolo_obj.model is None:
                return
            for m in yolo_obj.model.modules():
                if not hasattr(m, 'savpe'):
                    if hasattr(m, 'save') and callable(getattr(m, 'save', None)):
                        m.savpe = m.save
                    else:
                        m.savpe = lambda x, vpe=None: vpe
        except Exception as e:
            print(f"WARN: _ensure_savpe_methods: {e}")
    def _try_load_yolo_on_startup(self):
        if self.yolo_model_path and self.yolo_model_path.is_file() and ULTRALYTICS_AVAILABLE:
            try:
                self.loaded_yolo_model = YOLO(self.yolo_model_path)
                if hasattr(self.loaded_yolo_model, 'model') and \
                   hasattr(self.loaded_yolo_model.model, 'save') and \
                   not hasattr(self.loaded_yolo_model.model, 'savpe'):
                    self.loaded_yolo_model.model.savpe = self.loaded_yolo_model.model.save
                self._ensure_savpe_methods(self.loaded_yolo_model)
                print(f"INFO: Modelo YOLOE principal '{self.yolo_model_path.name}' cargado.")
            except Exception as e: print(f"ERROR: Carga YOLOE: {e}"); self.loaded_yolo_model=None

        elif self.yolo_model_path: print(f"WARN: Ruta YOLOE '{self.yolo_model_path}' no encontrada.")
        self.class_names_yolo_compat=[]
    def get_class_names_yolo_compat(self) -> List[str]: return self.class_names_yolo_compat if self.class_names_yolo_compat else [DEFAULT_LABEL]
    def _try_load_groundingdino_on_startup(self):
        if not GROUNDINGDINO_AVAILABLE: return
        if self.groundingdino_config_path and self.groundingdino_checkpoint_path and self.groundingdino_config_path.is_file() and self.groundingdino_checkpoint_path.is_file():
            try:
                model=gd_load_model(str(self.groundingdino_config_path),str(self.groundingdino_checkpoint_path),device=DEVICE)
                if model is not None: self.loaded_groundingdino_model=model.to(DEVICE); print("INFO: GroundingDINO (aux) cargado.")
            except Exception as e: print(f"ERROR carga GD (aux): {e}")
    def _build_menu(self):
        mb=tk.Menu(self); self.config(menu=mb); fm=tk.Menu(mb,tearoff=False); mb.add_cascade(label="Archivo",menu=fm)
        fm.add_command(label="Abrir Imagen(es)...",command=self._open_images,accelerator="Ctrl+O")
        fm.add_command(label="Establecer Archivo de Descripciones...",command=self._set_classes_file,accelerator="Ctrl+L")
        fm.add_command(label="Editar Descripciones...", command=self._edit_descriptions)
        fm.add_separator()
        fm.add_command(label="Cargar Modelo YOLOE (.pt)...",command=self._load_yolo_model_path,accelerator="Ctrl+M")
        fm.add_command(label="Cargar Modelo GroundingDINO (Aux.)...",command=self._load_groundingdino_model_files)
        pm=tk.Menu(fm,tearoff=False); fm.add_cascade(label="Detector Pipeline Auxiliar",menu=pm)
        self.pipeline_detector_var=tk.StringVar(value=self.active_pipeline_detector)
        pm.add_radiobutton(label="Usar GroundingDINO (Aux)",variable=self.pipeline_detector_var,value="groundingdino",command=self._set_active_pipeline_detector)
        pm.add_radiobutton(label="Usar YOLO (Aux)",variable=self.pipeline_detector_var,value="yolo",command=self._set_active_pipeline_detector)
        fm.add_separator(); fm.add_command(label="Guardar Pestaña Actual",command=self._save_current_tab,accelerator="Ctrl+S")
        fm.add_command(label="Cerrar Pestaña Actual",command=self._close_current_tab,accelerator="Ctrl+W"); fm.add_separator()
        fm.add_command(label="Salir",command=self._quit)
    def _edit_descriptions(self): EditDescriptionsWindow(self, self)
    def _build_notebook(self): self.notebook=tb.Notebook(self,bootstyle="dark"); self.notebook.pack(expand=True,fill="both",padx=5,pady=5); self.notebook.bind("<<NotebookTabChanged>>",self._on_tab_changed)
    def _bind_global_shortcuts(self):
        self.bind_all("<Control-o>",lambda e:self._open_images() or "break"); self.bind_all("<Control-l>",lambda e:self._set_classes_file() or "break")
        self.bind_all("<Control-m>",lambda e:self._load_yolo_model_path() or "break"); self.bind_all("<Control-s>",lambda e:self._save_current_tab() or "break")
        self.bind_all("<Control-w>",lambda e:self._close_current_tab() or "break")
        self.bind_all("<Control-z>", lambda e: self._delegate_to_current_tab('_undo_if_active_tab', e) or "break")
        self.bind_all("<Control-Shift-KeyPress-Z>", lambda e: self._delegate_to_current_tab('_redo_if_active_tab', e) or "break")
        self.bind_all("<Control-Y>", lambda e: self._delegate_to_current_tab('_redo_if_active_tab', e) or "break") 
        self.bind_all("<Control-y>", lambda e: self._delegate_to_current_tab('_redo_if_active_tab', e) or "break") 
    def _delegate_to_current_tab(self, method_name: str, event=None):
        current_tab = self.get_current_tab()
        if current_tab and hasattr(current_tab, method_name):
            method_to_call = getattr(current_tab, method_name)
            if callable(method_to_call):
                try:
                    if event: method_to_call(event)
                    else: method_to_call()
                except Exception as e: print(f"Error delegando '{method_name}': {e}"); traceback.print_exc()
    def _set_classes_file(self):
        idir=self.classes_file_path.parent if self.classes_file_path and self.classes_file_path.parent.is_dir() else SCRIPT_DIR
        ifile=self.classes_file_path.name if self.classes_file_path else DEFAULT_CLASSES_FILENAME
        fp=filedialog.asksaveasfilename(title="Archivo Descripciones",initialdir=str(idir),initialfile=ifile,defaultextension=".txt",filetypes=[("Texto","*.txt")])
        if fp and (ncfp:=pathlib.Path(fp))!=self.classes_file_path or not self.class_names:
            self.classes_file_path=ncfp; self._load_class_names(); self._save_config()
            if self.training_manager: self.training_manager.update_classes(self.class_names)
            if self.memory_manager: self.memory_manager.update_class_names(self.class_names)
            self.refresh_class_lists_in_tabs(); messagebox.showinfo("Archivo Descripciones",f"Archivo: {ncfp.name}\n{len(self.class_names)} descripciones.")
    def _load_yolo_model_path(self): 
        if not ULTRALYTICS_AVAILABLE: messagebox.showerror("Error","'ultralytics' no instalado."); return
        idir = self.yolo_model_path.parent if self.yolo_model_path and self.yolo_model_path.parent.is_dir() else MODELS_DIR
        fp_str = filedialog.askopenfilename(title="Seleccionar Modelo YOLOE (.pt)", initialdir=str(idir), filetypes=[("Modelo PyTorch", "*.pt")])
        if fp_str and (nmp := pathlib.Path(fp_str)).is_file():

            try:
                newly_loaded_model = YOLO(nmp)
                if hasattr(newly_loaded_model, 'model') and \
                   hasattr(newly_loaded_model.model, 'save') and \
                   not hasattr(newly_loaded_model.model, 'savpe'):
                    newly_loaded_model.model.savpe = newly_loaded_model.model.save
                self._ensure_savpe_methods(newly_loaded_model)
                self.yolo_model_path = nmp; self.loaded_yolo_model = newly_loaded_model
                print(f"INFO: Modelo YOLOE '{nmp.name}' cargado."); self.class_names_yolo_compat = []

                self._save_config(); self.update_yolo_model_in_tabs(self.loaded_yolo_model)
                messagebox.showinfo("Modelo Cargado", f"Modelo YOLOE cargado:\n{nmp.name}")
            except Exception as e: messagebox.showerror("Error Carga Modelo", f"No se pudo cargar YOLOE:\n{nmp}\n{e}"); traceback.print_exc()
    def reload_yolo_model_from_path(self, model_path: pathlib.Path):
        if not model_path.is_file() or not ULTRALYTICS_AVAILABLE: return

        try:
            nlm = YOLO(model_path)
            if hasattr(nlm, 'model') and hasattr(nlm.model, 'save') and not hasattr(nlm.model, 'savpe'):
                nlm.model.savpe = nlm.model.save
            self._ensure_savpe_methods(nlm)
            self.yolo_model_path = model_path; self.loaded_yolo_model = nlm
            print(f"INFO: Modelo YOLOE recargado: {model_path.name}"); self.class_names_yolo_compat = []

            self._save_config(); self.update_yolo_model_in_tabs(nlm)
            messagebox.showinfo("Modelo Actualizado",f"Modelo YOLOE actualizado:\n{model_path.name}")
        except Exception as e: messagebox.showerror("Error Recarga YOLOE",f"No se pudo recargar:\n{model_path}\n{e}")
    def _open_images(self):
        idir=IMAGE_DIR if IMAGE_DIR.is_dir() else SCRIPT_DIR
        fps_tuple=filedialog.askopenfilenames(title="Seleccionar Imágenes",initialdir=str(idir),filetypes=[("Imágenes","*.png *.jpg *.jpeg *.bmp *.webp"),("Todos","*.*")])
        if fps_tuple:
            opened_count=0
            for fp_str in fps_tuple:
                orig_path=pathlib.Path(fp_str); target_path=IMAGE_DIR/orig_path.name
                if not target_path.exists() or not orig_path.resolve().samefile(target_path.resolve()):
                    try: IMAGE_DIR.mkdir(parents=True,exist_ok=True); shutil.copy(orig_path,target_path)
                    except Exception as e: messagebox.showerror("Error Copia",f"No se pudo copiar:\n{orig_path}\na\n{target_path}\n{e}"); continue
                tab_img_path=target_path; already_open=False
                for tid,tab_inst in self.tabs.items():
                    if tab_inst.image_path.resolve()==tab_img_path.resolve():
                        try: self.notebook.select(tid); already_open=True
                        except tk.TclError: pass; break
                if not already_open:
                    try:
                        new_tab=ImageTab(self.notebook,self,tab_img_path,LABELS_JSON_DIR,self.api_key,self.loaded_yolo_model,self.memory_manager)
                        self.notebook.add(new_tab,text=tab_img_path.name)
                        ctid=self.notebook.select() 
                        if ctid: self.tabs[str(ctid)]=new_tab; self.notebook.select(ctid); opened_count+=1 
                    except Exception as e: messagebox.showerror("Error Abrir",f"No se pudo crear pestaña para:\n{tab_img_path}\n{e}"); traceback.print_exc()
            if opened_count > 0: self.update_pipeline_button_states_in_tabs()
    def get_current_tab(self)->Optional[ImageTab]:
        try: sel_tab_id_tk = self.notebook.select(); return self.tabs.get(str(sel_tab_id_tk)) if sel_tab_id_tk else None
        except (tk.TclError, AttributeError): return None
    def _save_current_tab(self): ct=self.get_current_tab(); ct._save() if ct else None
    def _close_current_tab(self):
        if not self.tabs: return
        try: ctid_tk=self.notebook.select()
        except tk.TclError: return
        if ctid_tk:
            ct_inst=self.tabs.get(str(ctid_tk)); should_save=False
            if ct_inst and ct_inst.winfo_exists():
                if ct_inst.editing_suggestion_idx is not None:
                    if messagebox.askyesno("Cerrar Pestaña", "Hay una sugerencia en edición. ¿Descartar cambios y cerrar?", parent=self):
                        ct_inst._cancel_edit_suggestion() # Discard edits to suggestion
                    else:
                        return # Don't close tab

                if ct_inst.get_state_for_save():
                    save_q=messagebox.askyesnocancel("Cerrar",f"Guardar cambios en '{ct_inst.image_path.name}'?")
                    if save_q is None: return
                    should_save=save_q
                if should_save: ct_inst._save()
                if str(ctid_tk) in self.tabs: del self.tabs[str(ctid_tk)]
                try: self.notebook.forget(ctid_tk)
                except tk.TclError: pass
            else:
                if str(ctid_tk) in self.tabs: del self.tabs[str(ctid_tk)]
                try: self.notebook.forget(ctid_tk)
                except tk.TclError: pass
    def _on_tab_changed(self, event):
        ct=self.get_current_tab()
        if ct: 
            ct.request_focus(); ct._update_group_button_state(); ct.update_pipeline_button_state()
            ct.class_names = list(self.class_names); ct._refresh_class_list()
            ct._update_cursor_and_crosshair()
            ct._update_suggestion_buttons_state() # Update suggestion edit buttons for new tab
    def _quit(self):
        for tab_id, tab_instance in list(self.tabs.items()): # Iterate over a copy for safe deletion
            if tab_instance.editing_suggestion_idx is not None:
                confirm_quit = messagebox.askyesnocancel("Salir", 
                    f"La pestaña '{tab_instance.image_path.name}' tiene una sugerencia en edición.\n"
                    "¿Descartar cambios en la sugerencia y continuar saliendo?",
                    parent=self)
                if confirm_quit is None: # Cancel quit
                    return
                elif confirm_quit is True: # Yes, discard and continue
                    tab_instance._cancel_edit_suggestion() 
                # If False, it means user wants to handle it manually, but we can't wait here.
                # Forcing a cancel of edit if they choose to proceed with quit after this.
                # A more complex flow would save/re-select tab, but this is simpler for quit.

        if self.memory_manager: self.memory_manager._save_memory()
        if self.training_manager: self.training_manager.stop_training_thread()
        self._save_config(); self.destroy()
    def get_class_names(self)->List[str]: return list(self.class_names)
    def save_class_names(self):
        if not self.classes_file_path: return
        try:
            self.classes_file_path.parent.mkdir(parents=True,exist_ok=True)
            save_list = sorted([desc for desc in list(set(self.class_names)) if desc and desc.strip()])
            self.classes_file_path.write_text("\n".join(save_list)+"\n",encoding="utf-8")
        except Exception as e: print(f"ERROR guardando descripciones: {e}")
    def _load_class_names(self):
        loaded_descriptions=set(); op_desc=""
        if self.classes_file_path and self.classes_file_path.is_file():
            try:
                lines=self.classes_file_path.read_text(encoding="utf-8").splitlines()
                loaded_descriptions={normalize_description_text(l) for l in lines if normalize_description_text(l)}
                if loaded_descriptions: self.class_names=sorted(list(loaded_descriptions)); op_desc=f"Descripciones ({len(self.class_names)}) cargadas."
                else: self.class_names=[]; op_desc=f"Archivo {self.classes_file_path.name} vacío."
            except Exception: self.class_names=[]; op_desc=f"Error leyendo {self.classes_file_path.name}."
        else: self.class_names=[]; op_desc="Archivo descripciones no encontrado."
        if not self.class_names: op_desc+=" Lista vacía."
        print(f"INFO: {op_desc.strip()}")
        if self.classes_file_path and not self.classes_file_path.exists():
            try: self.classes_file_path.parent.mkdir(parents=True,exist_ok=True); self.classes_file_path.touch()
            except Exception: pass
    def add_description_if_new(self, description_raw: str) -> str:
        normalized_desc = normalize_description_text(description_raw)
        if not normalized_desc: return DEFAULT_LABEL
        if normalized_desc not in self.class_names:
            self.class_names.append(normalized_desc); self.class_names.sort(); self.save_class_names(); self.refresh_class_lists_in_tabs()
        return normalized_desc
    def refresh_class_lists_in_tabs(self):
        for tab_instance in self.tabs.values():
            if tab_instance and tab_instance.winfo_exists():
                try: 
                    tab_instance.class_names=list(self.class_names)
                    tab_instance.class2id={n:i for i,n in enumerate(tab_instance.class_names)}
                    tab_instance._refresh_class_list()
                except tk.TclError: pass
    def _load_groundingdino_model_files(self):
        if not GROUNDINGDINO_AVAILABLE: messagebox.showerror("Error","'groundingdino-py' no disponible."); return
        icfg_dir=self.groundingdino_config_path.parent if self.groundingdino_config_path and self.groundingdino_config_path.parent.is_dir() else MODELS_DIR
        cfg_pstr=filedialog.askopenfilename(title="Config GD (.py)",initialdir=str(icfg_dir),filetypes=[("Py","*.py")])
        if not cfg_pstr: return
        ngd_cfg_p=pathlib.Path(cfg_pstr)
        ickpt_dir=self.groundingdino_checkpoint_path.parent if self.groundingdino_checkpoint_path and self.groundingdino_checkpoint_path.parent.is_dir() else MODELS_DIR
        ckpt_pstr=filedialog.askopenfilename(title="Checkpoint GD (.pth)",initialdir=str(ickpt_dir),filetypes=[("PTH","*.pth")])
        if not ckpt_pstr: return
        ngd_ckpt_p=pathlib.Path(ckpt_pstr)
        if ngd_cfg_p.is_file() and ngd_ckpt_p.is_file():
            if ngd_cfg_p!=self.groundingdino_config_path or ngd_ckpt_p!=self.groundingdino_checkpoint_path or self.loaded_groundingdino_model is None:
                try:
                    ngdm=gd_load_model(str(ngd_cfg_p),str(ngd_ckpt_p),device=DEVICE)
                    if ngdm is not None:
                        ngdm.to(DEVICE); self.groundingdino_config_path=ngd_cfg_p; self.groundingdino_checkpoint_path=ngd_ckpt_p
                        self.loaded_groundingdino_model=ngdm; self._save_config(); self.update_pipeline_button_states_in_tabs()
                        messagebox.showinfo("Modelo GD Cargado",f"Cargado (Aux):\nConfig: {ngd_cfg_p.name}\nCkpt: {ngd_ckpt_p.name}")
                except Exception as e: messagebox.showerror("Error Carga GD",f"No se pudo cargar GD (Aux):\n{e}"); self.update_pipeline_button_states_in_tabs()
    def _set_active_pipeline_detector(self):
        nd=self.pipeline_detector_var.get()
        if nd!=self.active_pipeline_detector: self.active_pipeline_detector=nd; self._save_config(); self.update_pipeline_button_states_in_tabs()
    def update_yolo_model_in_tabs(self, new_model_object: Optional[YOLO]):
        self.update_pipeline_button_states_in_tabs() # Pipeline might depend on main YOLO model
        for tab in self.tabs.values():
            if tab and tab.winfo_exists():
                tab.yolo_model = new_model_object # This is for YOLOE VP
                if hasattr(tab, 'btn_yoloe_visual_prompt'):
                    state = "normal" if ULTRALYTICS_AVAILABLE and self.loaded_yolo_model and YOLOVP_PREDICTORS_AVAILABLE else "disabled"
                    try: tab.btn_yoloe_visual_prompt.config(state=state)
                    except tk.TclError: pass
    def update_pipeline_button_states_in_tabs(self):
        for tab_instance in self.tabs.values():
            if tab_instance and tab_instance.winfo_exists():
                try: tab_instance.update_pipeline_button_state()
                except tk.TclError: pass

def main():
    print(f"Iniciando Jarvis UI Helper - YOLOE Visual Prompt Mode...")
    app = MainApp()
    app.mainloop()
    print("Jarvis UI Helper cerrado.")

if __name__=="__main__":
    main()
