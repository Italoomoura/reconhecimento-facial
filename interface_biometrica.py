"""
=============================================================================
  SISTEMA DE IDENTIFICAÇÃO E AUTENTICAÇÃO BIOMÉTRICA — INTERFACE GRÁFICA
  Ministério do Meio Ambiente | Controle de Agrotóxicos
  APS — Ciência da Computação 5°/6° — UNIP 2025/2
=============================================================================
  Dependências:  pip install opencv-contrib-python Pillow
  Uso:           python interface_biometrica.py
=============================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import sqlite3
import os
import sys
from datetime import datetime

try:
    from PIL import Image, ImageTk
except ImportError:
    print("\n[ERRO] Pillow não encontrado. Execute:\n  pip install Pillow\n"); sys.exit(1)

if not hasattr(cv2, "face"):
    print("\n[ERRO] opencv-contrib-python não encontrado. Execute:\n  pip install opencv-contrib-python\n"); sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  ESCALA ADAPTATIVA — calculada antes de qualquer widget
#  Referência de projeto: 1366×768 (notebook comum)
#  Em telas maiores/menores tudo escala proporcionalmente via sc() e fs()
# ═══════════════════════════════════════════════════════════════════════════════
_root_tmp = tk.Tk()
_root_tmp.withdraw()
_SW = _root_tmp.winfo_screenwidth()
_SH = _root_tmp.winfo_screenheight()
_root_tmp.destroy()

SCALE = min(_SW / 1366, _SH / 768)
SCALE = max(0.72, min(SCALE, 1.6))   # limita entre 72 % e 160 %

def sc(n):  return max(1, int(n * SCALE))   # escala pixels
def fs(n):  return max(7, int(n * SCALE))   # escala fontes

WIN_W = min(int(_SW * 0.88), sc(1300))
WIN_H = min(int(_SH * 0.88), sc(800))

# ── Paleta ──────────────────────────────────────────────────────────────────
BG      = "#09111c"
SURFACE = "#111d2b"
PANEL   = "#0d1825"
ACCENT  = "#1d7dcc"
ACCENT2 = "#2596e1"
SUCCESS = "#1e8f56"
DANGER  = "#b83232"
GOLD    = "#c8981f"
TEXT    = "#d6e8f7"
DIM     = "#4a6a85"
BORDER  = "#1a3352"
HEADER  = "#060e18"
GREEN2  = "#17a85a"

FONT    = "Courier New"
FONT_UI = "Segoe UI"

NIVEIS      = {1:"Nível 1 — Público", 2:"Nível 2 — Diretor de Divisão", 3:"Nível 3 — Ministro"}
NIVEL_CORES = {1:ACCENT, 2:GOLD, 3:DANGER}

# ── Config biométrica ────────────────────────────────────────────────────────
DB_PATH      = "biometrico.db"
MODELS_DIR   = "modelos_faciais"
HAAR         = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
NUM_AMOSTRAS = 40
CONF_MAX     = 75
FACE_SZ      = (200, 200)

# ═══════════════════════════════════════════════════════════════════════════════
#  BANCO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════
def init_db():
    os.makedirs(MODELS_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS usuarios(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT, cargo TEXT, nivel_acesso INTEGER,
                modelo_path TEXT, cadastrado_em TEXT);
            CREATE TABLE IF NOT EXISTS logs_acesso(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER, resultado TEXT,
                confianca REAL, data_hora TEXT);
        """)

def _now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def db_add_user(nome, cargo, nivel):
    with sqlite3.connect(DB_PATH) as c:
        return c.execute(
            "INSERT INTO usuarios(nome,cargo,nivel_acesso,cadastrado_em) VALUES(?,?,?,?)",
            (nome, cargo, nivel, _now())).lastrowid

def db_set_model(uid, path):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("UPDATE usuarios SET modelo_path=? WHERE id=?", (path, uid))

def db_del_user(uid):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("DELETE FROM usuarios WHERE id=?", (uid,))

def db_users(model_only=False):
    q = "SELECT id,nome,cargo,nivel_acesso,modelo_path FROM usuarios"
    if model_only: q += " WHERE modelo_path IS NOT NULL"
    with sqlite3.connect(DB_PATH) as c:
        return c.execute(q + " ORDER BY id").fetchall()

def db_log(uid, result, conf):
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            "INSERT INTO logs_acesso(usuario_id,resultado,confianca,data_hora) VALUES(?,?,?,?)",
            (uid, result, conf, _now()))

def db_logs(n=40):
    with sqlite3.connect(DB_PATH) as c:
        return c.execute("""
            SELECT l.data_hora, u.nome, l.resultado, l.confianca
            FROM logs_acesso l LEFT JOIN usuarios u ON l.usuario_id=u.id
            ORDER BY l.id DESC LIMIT ?""", (n,)).fetchall()

# ═══════════════════════════════════════════════════════════════════════════════
#  FACE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
_detector = cv2.CascadeClassifier(HAAR)

def detect_faces(gray):
    return _detector.detectMultiScale(gray, 1.2, 6, minSize=(80,80))

def detect_faces_file(gray):
    MAX_DIM = 1200
    h, w = gray.shape
    if max(h, w) > MAX_DIM:
        s = MAX_DIM / max(h, w)
        gray = cv2.resize(gray, (int(w*s), int(h*s)))
    gray = cv2.equalizeHist(gray)
    for p in [
        dict(scaleFactor=1.1,  minNeighbors=5, minSize=(60,60)),
        dict(scaleFactor=1.05, minNeighbors=4, minSize=(40,40)),
        dict(scaleFactor=1.05, minNeighbors=3, minSize=(25,25)),
        dict(scaleFactor=1.05, minNeighbors=2, minSize=(20,20)),
    ]:
        f = _detector.detectMultiScale(gray, **p)
        if len(f) > 0: return f, gray
    return [], gray

def model_path(uid): return os.path.join(MODELS_DIR, f"usuario_{uid}.yml")

def train_save(samples, uid):
    r = cv2.face.LBPHFaceRecognizer_create()
    r.train(samples, np.array([uid]*len(samples), dtype=np.int32))
    p = model_path(uid); r.save(p); return p

def recognize(roi, uid, mpath):
    if not os.path.exists(mpath): return False, 999.0
    r = cv2.face.LBPHFaceRecognizer_create(); r.read(mpath)
    label, conf = r.predict(cv2.resize(roi, FACE_SZ))
    return (label == uid and conf < CONF_MAX), float(conf)

# ═══════════════════════════════════════════════════════════════════════════════
#  DADOS DO MINISTÉRIO
# ═══════════════════════════════════════════════════════════════════════════════
MMA_DATA = {
    1: {"titulo":"NÍVEL 1 — INFORMAÇÕES PÚBLICAS","cor":ACCENT,"icone":"◈","conteudo":[
        ("Relatório Público — Uso de Agrotóxicos no Brasil (2025)","titulo"),
        ("",""),
        ("Propriedades rurais cadastradas (SICAR)","item","5.162.000"),
        ("Agrotóxicos monitorados (substâncias)","item","147"),
        ("Alertas públicos vigentes","item","3"),
        ("Portal de transparência","item","mma.gov.br/relatorios"),
        ("",""),
        ("Fonte: Sistema Nacional de Informações sobre Agrotóxicos (SIAGRO)","rodape"),
    ]},
    2: {"titulo":"NÍVEL 2 — CONFIDENCIAL  |  Diretores de Divisão","cor":GOLD,"icone":"◆","conteudo":[
        ("Propriedades com uso de Agrotóxicos Proibidos","titulo"),
        ("",""),
        ("Estado","cabecalho","Propriedades","Principal substância ilegal"),
        ("MT","linha","1.243","Clorpirifós  (banido 2020)"),
        ("PA","linha","987",  "Paraquate    (banido 2020)"),
        ("MS","linha","654",  "Endossulfam  (banido 2013)"),
        ("GO","linha","432",  "Carbofurano  (banido 2017)"),
        ("MG","linha","318",  "Abamectina   (uso restrito)"),
        ("",""),
        ("Rios contaminados mapeados","item","73"),
        ("Regiões hidrográficas afetadas","item","18"),
        ("",""),
        ("⚠  RESTRITO — Uso exclusivo de Diretores de Divisão","aviso"),
    ]},
    3: {"titulo":"NÍVEL 3 — ULTRA SIGILOSO  |  Ministro","cor":DANGER,"icone":"⬟","conteudo":[
        ("Operações Classificadas em Andamento","titulo"),
        ("",""),
        ("OP-2026-MMA-07","op","Fiscalização Cerrado (3 estados)"),
        ("OP-2026-MMA-12","op","Monitoramento Aquífero Guarani"),
        ("OP-2026-MMA-19","op","Rastreamento Fornecedores Ilegais"),
        ("",""),
        ("Propriedades Prioritárias — Ação Judicial Imediata","titulo"),
        ("",""),
        ("#ID-00412","prop","Fazenda Aurora (MT)","Glyphosate + Paraquate"),
        ("#ID-00789","prop","Agropec Serra (PA)","Endossulfam"),
        ("#ID-01103","prop","Grupo Terras S/A (GO)","Múltiplas substâncias"),
        ("",""),
        ("Parceria Internacional: UNEP — código UNEP-7 (ativo)","item",""),
        ("Próxima reunião de crise: 22/05/2026 — Brasília, Sala 403","item",""),
        ("",""),
        ("🔒  ULTRA SIGILOSO — ACESSO EXCLUSIVO AO MINISTRO","aviso"),
    ]},
}

# ═══════════════════════════════════════════════════════════════════════════════
#  WIDGETS BASE  (todos usam fs() e sc())
# ═══════════════════════════════════════════════════════════════════════════════
def make_btn(parent, text, cmd, bg=ACCENT, fg=TEXT, size=11, bold=True, pad=10, **kw):
    return tk.Button(parent, text=text, command=cmd,
                     bg=bg, fg=fg, activebackground=ACCENT2, activeforeground=TEXT,
                     font=(FONT_UI, fs(size), "bold" if bold else "normal"),
                     relief="flat", cursor="hand2", pady=sc(pad), bd=0, **kw)

def make_entry(parent, var):
    return tk.Entry(parent, textvariable=var, font=(FONT_UI, fs(10)),
                    bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                    relief="flat", bd=sc(6),
                    highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT)

def sep(parent, color=BORDER, pady=0):
    f = tk.Frame(parent, bg=color, height=1)
    f.pack(fill="x", pady=pady); return f

def card(parent, **kw):
    return tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, **kw)

def sec_label(parent, text):
    sep(parent, pady=sc(7))
    tk.Label(parent, text=text, font=(FONT, fs(8), "bold"), fg=DIM, bg=SURFACE
             ).pack(anchor="w", padx=sc(12))

def make_topbar(parent, title, title_color, back_cmd, back_label="←  INÍCIO"):
    bar = tk.Frame(parent, bg=PANEL, pady=sc(10))
    bar.pack(fill="x")
    make_btn(bar, back_label, back_cmd, bg=PANEL, fg=DIM, size=9, pad=5, bold=False
             ).pack(side="left", padx=sc(12))
    tk.Label(bar, text=title, font=(FONT, fs(12), "bold"), fg=title_color, bg=PANEL
             ).pack(side="left", padx=sc(8))
    return bar

# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA WIDGET — canvas fill="both"/expand=True → redimensiona com container
# ═══════════════════════════════════════════════════════════════════════════════
class CameraWidget:
    def __init__(self, parent):
        self.parent    = parent
        self.cap       = None
        self.running   = False
        self.last_face = None
        self.samples   = []
        self.on_progress = None
        self._photo    = None
        self._cw = self._ch = 1

        self.canvas = tk.Canvas(parent, bg="#040a11", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self._draw_placeholder()

    def _on_resize(self, event):
        self._cw = max(event.width, 1)
        self._ch = max(event.height, 1)
        if not self.running:
            self._draw_placeholder()

    def _draw_placeholder(self):
        w, h = self._cw, self._ch
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, w, h, fill="#040a11", outline="")
        step = max(28, sc(38))
        for x in range(0, w, step):
            for y in range(0, h, step):
                self.canvas.create_oval(x-1, y-1, x+1, y+1, fill="#0d2035", outline="")
        cx, cy = w//2, h//2
        self.canvas.create_text(cx, cy-sc(18), text="◉",  font=(FONT, fs(30)), fill=DIM)
        self.canvas.create_text(cx, cy+sc(14), text="CÂMERA INATIVA", font=(FONT, fs(9)), fill=DIM)

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): return False
        self.running = True; self._loop(); return True

    def stop(self):
        self.running = False
        if self.cap: self.cap.release(); self.cap = None
        self._draw_placeholder()

    def _loop(self):
        if not self.running or not self.cap: return
        ok, frame = self.cap.read()
        if ok:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)
            fh, fw = frame.shape[:2]
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (29,125,204), 2)
                l = 20
                for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                    cv2.line(frame,(px,py),(px+dx*l,py),(29,200,255),2)
                    cv2.line(frame,(px,py),(px,py+dy*l),(29,200,255),2)
                self.last_face = gray[y:y+h, x:x+w]
                if self.on_progress is not None and len(self.samples) < NUM_AMOSTRAS:
                    self.samples.append(cv2.resize(self.last_face, FACE_SZ))
                    self.on_progress(len(self.samples))
            cv2.line(frame, (0, fh//2), (fw, fh//2), (29,125,204), 1)
            cw, ch = max(self._cw,1), max(self._ch,1)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((cw, ch), Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.after(30, self._loop)

    def capture(self): return self.last_face
    def reset(self): self.samples = []; self.last_face = None

# ═══════════════════════════════════════════════════════════════════════════════
#  BASE PAGE
# ═══════════════════════════════════════════════════════════════════════════════
class Page(tk.Frame):
    def __init__(self, app):
        super().__init__(app.container, bg=BG)
        self.app = app
    def show(self): self.tkraise()
    def hide(self): pass

# ═══════════════════════════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════════════════════════
class HomePage(Page):
    def __init__(self, app):
        super().__init__(app); self._build()

    def _build(self):
        # Sidebar largura proporcional
        sw = max(sc(190), min(sc(255), int(_SW * 0.17)))
        side = tk.Frame(self, bg=HEADER, width=sw)
        side.pack(side="left", fill="y"); side.pack_propagate(False)

        tk.Frame(side, bg=HEADER, height=sc(36)).pack()
        tk.Label(side, text="◈", font=(FONT, fs(42)), fg=ACCENT, bg=HEADER).pack(pady=(sc(14),0))
        tk.Label(side, text="MMA", font=(FONT, fs(19), "bold"), fg=TEXT, bg=HEADER).pack()
        tk.Label(side, text="Ministério do\nMeio Ambiente",
                 font=(FONT_UI, fs(9)), fg=DIM, bg=HEADER, justify="center").pack()
        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=sc(18), pady=sc(22))
        tk.Label(side, text="SISTEMA DE CONTROLE\nDE AGROTÓXICOS",
                 font=(FONT, fs(7), "bold"), fg=DIM, bg=HEADER, justify="center").pack()
        tk.Frame(side, bg=HEADER).pack(expand=True)
        self.clock_lbl = tk.Label(side, text="", font=(FONT, fs(9)), fg=DIM, bg=HEADER)
        self.clock_lbl.pack(pady=sc(12))
        self._tick()

        # Área principal com grid para controle total de proporções
        main = tk.Frame(self, bg=BG)
        main.pack(side="left", expand=True, fill="both")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        bar = tk.Frame(main, bg=PANEL, pady=sc(13))
        bar.grid(row=0, column=0, sticky="ew")
        tk.Label(bar, text="AUTENTICAÇÃO BIOMÉTRICA FACIAL",
                 font=(FONT, fs(12), "bold"), fg=TEXT, bg=PANEL).pack(side="left", padx=sc(22))
        tk.Label(bar, text="v2.0  |  APS UNIP CC 2025/2",
                 font=(FONT, fs(8)), fg=DIM, bg=PANEL).pack(side="right", padx=sc(22))

        center = tk.Frame(main, bg=BG)
        center.grid(row=1, column=0, sticky="nsew", padx=sc(24), pady=sc(18))
        center.rowconfigure(0, weight=1)

        items = [
            ("AUTENTICAR",
             "Identificar-se e acessar o banco\nde dados conforme nível de acesso.",
             "🔑", self.app.go_auth, ACCENT, "#061829"),
            ("CADASTRAR USUÁRIO",
             "Registrar novo usuário com captura\nbiométrica via câmera ou arquivo.",
             "👤", self.app.go_register, GREEN2, "#071a10"),
            ("PAINEL ADMIN",
             "Visualizar usuários e histórico\ncompleto de acessos ao sistema.",
             "⚙", self.app.go_admin, GOLD, "#1a1000"),
        ]
        for i, (title, desc, icon, cmd, color, bg_c) in enumerate(items):
            center.columnconfigure(i, weight=1)
            c = tk.Frame(center, bg=bg_c, highlightbackground=color,
                         highlightthickness=1, cursor="hand2")
            c.grid(row=0, column=i, sticky="nsew", padx=sc(7), pady=sc(4))
            c.rowconfigure(1, weight=1); c.columnconfigure(0, weight=1)

            tk.Frame(c, bg=color, height=3).grid(row=0, column=0, sticky="ew")
            inner = tk.Frame(c, bg=bg_c)
            inner.grid(row=1, column=0, sticky="nsew", padx=sc(16), pady=sc(16))

            tk.Label(inner, text=icon, font=(FONT_UI, fs(26)), fg=color, bg=bg_c
                     ).pack(anchor="w")
            tk.Label(inner, text=title, font=(FONT, fs(10), "bold"), fg=color, bg=bg_c,
                     wraplength=sc(240), justify="left"
                     ).pack(anchor="w", pady=(sc(7), sc(3)))
            tk.Label(inner, text=desc, font=(FONT_UI, fs(9)), fg=DIM, bg=bg_c,
                     justify="left", wraplength=sc(240)
                     ).pack(anchor="w")
            tk.Frame(inner, bg=bg_c, height=sc(10)).pack()
            make_btn(inner, "ACESSAR  →", cmd, bg=color, size=9, pad=6).pack(anchor="w")

            for w in [c, inner] + list(inner.winfo_children()):
                try: w.bind("<Button-1>", lambda e, f=cmd: f())
                except: pass

        foot = tk.Frame(main, bg=PANEL, pady=sc(5))
        foot.grid(row=2, column=0, sticky="ew")
        tk.Label(foot, text="ACESSO RESTRITO — USO EXCLUSIVO DE SERVIDORES AUTORIZADOS DO MMA",
                 font=(FONT, fs(7)), fg=DIM, bg=PANEL).pack()

    def _tick(self):
        self.clock_lbl.config(text=datetime.now().strftime("%d/%m/%Y\n%H:%M:%S"))
        self.after(1000, self._tick)

# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
# ═══════════════════════════════════════════════════════════════════════════════
class AuthPage(Page):
    def __init__(self, app):
        super().__init__(app)
        self._users=[]; self._auth_nivel=None; self._auth_nome=None
        self._build()

    def _build(self):
        make_topbar(self, "AUTENTICAÇÃO BIOMÉTRICA", ACCENT, self.app.home)

        body = tk.Frame(self, bg=BG)
        body.pack(expand=True, fill="both", padx=sc(16), pady=sc(10))
        body.columnconfigure(0, weight=36)
        body.columnconfigure(1, weight=64)
        body.rowconfigure(0, weight=1)

        # ── Painel esquerdo (controles) ───────────────────────────────────────
        left = card(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, sc(9)))

        # ScrollFrame interno para telas muito pequenas
        lf = tk.Frame(left, bg=SURFACE)
        lf.pack(expand=True, fill="both")

        tk.Frame(lf, bg=SURFACE, height=sc(8)).pack()

        sec_label(lf, "USUÁRIO")
        self.user_var = tk.StringVar()
        self.user_cb  = ttk.Combobox(lf, textvariable=self.user_var,
                                      state="readonly", font=(FONT_UI, fs(10)))
        self.user_cb.pack(padx=sc(12), pady=sc(5), fill="x")

        sec_label(lf, "MODO DE CAPTURA")
        self.mode_var = tk.StringVar(value="camera")
        for txt, val in [("Câmera (webcam ao vivo)","camera"), ("Arquivo de imagem","file")]:
            tk.Radiobutton(lf, text=txt, variable=self.mode_var, value=val,
                           bg=SURFACE, fg=TEXT, activebackground=SURFACE,
                           selectcolor=BG, font=(FONT_UI, fs(10))
                           ).pack(anchor="w", padx=sc(16), pady=sc(2))

        self.file_path = None
        self.file_lbl  = tk.Label(lf, text="", font=(FONT_UI, fs(8)),
                                   fg=DIM, bg=SURFACE, wraplength=sc(240), anchor="w", justify="left")
        self.file_lbl.pack(anchor="w", padx=sc(16), fill="x")
        make_btn(lf, "📁  Selecionar arquivo", self._pick_file,
                 bg=PANEL, size=9, pad=5, bold=False
                 ).pack(padx=sc(12), pady=sc(5), fill="x")

        sep(lf, pady=sc(6))
        self.btn_auth = make_btn(lf, "◉  AUTENTICAR", self._authenticate, bg=ACCENT, size=11, pad=10)
        self.btn_auth.pack(fill="x", padx=sc(12), pady=sc(4))

        sep(lf, pady=sc(5))
        self.result_lbl = tk.Label(lf, text="", font=(FONT, fs(12), "bold"),
                                    bg=SURFACE, fg=TEXT, wraplength=sc(260), justify="center")
        self.result_lbl.pack(pady=(sc(5), sc(2)), padx=sc(6))
        self.detail_lbl = tk.Label(lf, text="", font=(FONT_UI, fs(9)),
                                    bg=SURFACE, fg=DIM, wraplength=sc(260), justify="center")
        self.detail_lbl.pack(pady=(0, sc(5)), padx=sc(6))

        sep(lf, pady=sc(4))
        self.btn_dados = make_btn(lf, "📂  ACESSAR BANCO DE DADOS", self._open_data,
                                   bg=GREEN2, size=9, pad=9)
        self.btn_dados.pack(fill="x", padx=sc(12), pady=sc(6))
        self.btn_dados.config(state="disabled")

        # ── Painel direito (câmera) ───────────────────────────────────────────
        right = card(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1); right.columnconfigure(0, weight=1)

        cam_hdr = tk.Frame(right, bg=SURFACE, pady=sc(7))
        cam_hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(cam_hdr, text="◉  CÂMERA  /  PREVIEW FACIAL",
                 font=(FONT, fs(9), "bold"), fg=ACCENT, bg=SURFACE
                 ).pack(side="left", padx=sc(12))

        cam_frame = tk.Frame(right, bg="#040a11")
        cam_frame.grid(row=1, column=0, sticky="nsew", padx=sc(6), pady=(0, sc(3)))
        self.cam = CameraWidget(cam_frame)

        btns = tk.Frame(right, bg=SURFACE, pady=sc(7))
        btns.grid(row=2, column=0)
        make_btn(btns, "▶  Iniciar câmera", self._start_cam, bg=ACCENT, size=9, pad=5
                 ).pack(side="left", padx=sc(5))
        make_btn(btns, "■  Parar", self._stop_cam, bg="#2a1212", size=9, pad=5
                 ).pack(side="left", padx=sc(5))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _pick_file(self):
        p = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens","*.jpg *.jpeg *.png *.bmp *.tiff")])
        if p:
            self.file_path = p
            self.file_lbl.config(text=f"  {os.path.basename(p)}")

    def _start_cam(self):
        if not self.cam.start():
            self._set_result("Câmera não encontrada", DIM, "")

    def _stop_cam(self): self.cam.stop()

    def _authenticate(self):
        idx = self.user_cb.current()
        if idx < 0:
            self._set_result("Selecione um usuário", DANGER, ""); return
        u = self._users[idx]
        uid, nome, nivel, mpath = u[0], u[1], u[3], u[4]
        face_roi = None

        if self.mode_var.get() == "camera":
            face_roi = self.cam.capture()
            if face_roi is None:
                self._set_result("Nenhum rosto detectado", DANGER,
                                  "Ative a câmera e posicione o rosto"); return
        else:
            if not self.file_path:
                self._set_result("Selecione um arquivo", DANGER, ""); return
            img = cv2.imread(self.file_path)
            if img is None:
                self._set_result("Erro ao abrir imagem", DANGER, ""); return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces, gp = detect_faces_file(gray)
            if len(faces) == 0:
                self._set_result("Nenhum rosto detectado", DANGER,
                                  "Verifique se a foto mostra o rosto com clareza"); return
            x, y, w, h = faces[0]
            face_roi = gp[y:y+h, x:x+w]

        ok, conf = recognize(face_roi, uid, mpath)
        db_log(uid, "SUCESSO" if ok else "FALHA", conf)
        if ok:
            self._auth_nivel = nivel; self._auth_nome = nome
            self._set_result("✔  ACESSO CONCEDIDO", SUCCESS,
                              f"Bem-vindo(a), {nome}\n{NIVEIS[nivel]}  |  Score: {conf:.1f}")
            self.btn_dados.config(state="normal")
        else:
            self._auth_nivel = None
            self._set_result("✘  ACESSO NEGADO", DANGER,
                              f"Biometria não reconhecida\nScore: {conf:.1f}  |  Limite: {CONF_MAX}")
            self.btn_dados.config(state="disabled")

    def _set_result(self, msg, cor, detalhe):
        self.result_lbl.config(text=msg, fg=cor)
        self.detail_lbl.config(text=detalhe)

    def _open_data(self):
        if self._auth_nivel:
            self.app.go_data(self._auth_nivel, self._auth_nome)

    def on_show(self):
        self._users = db_users(model_only=True)
        self.user_cb["values"] = [f"[{u[0]}]  {u[1]}  —  {NIVEIS[u[3]]}" for u in self._users]
        self.result_lbl.config(text=""); self.detail_lbl.config(text="")
        self.btn_dados.config(state="disabled"); self._auth_nivel = None

    def hide(self): self.cam.stop()
    def show(self): self.on_show(); super().show()

# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTER PAGE
# ═══════════════════════════════════════════════════════════════════════════════
class RegisterPage(Page):
    def __init__(self, app):
        super().__init__(app); self._build()

    def _build(self):
        make_topbar(self, "CADASTRO DE USUÁRIO", GREEN2, self.app.home)

        body = tk.Frame(self, bg=BG)
        body.pack(expand=True, fill="both", padx=sc(16), pady=sc(10))
        body.columnconfigure(0, weight=36)
        body.columnconfigure(1, weight=64)
        body.rowconfigure(0, weight=1)

        # ── Esquerda ──────────────────────────────────────────────────────────
        left = card(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, sc(9)))
        lf = tk.Frame(left, bg=SURFACE)
        lf.pack(expand=True, fill="both")

        tk.Frame(lf, bg=SURFACE, height=sc(8)).pack()

        sec_label(lf, "DADOS DO USUÁRIO")
        for lbl, attr in [("Nome completo","v_nome"), ("Cargo","v_cargo")]:
            setattr(self, attr, tk.StringVar())
            tk.Label(lf, text=lbl, font=(FONT_UI, fs(9)), fg=DIM, bg=SURFACE
                     ).pack(anchor="w", padx=sc(14), pady=(sc(5),sc(1)))
            make_entry(lf, getattr(self, attr)).pack(padx=sc(14), fill="x", ipady=sc(3))

        sec_label(lf, "NÍVEL DE ACESSO")
        self.nivel_var = tk.IntVar(value=1)
        for n, desc in [(1,"Público"),(2,"Diretor de Divisão"),(3,"Ministro")]:
            row = tk.Frame(lf, bg=SURFACE); row.pack(fill="x", padx=sc(14), pady=sc(2))
            tk.Radiobutton(row, text=f"  Nível {n} — {desc}", variable=self.nivel_var, value=n,
                           bg=SURFACE, fg=TEXT, activebackground=SURFACE,
                           selectcolor=BG, font=(FONT_UI, fs(10))
                           ).pack(side="left")
            tk.Label(row, text="●", fg=NIVEL_CORES[n], bg=SURFACE,
                     font=(FONT, fs(8))).pack(side="right", padx=sc(6))

        sec_label(lf, "MODO DE CAPTURA")
        self.mode2_var = tk.StringVar(value="camera")
        for txt, val in [("Câmera (webcam)","camera"), ("Arquivos de imagem","files")]:
            tk.Radiobutton(lf, text=txt, variable=self.mode2_var, value=val,
                           bg=SURFACE, fg=TEXT, activebackground=SURFACE,
                           selectcolor=BG, font=(FONT_UI, fs(10))
                           ).pack(anchor="w", padx=sc(16), pady=sc(2))

        fb = tk.Frame(lf, bg=SURFACE)
        fb.pack(fill="x", padx=sc(14), pady=sc(6))
        fb.columnconfigure(0, weight=1); fb.columnconfigure(1, weight=1)
        make_btn(fb, "▶  Câmera",    self._start_cam,  bg=ACCENT, size=9, pad=5
                 ).grid(row=0, column=0, sticky="ew", padx=(0, sc(4)))
        make_btn(fb, "📁  Arquivos", self._pick_files, bg=PANEL,  size=9, pad=5
                 ).grid(row=0, column=1, sticky="ew")

        sec_label(lf, "AMOSTRAS COLETADAS")
        self.prog_var = tk.IntVar(value=0)
        st = ttk.Style()
        st.configure("G.Horizontal.TProgressbar",
                      troughcolor=PANEL, background=GREEN2,
                      bordercolor=PANEL, lightcolor=GREEN2, darkcolor=GREEN2)
        ttk.Progressbar(lf, variable=self.prog_var, maximum=NUM_AMOSTRAS,
                         style="G.Horizontal.TProgressbar"
                         ).pack(padx=sc(14), pady=(sc(4),0), fill="x")
        self.prog_lbl = tk.Label(lf, text="0 / 40 amostras",
                                  font=(FONT_UI, fs(9)), fg=DIM, bg=SURFACE)
        self.prog_lbl.pack(anchor="w", padx=sc(14), pady=sc(2))

        self.status_lbl = tk.Label(lf, text="", font=(FONT_UI, fs(9), "bold"),
                                    bg=SURFACE, fg=DIM, wraplength=sc(270), justify="left")
        self.status_lbl.pack(padx=sc(14), pady=sc(5), anchor="w")

        sep(lf, pady=sc(4))
        make_btn(lf, "💾  CADASTRAR USUÁRIO", self._save, bg=GREEN2, size=10, pad=10
                 ).pack(fill="x", padx=sc(14), pady=sc(8))

        # ── Direita ───────────────────────────────────────────────────────────
        right = card(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1); right.columnconfigure(0, weight=1)

        cam_hdr = tk.Frame(right, bg=SURFACE, pady=sc(7))
        cam_hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(cam_hdr, text="◉  CÂMERA  /  COLETA DE AMOSTRAS",
                 font=(FONT, fs(9), "bold"), fg=GREEN2, bg=SURFACE
                 ).pack(side="left", padx=sc(12))

        cam_frame = tk.Frame(right, bg="#040a11")
        cam_frame.grid(row=1, column=0, sticky="nsew", padx=sc(6), pady=(0, sc(3)))
        self.cam = CameraWidget(cam_frame)
        self.cam.on_progress = self._on_progress

        make_btn(right, "■  Parar câmera", self._stop_cam, bg="#1a2a1a", size=9, pad=6
                 ).grid(row=2, column=0, pady=sc(8))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_progress(self, n):
        self.prog_var.set(n)
        self.prog_lbl.config(text=f"{n} / {NUM_AMOSTRAS} amostras")
        if n >= NUM_AMOSTRAS:
            self.status_lbl.config(text=f"✔ {NUM_AMOSTRAS} amostras coletadas!", fg=SUCCESS)
            self.cam.stop()

    def _start_cam(self):
        self.cam.reset(); self.prog_var.set(0)
        self.prog_lbl.config(text="0 / 40 amostras")
        self.status_lbl.config(text="Coletando amostras...", fg=DIM)
        if not self.cam.start():
            self.status_lbl.config(text="Câmera não encontrada", fg=DANGER)

    def _stop_cam(self): self.cam.stop()

    def _pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Selecionar imagens do rosto",
            filetypes=[("Imagens","*.jpg *.jpeg *.png *.bmp")])
        if not paths: return
        samples, erros = [], []
        for p in paths:
            img = cv2.imread(p)
            if img is None: erros.append(os.path.basename(p)); continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces, gp = detect_faces_file(gray)
            if len(faces) == 0: erros.append(os.path.basename(p)); continue
            for x,y,w,h in faces:
                samples.append(cv2.resize(gp[y:y+h, x:x+w], FACE_SZ))
        self.cam.samples = samples
        n = len(samples)
        self.prog_var.set(min(n, NUM_AMOSTRAS))
        self.prog_lbl.config(text=f"{n} amostras")
        if n:
            extra = f"\n  Sem rosto: {', '.join(erros)}" if erros else ""
            self.status_lbl.config(text=f"✔ {n} rosto(s) encontrado(s){extra}", fg=SUCCESS)
        else:
            self.status_lbl.config(
                text="Nenhum rosto detectado.\nVerifique se as fotos mostram rosto com clareza.",
                fg=DANGER)

    def _save(self):
        nome  = self.v_nome.get().strip()
        cargo = self.v_cargo.get().strip()
        nivel = self.nivel_var.get()
        samples = self.cam.samples
        if not nome or not cargo:
            self.status_lbl.config(text="Preencha nome e cargo", fg=DANGER); return
        if len(samples) < 5:
            self.status_lbl.config(
                text=f"Amostras insuficientes ({len(samples)}). Mínimo: 5", fg=DANGER); return
        self.cam.stop()
        uid   = db_add_user(nome, cargo, nivel)
        mpath = train_save(samples, uid)
        db_set_model(uid, mpath)
        self.status_lbl.config(
            text=f"✔ {nome} cadastrado!\n  ID #{uid}  |  {NIVEIS[nivel]}", fg=SUCCESS)
        self.v_nome.set(""); self.v_cargo.set("")
        self.cam.reset(); self.prog_var.set(0)
        self.prog_lbl.config(text="0 / 40 amostras")

    def hide(self): self.cam.stop()
    def show(self):
        self.cam.reset(); self.prog_var.set(0)
        self.prog_lbl.config(text="0 / 40 amostras")
        self.status_lbl.config(text=""); super().show()

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PAGE
# ═══════════════════════════════════════════════════════════════════════════════
class DataPage(Page):
    def __init__(self, app):
        super().__init__(app); self._build()

    def _build(self):
        bar = make_topbar(self, "BANCO DE DADOS — MMA", TEXT, self.app.go_auth, "←  VOLTAR")
        self.user_lbl = tk.Label(bar, text="", font=(FONT_UI, fs(10)), fg=GOLD, bg=PANEL)
        self.user_lbl.pack(side="right", padx=sc(16))

        outer = tk.Frame(self, bg=BG)
        outer.pack(expand=True, fill="both", padx=sc(14), pady=sc(10))
        outer.rowconfigure(0, weight=1); outer.columnconfigure(0, weight=1)

        cv = tk.Canvas(outer, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        cv.grid(row=0, column=0, sticky="nsew")

        self.data_frame = tk.Frame(cv, bg=BG)
        win = cv.create_window((0,0), window=self.data_frame, anchor="nw")
        self.data_frame.bind("<Configure>",
            lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(win, width=e.width))
        cv.bind_all("<MouseWheel>",
            lambda e: cv.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._cv = cv

    def load(self, nivel, nome):
        self.user_lbl.config(text=f"◉  {nome}  |  {NIVEIS[nivel]}")
        for w in self.data_frame.winfo_children(): w.destroy()

        for n in range(1, 4):
            d    = MMA_DATA[n]
            cor  = d["cor"]
            ativo= (n <= nivel)
            bg_c = SURFACE if ativo else "#0b1420"

            sec = tk.Frame(self.data_frame, bg=bg_c,
                            highlightbackground=cor if ativo else BORDER, highlightthickness=1)
            sec.pack(fill="x", pady=sc(5), padx=sc(6))
            tk.Frame(sec, bg=cor if ativo else BORDER, height=3).pack(fill="x")

            hdr = tk.Frame(sec, bg=bg_c, pady=sc(9))
            hdr.pack(fill="x", padx=sc(10))
            tk.Label(hdr, text=d["icone"], font=(FONT, fs(15)),
                     fg=cor if ativo else BORDER, bg=bg_c).pack(side="left")
            tk.Label(hdr, text=f"  {d['titulo']}", font=(FONT, fs(10), "bold"),
                     fg=cor if ativo else BORDER, bg=bg_c,
                     wraplength=sc(580), justify="left").pack(side="left")
            if not ativo:
                tk.Label(hdr, text="🔒  ACESSO NÃO AUTORIZADO",
                         font=(FONT, fs(8)), fg=BORDER, bg=bg_c).pack(side="right")
                continue

            cf = tk.Frame(sec, bg=bg_c)
            cf.pack(fill="x", padx=sc(20), pady=(0, sc(10)))

            for row in d["conteudo"]:
                tipo = row[1] if len(row)>1 else ""
                if not row[0]:
                    tk.Frame(cf, bg=bg_c, height=sc(4)).pack()
                elif tipo == "titulo":
                    tk.Label(cf, text=row[0], font=(FONT, fs(10),"bold"),
                             fg=TEXT, bg=bg_c, anchor="w", wraplength=sc(680)
                             ).pack(fill="x")
                    tk.Frame(cf, bg=BORDER, height=1).pack(fill="x", pady=sc(3))
                elif tipo == "item":
                    f = tk.Frame(cf, bg=bg_c); f.pack(fill="x", pady=sc(1))
                    f.columnconfigure(0, weight=1)
                    tk.Label(f, text=f"  {row[0]}", font=(FONT_UI, fs(10)),
                             fg=DIM, bg=bg_c, anchor="w", wraplength=sc(450)
                             ).grid(row=0, column=0, sticky="ew")
                    if len(row)>2 and row[2]:
                        tk.Label(f, text=row[2], font=(FONT, fs(10),"bold"),
                                 fg=TEXT, bg=bg_c, anchor="e"
                                 ).grid(row=0, column=1, sticky="e", padx=sc(6))
                elif tipo == "cabecalho":
                    f = tk.Frame(cf, bg="#0d1e30"); f.pack(fill="x", pady=(sc(3),0))
                    f.columnconfigure(0, weight=1); f.columnconfigure(1, weight=1); f.columnconfigure(2, weight=3)
                    for ci, col in enumerate([row[0], row[2], row[3]]):
                        tk.Label(f, text=col, font=(FONT, fs(8),"bold"), fg=cor, bg="#0d1e30",
                                 anchor="w").grid(row=0, column=ci, sticky="ew",
                                                   padx=sc(5), pady=sc(3))
                elif tipo == "linha":
                    f = tk.Frame(cf, bg=SURFACE); f.pack(fill="x")
                    f.columnconfigure(0, weight=1); f.columnconfigure(1, weight=1); f.columnconfigure(2, weight=3)
                    tk.Label(f, text=row[0], font=(FONT, fs(10),"bold"), fg=TEXT, bg=SURFACE,
                             anchor="w").grid(row=0, column=0, sticky="w", padx=sc(5), pady=sc(2))
                    tk.Label(f, text=row[2], font=(FONT_UI, fs(10)), fg=TEXT, bg=SURFACE,
                             anchor="w").grid(row=0, column=1, sticky="ew", padx=sc(4))
                    tk.Label(f, text=row[3], font=(FONT_UI, fs(10)), fg=DIM, bg=SURFACE,
                             anchor="w", wraplength=sc(300)
                             ).grid(row=0, column=2, sticky="ew", padx=sc(4))
                elif tipo == "op":
                    f = tk.Frame(cf, bg=bg_c); f.pack(fill="x", pady=sc(2))
                    f.columnconfigure(1, weight=1)
                    tk.Label(f, text=f"  {row[0]}", font=(FONT, fs(9),"bold"), fg=cor, bg=bg_c,
                             anchor="w").grid(row=0, column=0, sticky="w", padx=sc(4))
                    tk.Label(f, text=row[2], font=(FONT_UI, fs(10)), fg=TEXT, bg=bg_c,
                             anchor="w", wraplength=sc(400)
                             ).grid(row=0, column=1, sticky="ew", padx=sc(6))
                elif tipo == "prop":
                    f = tk.Frame(cf, bg=SURFACE); f.pack(fill="x", pady=sc(1))
                    f.columnconfigure(0, weight=1); f.columnconfigure(1, weight=2); f.columnconfigure(2, weight=2)
                    tk.Label(f, text=f"  {row[0]}", font=(FONT, fs(9),"bold"), fg=cor, bg=SURFACE,
                             anchor="w").grid(row=0, column=0, sticky="ew", padx=sc(4), pady=sc(2))
                    tk.Label(f, text=row[2], font=(FONT_UI, fs(10)), fg=TEXT, bg=SURFACE,
                             anchor="w", wraplength=sc(200)
                             ).grid(row=0, column=1, sticky="ew", padx=sc(4))
                    tk.Label(f, text=row[3], font=(FONT_UI, fs(10)), fg=DANGER, bg=SURFACE,
                             anchor="w", wraplength=sc(200)
                             ).grid(row=0, column=2, sticky="ew", padx=sc(4))
                elif tipo == "aviso":
                    tk.Label(cf, text=row[0], font=(FONT, fs(9),"bold"), fg=cor, bg=bg_c,
                             anchor="w", wraplength=sc(680)
                             ).pack(fill="x", pady=sc(4))
                elif tipo == "rodape":
                    tk.Label(cf, text=row[0], font=(FONT_UI, fs(8),"italic"), fg=DIM, bg=bg_c,
                             anchor="w", wraplength=sc(680)
                             ).pack(fill="x")

# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════
class AdminPage(Page):
    def __init__(self, app):
        super().__init__(app); self._build()

    def _build(self):
        bar = make_topbar(self, "PAINEL ADMINISTRATIVO", GOLD, self.app.home)
        make_btn(bar, "↺  Atualizar", self._refresh, bg=PANEL, fg=DIM, size=9, pad=5, bold=False
                 ).pack(side="right", padx=sc(12))

        nb = ttk.Notebook(self)
        nb.pack(expand=True, fill="both", padx=sc(14), pady=sc(10))

        s = ttk.Style()
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=SURFACE, foreground=TEXT,
                     font=(FONT_UI, fs(10)), padding=[sc(12), sc(5)])
        s.map("TNotebook.Tab",
              background=[("selected", GOLD)], foreground=[("selected","#000")])
        s.configure("Treeview", background=SURFACE, foreground=TEXT,
                     fieldbackground=SURFACE, rowheight=max(22, sc(27)),
                     font=(FONT_UI, fs(10)), borderwidth=0)
        s.configure("Treeview.Heading", background=PANEL, foreground=DIM,
                     font=(FONT, fs(8),"bold"), relief="flat")
        s.map("Treeview",
              background=[("selected", ACCENT)], foreground=[("selected", TEXT)])

        def make_tree(parent, cols, minws):
            f = tk.Frame(parent, bg=BG)
            f.pack(expand=True, fill="both")
            tree = ttk.Treeview(f, columns=cols, show="headings")
            for col in cols:
                tree.heading(col, text=col)
                # stretch=True → divide o espaço restante entre colunas
                tree.column(col, minwidth=sc(minws[col]),
                             width=sc(minws[col]), stretch=True, anchor="center")
            sb = ttk.Scrollbar(f, command=tree.yview)
            tree.configure(yscrollcommand=sb.set)
            tree.pack(side="left", expand=True, fill="both",
                      padx=(sc(7),0), pady=sc(7))
            sb.pack(side="left", fill="y", pady=sc(7), padx=(0,sc(7)))
            return tree

        f1 = tk.Frame(nb, bg=BG)
        nb.add(f1, text="  👤  Usuários Cadastrados  ")
        self.tree_users = make_tree(f1,
            ("ID","Nome","Cargo","Nível de Acesso","Biometria"),
            {"ID":42,"Nome":140,"Cargo":130,"Nível de Acesso":180,"Biometria":85})
        self.tree_users.tag_configure("n1", foreground=ACCENT)
        self.tree_users.tag_configure("n2", foreground=GOLD)
        self.tree_users.tag_configure("n3", foreground=DANGER)

        f2 = tk.Frame(nb, bg=BG)
        nb.add(f2, text="  📋  Logs de Acesso  ")
        self.tree_logs = make_tree(f2,
            ("Data / Hora","Usuário","Resultado","Score LBPH"),
            {"Data / Hora":145,"Usuário":170,"Resultado":105,"Score LBPH":95})
        self.tree_logs.tag_configure("ok",   foreground=SUCCESS)
        self.tree_logs.tag_configure("fail", foreground=DANGER)

    def _refresh(self):
        self.tree_users.delete(*self.tree_users.get_children())
        for u in db_users():
            self.tree_users.insert("", "end",
                values=(u[0], u[1], u[2],
                        NIVEIS.get(u[3], str(u[3])),
                        "✔  SIM" if u[4] else "✘  NÃO"),
                tags=(f"n{u[3]}",))

        self.tree_logs.delete(*self.tree_logs.get_children())
        for log in db_logs():
            hora   = (log[0] or "—")[:19]
            nome   = log[1] or "—"
            result = log[2] or "—"
            score  = f"{log[3]:.1f}" if log[3] is not None else "—"
            self.tree_logs.insert("", "end",
                values=(hora, nome, result, score),
                tags=("ok" if result=="SUCESSO" else "fail",))

    def show(self): self._refresh(); super().show()

# ═══════════════════════════════════════════════════════════════════════════════
#  APLICAÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema Biométrico — MMA")
        # Centraliza na tela com tamanho calculado pela resolução detectada
        x = (_SW - WIN_W) // 2
        y = (_SH - WIN_H) // 2
        self.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
        self.minsize(sc(740), sc(500))
        self.configure(bg=BG)

        self.container = tk.Frame(self, bg=BG)
        self.container.pack(expand=True, fill="both")

        self._pages = {}
        for Cls in (HomePage, AuthPage, RegisterPage, DataPage, AdminPage):
            p = Cls(self)
            p.place(in_=self.container, relx=0, rely=0, relwidth=1, relheight=1)
            self._pages[Cls] = p

        self._current = None
        self.home()

    def _go(self, cls):
        if self._current and hasattr(self._current, "hide"):
            self._current.hide()
        self._current = self._pages[cls]
        self._current.show()

    def home(self):        self._go(HomePage)
    def go_auth(self):     self._go(AuthPage)
    def go_register(self): self._go(RegisterPage)
    def go_admin(self):    self._go(AdminPage)

    def go_data(self, nivel, nome):
        self._pages[DataPage].load(nivel, nome)
        self._go(DataPage)


if __name__ == "__main__":
    init_db()
    App().mainloop()