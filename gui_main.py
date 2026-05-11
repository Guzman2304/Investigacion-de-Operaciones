import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
import time
import random
import threading
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===============================
# CONFIGURACIÓN Y TEMA "CYBERPUNK"
# ===============================
BG_COLOR = "#0b0e14"      # Fondo profundo (Casi negro)
CARD_COLOR = "#161b22"    # Fondo de tarjetas (Gris azulado muy oscuro)
ACCENT_CYAN = "#00f2ff"   # Cyan neón para Exacto
ACCENT_PURPLE = "#bc00ff" # Púrpura neón para GA
ACCENT_GREEN = "#39ff14"  # Verde neón para Éxito
TEXT_COLOR = "#e6edf3"    # Texto claro
SUB_TEXT_COLOR = "#8b949e" # Texto secundario

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "set_cover_500x500.csv")
xlsx_path = os.path.join(base_dir, "Costo_S.xlsx")

def cargar_datos():
    if not os.path.exists(csv_path) or not os.path.exists(xlsx_path):
        return None, None
    
    # 1. Cargar Excel de Costos (Viene horizontal: Conjunto, 1, 2, 3...)
    # Saltamos la primera columna 'Conjunto' y tomamos los valores numéricos
    costos_df = pd.read_excel(xlsx_path)
    # iloc[0, 1:] toma la primera fila, desde la segunda columna en adelante
    costos = pd.to_numeric(costos_df.iloc[0, 1:], errors='coerce').values
    
    # 2. Cargar CSV de Cobertura (500x500)
    # Al parecer la primera fila son los IDs de los clientes (0, 1, 2... 499)
    # Necesitamos los datos reales a partir de la fila 1
    matriz_df = pd.read_csv(csv_path, header=None)
    # Saltamos la primera fila (encabezados de clientes) y convertimos a numérico
    matriz = matriz_df.iloc[1:, :].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # Asegurar que las dimensiones coincidan (500 antenas x 500 clientes)
    # matriz debe ser (n_antenas, n_clientes) -> (500, 500)
    # costos debe ser (n_antenas,) -> (500,)
    
    return matriz, costos

# ===============================
# CLASE PARA TOOLTIPS (DIDÁCTICO)
# ===============================
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                       background="#30363d", foreground=TEXT_COLOR,
                       relief='flat', borderwidth=1,
                       font=("Segoe UI", "9", "normal"), padx=10, pady=5)
        label.pack(side='bottom')

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
        self.tip_window = None

# ===============================
# LÓGICA DE OPTIMIZACIÓN
# ===============================

def ejecutar_exacto(A, costos):
    n_antenas, n_clientes = A.shape
    inicio = time.time()
    modelo = LpProblem("Set_Cover", LpMinimize)
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n_antenas)]
    modelo += lpSum(costos[i] * x[i] for i in range(n_antenas))
    for j in range(n_clientes):
        modelo += lpSum(A[i][j] * x[i] for i in range(n_antenas)) >= 1
    
    # Añadimos un tiempo límite de 30 segundos para evitar que se quede pegado
    # En problemas de 500x500, encontrar el óptimo global puede tardar horas,
    # pero encontrar una solución muy buena tarda segundos.
    modelo.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
    
    fin = time.time()
    solucion = [i for i in range(n_antenas) if x[i].value() == 1]
    return {
        "costo": sum(costos[i] for i in solucion),
        "antenas": len(solucion),
        "tiempo": fin - inicio
    }

def ejecutar_ga(A, costos, callback_progress=None, callback_fitness=None):
    n_antenas, n_clientes = A.shape
    POP_SIZE = 50 # Reducimos un poco el tamaño para ganar velocidad
    GENERACIONES = 100 # Subimos a 100 para dar más margen de mejora
    PROB_CRUCE = 0.8
    PROB_MUTACION = 0.02 # Subimos un poco para evitar estancamiento
    M = 10000 # Penalización más fuerte para asegurar cobertura

    def fitness(ind):
        # Usamos numpy para que sea veloz como un rayo
        ind_np = np.array(ind)
        costo = np.dot(ind_np, costos)
        cobertura = np.dot(ind_np, A)
        no_cubiertos = np.sum(cobertura < 1)
        return costo + M * no_cubiertos + np.sum(ind_np) * 0.01

    def reparar(ind):
        ind_np = np.array(ind)
        cobertura = np.dot(ind_np, A)
        
        # Solo revisamos las que están activas para ir más rápido
        activas = np.where(ind_np == 1)[0]
        # Mezclamos para que el algoritmo no sea determinista y explore mejor
        np.random.shuffle(activas)
        
        for i in activas:
            # Si quitamos la antena i y todo sigue cubierto (cobertura > 1 en sus zonas)
            if np.all(cobertura[A[i] > 0] > 1):
                cobertura -= A[i]
                ind_np[i] = 0
        return ind_np.tolist()

    # Población inicial con reparación
    poblacion = [reparar([random.randint(0, 1) for _ in range(n_antenas)]) for _ in range(POP_SIZE)]
    mejores_fitness = []
    
    for gen in range(GENERACIONES):
        nueva_poblacion = []
        # Elitismo: mantenemos a los 2 mejores
        poblacion.sort(key=lambda x: fitness(x))
        nueva_poblacion.extend(poblacion[:2])
        
        while len(nueva_poblacion) < POP_SIZE:
            # Torneo
            p1 = min(random.sample(poblacion, 3), key=lambda x: fitness(x))
            p2 = min(random.sample(poblacion, 3), key=lambda x: fitness(x))
            
            # Cruce
            if random.random() < PROB_CRUCE:
                pt = random.randint(1, n_antenas - 1)
                h1, h2 = p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
            else:
                h1, h2 = p1[:], p2[:]
            
            # Mutación y Reparación
            for h in [h1, h2]:
                if len(nueva_poblacion) < POP_SIZE:
                    for i in range(n_antenas):
                        if random.random() < PROB_MUTACION: h[i] = 1 - h[i]
                    nueva_poblacion.append(reparar(h))
        
        poblacion = nueva_poblacion
        mejor_gen = min(poblacion, key=lambda x: fitness(x))
        f_val = fitness(mejor_gen)
        mejores_fitness.append(f_val)
        
        # Actualizamos la UI cada 2 generaciones para no saturar el hilo principal
        if gen % 2 == 0 or gen == GENERACIONES - 1:
            if callback_progress: callback_progress(gen + 1, GENERACIONES)
            if callback_fitness: callback_fitness(mejores_fitness)
        
    mejor = min(poblacion, key=lambda x: fitness(x))
    return {
        "costo": sum(costos[i] for i in range(n_antenas) if mejor[i] == 1),
        "antenas": sum(mejor),
        "historial_fitness": mejores_fitness
    }

# ===============================
# INTERFAZ GRÁFICA (HIGH-TECH)
# ===============================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Panel de Control - Optimización de Antenas")
        self.root.geometry("1200x850")
        self.root.configure(bg=BG_COLOR)
        
        self.last_res_exacto = None
        self.last_res_ga = None
        self.is_running = False
        self.start_time = 0
        
        # Pre-cargar datos para evitar lag al presionar el botón
        self.A, self.costos = cargar_datos()

        self.setup_styles()
        self.create_layout()
        
        # Validar carga inicial
        if self.A is not None:
            self.lbl_status.config(text="Base de datos lista. ¡Hágale, patrón!", fg=ACCENT_GREEN)
        else:
            self.lbl_status.config(text="¡Pilas! No encontré los archivos de datos.", fg="#ef4444")
            self.btn_run.config(state="disabled", bg="#334155")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Pestañas
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=CARD_COLOR, foreground=SUB_TEXT_COLOR, 
                       padding=[20, 10], font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", BG_COLOR)], foreground=[("selected", ACCENT_CYAN)])

        # Treeview
        style.configure("Treeview", background=CARD_COLOR, foreground=TEXT_COLOR, fieldbackground=CARD_COLOR, 
                       borderwidth=0, font=("Segoe UI", 10), rowheight=30)
        style.configure("Treeview.Heading", background="#21262d", foreground=TEXT_COLOR, relief="flat", font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[('selected', "#1f6feb")])
        
        style.configure("TProgressbar", thickness=10, troughcolor=CARD_COLOR, background=ACCENT_CYAN, borderwidth=0)

    def create_layout(self):
        # Sidebar
        sidebar = tk.Frame(self.root, bg=CARD_COLOR, width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        
        tk.Label(sidebar, text="CYBER OPTIMIZER", font=("Orbitron", 16, "bold"), bg=CARD_COLOR, fg=ACCENT_CYAN).pack(pady=(50, 40))
        
        self.btn_run = tk.Button(sidebar, text="¡HÁGALE, ANALIZAR!", command=self.run_optimization, 
                                bg=ACCENT_CYAN, fg=BG_COLOR, font=("Segoe UI", 11, "bold"), 
                                relief="flat", padx=20, pady=15, cursor="hand2", activebackground="#7dd3fc")
        self.btn_run.pack(pady=10, padx=30, fill="x")
        ToolTip(self.btn_run, "Inicia el procesamiento simultáneo de ambos métodos de optimización.")

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(sidebar, variable=self.progress_var, maximum=100, style="TProgressbar")
        self.progress.pack(pady=(20, 5), padx=30, fill="x")

        self.lbl_timer = tk.Label(sidebar, text="Tiempo: 0.0s", bg=CARD_COLOR, fg=ACCENT_CYAN, font=("Segoe UI", 12, "bold"))
        self.lbl_timer.pack(pady=2)

        self.lbl_estimado = tk.Label(sidebar, text="Estimado: --", bg=CARD_COLOR, fg=SUB_TEXT_COLOR, font=("Segoe UI", 10))
        self.lbl_estimado.pack(pady=(0, 10))
        
        self.lbl_status = tk.Label(sidebar, text="Todo listo, patrón", bg=CARD_COLOR, fg=SUB_TEXT_COLOR, font=("Segoe UI", 10), wraplength=220)
        self.lbl_status.pack(pady=10, padx=20)

        # Main Content with Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side="right", expand=True, fill="both", padx=20, pady=20)

        # TAB 1: RESUMEN EJECUTIVO
        self.tab_resumen = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(self.tab_resumen, text="RESUMEN EJECUTIVO")
        self.setup_tab_resumen()

        # TAB 2: ANÁLISIS TÉCNICO
        self.tab_tecnico = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(self.tab_tecnico, text="ANÁLISIS TÉCNICO")
        self.setup_tab_tecnico()

    def setup_tab_resumen(self):
        tk.Label(self.tab_resumen, text="Panel de Optimización Estratégica", font=("Segoe UI", 24, "bold"), 
                 bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor="w", pady=(20, 30), padx=20)

        # KPI Frame
        kpi_container = tk.Frame(self.tab_resumen, bg=BG_COLOR)
        kpi_container.pack(fill="x", pady=10, padx=10)
        
        self.lbl_cost = self.create_kpi_card(kpi_container, "COSTO MÁS BARATO", "0.0000", ACCENT_GREEN, 
                                            "Mínimo costo total calculado para cubrir a todos los clientes.")
        self.lbl_antennas = self.create_kpi_card(kpi_container, "ANTENAS TOTALES", "0", ACCENT_CYAN, 
                                               "Cantidad de antenas activadas en la solución óptima.")
        self.lbl_time = self.create_kpi_card(kpi_container, "TIEMPO GASTADO", "0.00s", ACCENT_PURPLE, 
                                            "Tiempo total que tomó procesar ambos algoritmos.")

        # Comparison Charts (Bar Charts)
        charts_frame = tk.Frame(self.tab_resumen, bg=BG_COLOR)
        charts_frame.pack(expand=True, fill="both", pady=20)
        
        plt.style.use('dark_background')
        self.fig_res, (self.ax_bar1, self.ax_bar2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig_res.patch.set_facecolor(BG_COLOR)
        for ax in [self.ax_bar1, self.ax_bar2]:
            ax.set_facecolor(CARD_COLOR)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        self.canvas_res = FigureCanvasTkAgg(self.fig_res, master=charts_frame)
        self.canvas_res.get_tk_widget().pack(expand=True, fill="both")

    def setup_tab_tecnico(self):
        # Live Evolution Chart (Full Width)
        evolution_frame = tk.LabelFrame(self.tab_tecnico, text="Evolución del Algoritmo Genético (En Vivo)", 
                                       bg=BG_COLOR, fg=ACCENT_PURPLE, font=("Segoe UI", 11, "bold"), pady=10)
        evolution_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.fig_evo, self.ax_evo = plt.subplots(figsize=(10, 4))
        self.fig_evo.patch.set_facecolor(BG_COLOR)
        self.ax_evo.set_facecolor(CARD_COLOR)
        self.ax_evo.set_title("HISTORIAL DE FITNESS POR GENERACIÓN", color=TEXT_COLOR, fontsize=10)
        
        self.canvas_evo = FigureCanvasTkAgg(self.fig_evo, master=evolution_frame)
        self.canvas_evo.get_tk_widget().pack(expand=True, fill="both")

        # Data Table
        table_frame = tk.Frame(self.tab_tecnico, bg=BG_COLOR)
        table_frame.pack(fill="x", padx=20, pady=20)
        
        self.tree = ttk.Treeview(table_frame, columns=("Metodo", "Costo", "Antenas", "Tiempo"), show="headings", height=3)
        for col, text in zip(("Metodo", "Costo", "Antenas", "Tiempo"), ("Método", "Costo Total", "Antenas", "Tiempo (s)")):
            self.tree.heading(col, text=text)
            self.tree.column(col, anchor="center")
        self.tree.pack(fill="x")

    def create_kpi_card(self, parent, title, value, color, tooltip):
        card = tk.Frame(parent, bg=CARD_COLOR, padx=25, pady=25, highlightthickness=1, highlightbackground="#30363d")
        card.pack(side="left", expand=True, fill="both", padx=10)
        lbl_title = tk.Label(card, text=title, font=("Segoe UI", 10, "bold"), bg=CARD_COLOR, fg=SUB_TEXT_COLOR)
        lbl_title.pack(anchor="w")
        lbl_val = tk.Label(card, text=value, font=("Segoe UI", 24, "bold"), bg=CARD_COLOR, fg=color)
        lbl_val.pack(anchor="w", pady=(10, 0))
        ToolTip(card, tooltip)
        return lbl_val

    def update_live_fitness(self, history):
        self.ax_evo.clear()
        self.ax_evo.set_facecolor(CARD_COLOR)
        
        # Dibujar historial del GA
        self.ax_evo.plot(history, color=ACCENT_GREEN, linewidth=2, label="Progreso GA")
        self.ax_evo.fill_between(range(len(history)), history, color=ACCENT_GREEN, alpha=0.1)
        
        # Dibujar línea del óptimo exacto (si existe)
        if self.last_res_exacto:
            self.ax_evo.axhline(y=self.last_res_exacto['costo'], color="#ff4444", 
                               linestyle="--", alpha=0.6, label="Óptimo Exacto")
            self.ax_evo.legend(facecolor=CARD_COLOR, edgecolor=SUB_TEXT_COLOR, labelcolor=TEXT_COLOR)

        self.ax_evo.set_title("CONVERGENCIA HACIA EL ÓPTIMO", color=TEXT_COLOR, fontsize=10, fontweight="bold")
        self.ax_evo.set_xlabel("Generación", color=SUB_TEXT_COLOR)
        self.ax_evo.set_ylabel("Costo / Fitness", color=SUB_TEXT_COLOR)
        self.ax_evo.grid(True, alpha=0.05, color=TEXT_COLOR)
        self.root.after(0, self.canvas_evo.draw)

    def update_progress(self, current, total):
        val = (current / total) * 100
        self.root.after(0, lambda: self.progress_var.set(val))
        self.root.after(0, lambda: self.lbl_status.config(text=f"Generación {current}/{total} camellando..."))
        
        # Calcular tiempo estimado
        elapsed = time.time() - self.start_time
        if current > 1:
            # Una estimación simple basada en el progreso lineal
            total_est = (elapsed / current) * total
            remaining = total_est - elapsed
            if remaining > 0:
                self.root.after(0, lambda: self.lbl_estimado.config(text=f"Estimado: {remaining:.1f}s faltan"))

    def update_timer(self):
        if self.is_running:
            elapsed = time.time() - self.start_time
            self.lbl_timer.config(text=f"Tiempo: {elapsed:.1f}s")
            self.root.after(100, self.update_timer)

    def run_optimization(self):
        # 1. Prioridad inmediata: Feedback visual al usuario
        self.is_running = True
        self.start_time = time.time()
        self.btn_run.config(state="disabled", text="CAMELLANDO...", bg="#21262d")
        self.lbl_status.config(text="Prendiendo motores...", fg=ACCENT_CYAN)
        self.lbl_estimado.config(text="Estimado: Calculando...")
        self.progress_var.set(0)
        self.progress.config(mode="indeterminate")
        self.progress.start(10)
        self.update_timer()
        
        # 2. Lanzar el hilo de optimización inmediatamente
        threading.Thread(target=self._optimization_thread, daemon=True).start()

    def _optimization_thread(self):
        try:
            # 3. Limpiar gráficas dentro del hilo de forma segura
            self.root.after(0, self._clear_charts_ui)
            
            # Los datos ya están cargados en self.A y self.costos desde __init__
            self.root.after(0, lambda: [self.tree.delete(item) for item in self.tree.get_children()])

            # 1. Exacto
            self.root.after(0, lambda: self.lbl_status.config(text="Sacando el resultado exacto...", fg=ACCENT_CYAN))
            res_exacto = ejecutar_exacto(self.A, self.costos)
            self.last_res_exacto = res_exacto # Guardamos para la gráfica
            self.root.after(0, lambda: self.tree.insert("", "end", values=("Exacto", f"{res_exacto['costo']:.4f}", res_exacto['antenas'], f"{res_exacto['tiempo']:.4f}")))

            # 2. GA (Con animación)
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.progress.config(mode="determinate"))
            self.root.after(0, lambda: self.lbl_status.config(text="Ahora va el genético...", fg=ACCENT_PURPLE))
            
            # Reiniciamos el timer para el GA si queremos medir solo el GA, 
            # pero el usuario pidió el tiempo total del proceso.
            
            res_ga = ejecutar_ga(self.A, self.costos, callback_progress=self.update_progress, callback_fitness=self.update_live_fitness)
            # El tiempo ya se mide con el start_time global
            res_ga['tiempo'] = time.time() - self.start_time - res_exacto['tiempo']
            self.root.after(0, lambda: self.tree.insert("", "end", values=("Genético", f"{res_ga['costo']:.4f}", res_ga['antenas'], f"{res_ga['tiempo']:.4f}")))

            self.root.after(0, lambda: self._finalize_results(res_exacto, res_ga))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("¡Falla técnica!", str(e)))
            self.root.after(0, self._reset_ui)

    def _clear_charts_ui(self):
        # Reset charts de forma segura para evitar lag inicial
        self.ax_evo.clear()
        self.ax_evo.set_facecolor(CARD_COLOR)
        self.ax_evo.set_title("CONVERGENCIA HACIA EL ÓPTIMO", color=TEXT_COLOR, fontsize=10, fontweight="bold")
        self.canvas_evo.draw()
        
        self.ax_bar1.clear()
        self.ax_bar2.clear()
        self.canvas_res.draw()

    def _finalize_results(self, res_exacto, res_ga):
        # Update KPIs
        self.lbl_cost.config(text=f"{res_exacto['costo']:.4f}")
        self.lbl_antennas.config(text=f"{res_exacto['antenas']}")
        self.lbl_time.config(text=f"{time.time() - self.start_time:.2f}s")

        # Update Comparison Charts (Tab 1)
        self.ax_bar1.clear()
        self.ax_bar2.clear()
        
        self.ax_bar1.bar(['Exacto', 'GA'], [res_exacto['costo'], res_ga['costo']], color=[ACCENT_CYAN, ACCENT_PURPLE])
        self.ax_bar1.set_title("COMPARACIÓN DE COSTOS", fontsize=9, color=SUB_TEXT_COLOR)
        
        self.ax_bar2.bar(['Exacto', 'GA'], [res_exacto['tiempo'], res_ga['tiempo']], color=[ACCENT_CYAN, ACCENT_PURPLE])
        self.ax_bar2.set_title("TIEMPO GASTADO (S)", fontsize=9, color=SUB_TEXT_COLOR)

        self.canvas_res.draw()
        
        self.lbl_status.config(text="¡Análisis finalizado con éxito!", fg=ACCENT_GREEN)
        self.lbl_estimado.config(text="Estimado: Finalizado")
        self._reset_ui()
        messagebox.showinfo("¡Bien hecho!", "Optimización completada. Revise las pestañas para el detalle técnico.")

    def _reset_ui(self):
        self.is_running = False
        self.btn_run.config(state="normal", text="¡HÁGALE, ANALIZAR!", bg=ACCENT_CYAN)
        self.progress.stop()
        self.progress.config(mode="determinate")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
