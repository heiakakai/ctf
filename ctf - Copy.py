import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk, ImageOps
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.interpolate import PchipInterpolator
import pandas as pd
import datetime

# ==========================================
# 1. 핵심 알고리즘 (CTF -> MTF -> DQE)
# ==========================================
def calculate_coltman_mtf(freq_axis, measured_freqs, measured_ctfs, pixel_pitch=0.076):
    """
    Coltman Equation을 사용하여 다지점 CTF를 MTF로 정밀 변환
    """
    # 1) CTF 보간 (Interpolation)
    # Nyquist Frequency (ex: 0.076mm -> ~6.57 lp/mm)
    nyquist = 1.0 / (2 * pixel_pitch)
    
    # 0점(100%)과 Nyquist(0%) 추가하여 곡선 완성
    fit_freqs = [0.0] + sorted(measured_freqs) + [nyquist]
    fit_vals = [1.0] + [v/100.0 for v in sorted(measured_ctfs, reverse=True)] + [0.0]
    
    # PchipInterpolator: 튀는 현상 없이 단조 감소(Monotonic) 유지
    ctf_interp = PchipInterpolator(fit_freqs, fit_vals)
    
    # 2) Coltman Series Calculation
    mtf_curve = []
    for f in freq_axis:
        if f == 0:
            mtf_curve.append(1.0)
            continue
            
        # Series: CTF(f) + CTF(3f)/3 - CTF(5f)/5 + CTF(7f)/7 ...
        sum_val = 0.0
        sum_val += float(ctf_interp(f)) # Fundamental
        
        # Harmonics (Odd numbers: 3, 5, 7, 9...)
        # Signs pattern: 
        # k=1(+)
        # k=3(+)
        # k=5(-)
        # k=7(+)
        # k=9(-)
        # Formula sign: (-1)^(m+1) where m=(k-1)/2
        
        for k in range(3, 22, 2): # 충분한 차수까지 계산
            term_freq = k * f
            if term_freq > nyquist:
                break
                
            val = float(ctf_interp(term_freq))
            m = (k - 1) // 2
            sign = (-1)**(m + 1)
            
            sum_val += sign * (val / k)
            
        mtf_val = (np.pi / 4.0) * sum_val
        mtf_curve.append(max(0.0, mtf_val)) # 음수 방지
        
    return np.array(mtf_curve), fit_freqs, fit_vals

# ==========================================
# 2. GUI & Measurement Tool
# ==========================================
class AdvancedDQETool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pro DQE Analyzer (Custom Line Phantom)")
        self.geometry("1100x850")

        # --- Data Variables ---
        self.img_arr = None
        self.img_8bit = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Line Profile
        self.p1 = None
        self.p2 = None
        
        # Measurement Results {freq: ctf_value}
        self.ctf_data = {} 
        # 사용자가 언급한 주파수 리스트
        self.target_freqs = [1.0, 2.0, 2.25, 2.5, 2.8, 3.15]
        
        # DQE Params
        self.dose_val = 100.0
        self.q0_val = 54000.0 # Soft beam default
        self.pitch_val = 0.076

        self.setup_ui()

    def setup_ui(self):
        # Top Control Panel
        top = tk.Frame(self, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
        top.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # 1. Image Load
        btn_frm = tk.Frame(top, bg="#f0f0f0")
        btn_frm.pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frm, text="1. 이미지 로드 (CTF)", command=self.load_image, bg="#e1f5fe", font=("Arial", 10, "bold")).pack(pady=2)

        # 2. Measurement Buttons (Dynamic)
        meas_frm = tk.LabelFrame(top, text="2. CTF 측정 (순서대로)", bg="#f0f0f0", padx=5, pady=5)
        meas_frm.pack(side=tk.LEFT, padx=10)
        
        self.meas_btns = {}
        for i, f in enumerate(self.target_freqs):
            b = tk.Button(meas_frm, text=f"{f} lp/mm", width=8,
                          command=lambda freq=f: self.measure_step(freq))
            b.grid(row=0, column=i, padx=2)
            self.meas_btns[f] = b
            
        tk.Button(meas_frm, text="초기화", command=self.reset_meas, bg="#ffcdd2").grid(row=0, column=len(self.target_freqs), padx=5)

        # 3. Analysis
        calc_frm = tk.Frame(top, bg="#f0f0f0")
        calc_frm.pack(side=tk.RIGHT, padx=10)
        tk.Button(calc_frm, text="3. DQE 분석\n(NNPS 로드)", command=self.run_analysis, bg="#c8e6c9", height=2, font=("Arial", 10, "bold")).pack()

        # Canvas
        self.canvas = tk.Canvas(self, bg="#202020", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

    # --- Image Handling ---
    def load_image(self):
        path = filedialog.askopenfilename(title="CTF 이미지 선택", filetypes=[("Raw/Img", "*.img *.raw *.bin"), ("All", "*.*")])
        if not path: return
        
        # Config Dialog
        win = tk.Toplevel(self)
        win.title("설정")
        tk.Label(win, text="Width:").grid(row=0, column=0)
        e_w = tk.Entry(win); e_w.insert(0, "3840"); e_w.grid(row=0, column=1)
        tk.Label(win, text="Height:").grid(row=1, column=0)
        e_h = tk.Entry(win); e_h.insert(0, "3072"); e_h.grid(row=1, column=1)
        
        def apply():
            try:
                w, h = int(e_w.get()), int(e_h.get())
                with open(path, "rb") as f:
                    data = np.fromfile(f, dtype=np.uint16, count=w*h)
                self.img_arr = data.reshape((h, w))
                
                # Auto Leveling for Display
                p1, p99 = np.percentile(self.img_arr, 1), np.percentile(self.img_arr, 99)
                norm = np.clip((self.img_arr - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                self.img_8bit = Image.fromarray(norm)
                
                self.redraw()
                messagebox.showinfo("안내", "이미지 로드 완료!\n\n[사용법]\n1. 마우스로 드래그하여 노란 선(Line Profile)을 그립니다.\n2. 상단의 주파수 버튼을 눌러 구간을 선택합니다.")
                win.destroy()
            except Exception as e:
                messagebox.showerror("에러", str(e))
                
        tk.Button(win, text="확인", command=apply).grid(row=2, columnspan=2, pady=5)

    def redraw(self):
        if self.img_8bit is None: return
        w, h = self.img_8bit.size
        nw, nh = int(w*self.zoom), int(h*self.zoom)
        tmp = self.img_8bit.resize((nw, nh), Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(tmp)
        self.canvas.delete("all")
        # Center Image
        cx, cy = self.canvas.winfo_width()//2 + self.offset_x, self.canvas.winfo_height()//2 + self.offset_y
        self.canvas.create_image(cx, cy, image=self.tk_img, anchor=tk.CENTER)
        
        # Draw Line
        if self.p1 and self.p2:
            x1, y1 = self.img2canv(*self.p1)
            x2, y2 = self.img2canv(*self.p2)
            self.canvas.create_line(x1, y1, x2, y2, fill="yellow", width=2)
            self.canvas.create_oval(x1-3, y1-3, x1+3, y1+3, fill="red")
            self.canvas.create_oval(x2-3, y2-3, x2+3, y2+3, fill="blue")

    # --- Mouse Events ---
    def img2canv(self, ix, iy):
        w, h = self.img_8bit.size
        nw, nh = int(w*self.zoom), int(h*self.zoom)
        cx = self.canvas.winfo_width()//2 + self.offset_x - nw//2 + ix*self.zoom
        cy = self.canvas.winfo_height()//2 + self.offset_y - nh//2 + iy*self.zoom
        return cx, cy

    def canv2img(self, cx, cy):
        w, h = self.img_8bit.size
        nw, nh = int(w*self.zoom), int(h*self.zoom)
        ix = (cx - (self.canvas.winfo_width()//2 + self.offset_x - nw//2)) / self.zoom
        iy = (cy - (self.canvas.winfo_height()//2 + self.offset_y - nh//2)) / self.zoom
        return int(ix), int(iy)

    def on_press(self, e):
        self.drag_start = (e.x, e.y)
        self.start_off = (self.offset_x, self.offset_y)

    def on_drag(self, e):
        if not self.img_8bit: return
        dx, dy = e.x - self.drag_start[0], e.y - self.drag_start[1]
        self.offset_x, self.offset_y = self.start_off[0]+dx, self.start_off[1]+dy
        self.redraw()

    def on_release(self, e):
        if not self.img_8bit: return
        dist = ((e.x-self.drag_start[0])**2 + (e.y-self.drag_start[1])**2)**0.5
        if dist < 5: # Click
            ix, iy = self.canv2img(e.x, e.y)
            if 0 <= ix < self.img_8bit.width and 0 <= iy < self.img_8bit.height:
                if not self.p1: self.p1 = (ix, iy)
                elif not self.p2: self.p2 = (ix, iy)
                else: self.p1 = (ix, iy); self.p2 = None
                self.redraw()

    def on_wheel(self, e):
        s = 1.1 if e.delta > 0 else 0.9
        self.zoom *= s
        self.redraw()

    # --- Measurement Logic ---
    def measure_step(self, freq):
        if not self.p1 or not self.p2:
            messagebox.showwarning("경고", "먼저 이미지 위에 Line을 그려주세요 (클릭 2번)")
            return
            
        # Extract Profile
        prof = self.get_profile()
        
        # Pop-up Plot for Selection
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(prof, 'k-', alpha=0.6)
        ax.set_title(f"[{freq} lp/mm] 구간을 드래그하여 선택하세요 (Max ~ Min 포함)")
        ax.grid(True)
        
        sel_data = {}
        def on_select(xmin, xmax):
            idx_min, idx_max = int(xmin), int(xmax)
            if idx_max - idx_min < 2: return
            region = prof[idx_min:idx_max]
            v_max = np.max(region)
            v_min = np.min(region)
            ctf = (v_max - v_min) / (v_max + v_min) * 100
            sel_data['res'] = ctf
            ax.set_title(f"[{freq} lp/mm] CTF: {ctf:.2f}% (Max:{int(v_max)}, Min:{int(v_min)})")
            plt.draw()

        span = SpanSelector(ax, on_select, 'horizontal', useblit=True, props=dict(alpha=0.3, facecolor='green'))
        plt.show()
        
        if 'res' in sel_data:
            val = sel_data['res']
            self.ctf_data[freq] = val
            self.meas_btns[freq].config(bg="#a5d6a7", text=f"{freq}\n{val:.1f}%")
            print(f"Recorded {freq} lp/mm: {val:.2f}%")

    def get_profile(self):
        # Bresenham
        x1, y1 = self.p1
        x2, y2 = self.p2
        points = []
        dx, dy = abs(x2-x1), abs(y2-y1)
        sx, sy = (1 if x1<x2 else -1), (1 if y1<y2 else -1)
        err = dx-dy
        while True:
            points.append(self.img_arr[y1, x1])
            if x1==x2 and y1==y2: break
            e2 = 2*err
            if e2 > -dy: err -= dy; x1 += sx
            if e2 < dx: err += dx; y1 += sy
        return np.array(points)

    def reset_meas(self):
        self.ctf_data = {}
        for f, btn in self.meas_btns.items():
            btn.config(bg="#f0f0f0", text=f"{f} lp/mm")

    # --- Final Analysis ---
    def run_analysis(self):
        if len(self.ctf_data) < 3:
            messagebox.showwarning("경고", "최소 3개 이상의 주파수를 측정해야 합니다.")
            return

        nnps_path = filedialog.askopenfilename(title="NNPS CSV 파일 선택", filetypes=[("CSV", "*.csv")])
        if not nnps_path: return
        
        # Params Input
        win = tk.Toplevel(self)
        win.title("파라미터 입력")
        
        tk.Label(win, text="Dose (uGy):").grid(row=0, column=0)
        e_dose = tk.Entry(win); e_dose.insert(0, str(self.dose_val)); e_dose.grid(row=0, column=1)
        
        tk.Label(win, text="q0 (photons):").grid(row=1, column=0)
        e_q0 = tk.Entry(win); e_q0.insert(0, str(self.q0_val)); e_q0.grid(row=1, column=1)
        
        tk.Label(win, text="Pitch (mm):").grid(row=2, column=0)
        e_p = tk.Entry(win); e_p.insert(0, str(self.pitch_val)); e_p.grid(row=2, column=1)
        
        def process():
            try:
                dose = float(e_dose.get())
                q0 = float(e_q0.get())
                pitch = float(e_p.get())
                
                # 1. Load NNPS
                df = pd.read_csv(nnps_path, header=1) # Try 1 first
                if 'Freq(lp/mm)' not in df.columns:
                    df = pd.read_csv(nnps_path, header=0)
                
                freq = df['Freq(lp/mm)'].values
                nnps = df['NNPS(mm^2)'].values
                
                # 2. CTF to MTF
                meas_f = sorted(self.ctf_data.keys())
                meas_v = [self.ctf_data[k] for k in meas_f]
                
                mtf, fit_f, fit_v = calculate_coltman_mtf(freq, meas_f, meas_v, pitch)
                
                # 3. DQE
                dqe = np.zeros_like(mtf)
                mask = (nnps > 0)
                dqe[mask] = (mtf[mask]**2) / (nnps[mask] * q0 * dose)
                
                # 4. Plot
                fig, ax = plt.subplots(2, 1, figsize=(8, 10))
                
                # MTF Plot
                ax[0].plot(freq, mtf, 'r-', linewidth=2, label='MTF (Coltman)')
                ax[0].plot(fit_f, fit_v, 'bo--', alpha=0.5, label='Interpolated CTF')
                ax[0].plot(meas_f, [v/100 for v in meas_v], 'ks', label='Measured Points')
                ax[0].set_title("MTF Result")
                ax[0].set_xlim(0, max(meas_f)+1)
                ax[0].grid(True)
                ax[0].legend()
                
                # DQE Plot
                ax[1].plot(freq, dqe, 'g-', linewidth=2, label=f'DQE (q0={int(q0)})')
                ax[1].set_title("DQE Result")
                ax[1].set_xlim(0, max(meas_f)+1)
                ax[1].set_ylim(0, 1.0)
                ax[1].grid(True)
                
                # Markers
                for f_mk in [0.5, 1.0, 2.0, 3.0]:
                    idx = np.abs(freq - f_mk).argmin()
                    val = dqe[idx]
                    ax[1].plot(freq[idx], val, 'ro')
                    ax[1].text(freq[idx], val+0.02, f"{val*100:.1f}%", color='red', fontweight='bold')

                plt.tight_layout()
                plt.show()
                win.destroy()
                
            except Exception as e:
                messagebox.showerror("계산 오류", str(e))
                
        tk.Button(win, text="계산 시작", command=process, bg="#4caf50", fg="white").grid(row=3, columnspan=2, pady=10, sticky="ew")

if __name__ == "__main__":
    app = AdvancedDQETool()
    app.mainloop()