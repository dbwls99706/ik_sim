import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math

class InverseKinematicsSimulator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("2-Link Robot Arm Inverse Kinematics Simulator")
        self.root.geometry("1000x700")
        
        # 로봇 팔 파라미터
        self.L1 = 3.0  # 첫 번째 링크 길이
        self.L2 = 2.0  # 두 번째 링크 길이
        
        # 현재 관절 각도 (라디안)
        self.theta1 = 0.0
        self.theta2 = 0.0
        
        # 목표 위치
        self.target_x = 4.0
        self.target_y = 1.0
        
        self.setup_gui()
        self.update_plot()
        
    def setup_gui(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 왼쪽 프레임 (컨트롤)
        control_frame = ttk.LabelFrame(main_frame, text="제어 패널", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 로봇 파라미터 설정
        params_frame = ttk.LabelFrame(control_frame, text="로봇 파라미터", padding="5")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(params_frame, text="Link 1 길이:").pack(anchor=tk.W)
        self.l1_var = tk.DoubleVar(value=self.L1)
        l1_scale = ttk.Scale(params_frame, from_=1.0, to=5.0, variable=self.l1_var, 
                            orient=tk.HORIZONTAL, command=self.update_parameters)
        l1_scale.pack(fill=tk.X)
        self.l1_label = ttk.Label(params_frame, text=f"L1 = {self.L1:.1f}")
        self.l1_label.pack(anchor=tk.W)
        
        ttk.Label(params_frame, text="Link 2 길이:").pack(anchor=tk.W, pady=(10, 0))
        self.l2_var = tk.DoubleVar(value=self.L2)
        l2_scale = ttk.Scale(params_frame, from_=1.0, to=5.0, variable=self.l2_var, 
                            orient=tk.HORIZONTAL, command=self.update_parameters)
        l2_scale.pack(fill=tk.X)
        self.l2_label = ttk.Label(params_frame, text=f"L2 = {self.L2:.1f}")
        self.l2_label.pack(anchor=tk.W)
        
        # 목표 위치 입력
        target_frame = ttk.LabelFrame(control_frame, text="목표 위치", padding="5")
        target_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(target_frame, text="X 좌표:").pack(anchor=tk.W)
        self.x_var = tk.DoubleVar(value=self.target_x)
        x_entry = ttk.Entry(target_frame, textvariable=self.x_var, width=15)
        x_entry.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(target_frame, text="Y 좌표:").pack(anchor=tk.W)
        self.y_var = tk.DoubleVar(value=self.target_y)
        y_entry = ttk.Entry(target_frame, textvariable=self.y_var, width=15)
        y_entry.pack(fill=tk.X, pady=(0, 10))
        
        # 제어 버튼
        ttk.Button(target_frame, text="역기구학 계산", 
                  command=self.solve_inverse_kinematics).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(target_frame, text="리셋", command=self.reset_robot).pack(fill=tk.X)
        
        # 현재 상태 표시
        status_frame = ttk.LabelFrame(control_frame, text="현재 상태", padding="5")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.theta1_label = ttk.Label(status_frame, text=f"θ1 = {math.degrees(self.theta1):.1f}°")
        self.theta1_label.pack(anchor=tk.W)
        
        self.theta2_label = ttk.Label(status_frame, text=f"θ2 = {math.degrees(self.theta2):.1f}°")
        self.theta2_label.pack(anchor=tk.W)
        
        self.end_effector_label = ttk.Label(status_frame, text="엔드 이펙터: (0.0, 0.0)")
        self.end_effector_label.pack(anchor=tk.W)
        
        # 작업 공간 정보
        workspace_frame = ttk.LabelFrame(control_frame, text="작업 공간 정보", padding="5")
        workspace_frame.pack(fill=tk.X)
        
        self.workspace_label = ttk.Label(workspace_frame, 
                                       text=f"최대 도달 거리: {self.L1 + self.L2:.1f}\n최소 도달 거리: {abs(self.L1 - self.L2):.1f}")
        self.workspace_label.pack(anchor=tk.W)
        
        # 오른쪽 프레임 (플롯)
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # matplotlib 플롯 설정
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_parameters(self, event=None):
        self.L1 = self.l1_var.get()
        self.L2 = self.l2_var.get()
        self.l1_label.config(text=f"L1 = {self.L1:.1f}")
        self.l2_label.config(text=f"L2 = {self.L2:.1f}")
        self.workspace_label.config(
            text=f"최대 도달 거리: {self.L1 + self.L2:.1f}\n최소 도달 거리: {abs(self.L1 - self.L2):.1f}"
        )
        self.update_plot()
        
    def check_singularity(self, x, y):
        distance = math.sqrt(x**2 + y**2)
        if distance > (self.L1 + self.L2):
            return "OUT_OF_REACH", f"목표점이 작업 공간 밖에 있습니다.\n최대 도달 거리: {self.L1 + self.L2:.2f}\n현재 거리: {distance:.2f}"
        if distance < abs(self.L1 - self.L2):
            return "TOO_CLOSE", f"목표점이 너무 가깝습니다.\n최소 도달 거리: {abs(self.L1 - self.L2):.2f}\n현재 거리: {distance:.2f}"
        if abs(distance - (self.L1 + self.L2)) < 0.01:
            return "ELBOW_EXTENDED", "팔이 완전히 펼쳐진 상태입니다. 특이점 근처에서 제어가 불안정할 수 있습니다."
        if abs(distance - abs(self.L1 - self.L2)) < 0.01:
            return "ELBOW_FOLDED", "팔이 완전히 접힌 상태입니다. 특이점 근처에서 제어가 불안정할 수 있습니다."
        return "OK", ""
        
    def solve_inverse_kinematics(self):
        try:
            x = self.x_var.get()
            y = self.y_var.get()
            singularity_status, warning_msg = self.check_singularity(x, y)
            if singularity_status in ["OUT_OF_REACH", "TOO_CLOSE"]:
                messagebox.showerror("오류", warning_msg)
                return
            elif singularity_status in ["ELBOW_EXTENDED", "ELBOW_FOLDED"]:
                result = messagebox.askwarning("경고", 
                                             warning_msg + "\n\n계속 진행하시겠습니까?")
                if not result:
                    return
            
            distance = math.sqrt(x**2 + y**2)
            cos_theta2 = (x**2 + y**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
            cos_theta2 = max(-1, min(1, cos_theta2))
            theta2 = math.acos(cos_theta2)
            k1 = self.L1 + self.L2 * math.cos(theta2)
            k2 = self.L2 * math.sin(theta2)
            theta1 = math.atan2(y, x) - math.atan2(k2, k1)
            self.theta1 = theta1
            self.theta2 = theta2
            self.target_x = x
            self.target_y = y
            self.update_status()
            self.update_plot()
            messagebox.showinfo("성공", f"역기구학 해를 찾았습니다!\nθ1 = {math.degrees(theta1):.1f}°\nθ2 = {math.degrees(theta2):.1f}°")
            
        except ValueError as e:
            messagebox.showerror("입력 오류", "올바른 숫자를 입력해주세요.")
        except Exception as e:
            messagebox.showerror("계산 오류", f"역기구학 계산 중 오류가 발생했습니다:\n{str(e)}")
    
    def forward_kinematics(self):
        x1 = self.L1 * math.cos(self.theta1)
        y1 = self.L1 * math.sin(self.theta1)
        x2 = x1 + self.L2 * math.cos(self.theta1 + self.theta2)
        y2 = y1 + self.L2 * math.sin(self.theta1 + self.theta2)
        return (0, 0), (x1, y1), (x2, y2)
    
    def update_status(self):
        base, joint1, end_effector = self.forward_kinematics()
        self.theta1_label.config(text=f"θ1 = {math.degrees(self.theta1):.1f}°")
        self.theta2_label.config(text=f"θ2 = {math.degrees(self.theta2):.1f}°")
        self.end_effector_label.config(
            text=f"엔드 이펙터: ({end_effector[0]:.2f}, {end_effector[1]:.2f})"
        )
    
    def update_plot(self):
        self.ax.clear()
        base, joint1, end_effector = self.forward_kinematics()
        x_coords = [base[0], joint1[0], end_effector[0]]
        y_coords = [base[1], joint1[1], end_effector[1]]
        self.ax.plot(x_coords, y_coords, 'b-', linewidth=3, label='Robot Arm')
        self.ax.plot([base[0], joint1[0]], [base[1], joint1[1]], 'ro', markersize=8)
        self.ax.plot(joint1[0], joint1[1], 'go', markersize=10, label='Joint 1')
        self.ax.plot(end_effector[0], end_effector[1], 'bo', markersize=10, label='End Effector')
        self.ax.plot(self.target_x, self.target_y, 'r*', markersize=15, label='Target')
        theta = np.linspace(0, 2*np.pi, 100)
        max_reach_x = (self.L1 + self.L2) * np.cos(theta)
        max_reach_y = (self.L1 + self.L2) * np.sin(theta)
        self.ax.plot(max_reach_x, max_reach_y, 'g--', alpha=0.5, label='Max Reach')
        if abs(self.L1 - self.L2) > 0:
            min_reach_x = abs(self.L1 - self.L2) * np.cos(theta)
            min_reach_y = abs(self.L1 - self.L2) * np.sin(theta)
            self.ax.plot(min_reach_x, min_reach_y, 'r--', alpha=0.5, label='Min Reach')
        max_range = self.L1 + self.L2 + 1
        self.ax.set_xlim(-max_range, max_range)
        self.ax.set_ylim(-max_range, max_range)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_title('2-Link Robot Arm Inverse Kinematics')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.canvas.draw()
    
    def reset_robot(self):
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.target_x = self.L1 + self.L2 - 1
        self.target_y = 0.0
        self.x_var.set(self.target_x)
        self.y_var.set(self.target_y)
        self.update_status()
        self.update_plot()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    simulator = InverseKinematicsSimulator()
    simulator.run()
