import tkinter as tk
from tkinter import font, ttk
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# === PHẦN XỬ LÝ AI & LOGIC ===
class AITrainer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Lưu dữ liệu bài tập
        self.session_data = {
            'angles': [],
            'rom_data': [],  
            'form_violations': 0, # Số lỗi kỹ thuật
        }
        
    def calculate_angle(self, a, b, c):
        """
        Tính góc giữa 3 điểm (Vai - Khuỷu - Cổ tay) hoặc (Hông - Gối - Cổ chân)
        """
        a = np.array(a) # Điểm đầu
        b = np.array(b) # Điểm giữa (khớp xoay)
        c = np.array(c) # Điểm cuối
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0: 
            angle = 360.0 - angle
        return angle
    
    def analyze_form_quality(self, angle, back_angle, mode):
        """
        Kiểm tra lỗi sai kỹ thuật (Logic đã được sửa)
        """
        feedback = ""
        is_correct = True # Mặc định là đúng
        
        # --- LOGIC SQUAT ---
        if mode == 'squat':
            # 1. Kiểm tra lưng (Quan trọng nhất)
            if back_angle > 35:
                feedback = "CANH BAO: Giu thang lung!"
                is_correct = False
            
            # 2. Kiểm tra độ sâu khi Squat
            elif angle < 70:
                feedback = "CANH BAO: Xuong qua sau"
                is_correct = False # Tùy quan điểm y khoa, thường <70 là rủi ro cao
            
            # 3. Trạng thái đứng hoặc đang xuống
            elif angle > 160:
                feedback = "READY"
            elif angle > 100:
                feedback = "Hay xuong sau hon!" # Khuyến khích
            else:
                feedback = "Squat tot!" # Khoảng 70-100 độ
                
        # --- LOGIC BICEP CURL ---
        elif mode == 'curl':
            # 1. Kiểm tra gập tay quá mức (dùng đà ép tay)
            if angle < 30:
                feedback = "SAI: Gap qua sat!"
                is_correct = False
            
            # 2. Kiểm tra duỗi tay
            elif angle > 160:
                feedback = "Duoi tay tot (READY)"
            elif angle > 120:
                feedback = "Dang cuon tay..."
            else:
                feedback = "Tu the tot!" # Khoảng 30-120 độ là vùng hoạt động
                
        return feedback, is_correct

    def run_exercise(self, mode):
        
        #Chạy vòng lặp camera và phân tích
        
        cap = cv2.VideoCapture(0)
        
        # Biến đếm Reps
        counter = 0
        stage = None # Trạng thái: 'up' (lên) hoặc 'down' (xuống)
        
        # Biến theo dõi ROM và Lỗi
        angle_history = []
        max_angle = 0
        min_angle = 180
        form_violations = 0
        
        # Cấu hình cửa sổ hiển thị
        cv2.namedWindow('AI Rehabilitation Assistant', cv2.WINDOW_NORMAL)

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Xử lý ảnh: Lật ngược -> Chuyển RGB -> MediaPipe
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                h, w, c = image.shape
                
                # Màu mặc định: Xanh lá (Đúng)
                draw_color = (0, 255, 0)
                angle = 0
                back_angle = 0
                feedback = "Hãy di chuyển vào khung hình"
                is_correct = True

                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Helper lấy tọa độ pixel
                    def get_xy(lm_type):
                        return [landmarks[lm_type.value].x * w, landmarks[lm_type.value].y * h]

                    # === XỬ LÝ BÀI TẬP CURL ===
                    if mode == 'curl':
                        # Lấy điểm: Vai - Khuỷu - Cổ tay (Trái)
                        shoulder = get_xy(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
                        elbow = get_xy(self.mp_pose.PoseLandmark.LEFT_ELBOW)
                        wrist = get_xy(self.mp_pose.PoseLandmark.LEFT_WRIST)
                        
                        angle = self.calculate_angle(shoulder, elbow, wrist)
                        
                        # Phân tích lỗi
                        feedback, is_correct = self.analyze_form_quality(angle, 0, mode)
                        
                        # Logic đếm Rep (Cánh tay)
                        if angle > 160: 
                            stage = "down" # Tay đang duỗi
                        if angle < 40 and stage == 'down': # Tay gập lên
                            stage = "up"
                            counter += 1
                        
                        # Thanh hiển thị mức độ gập (0% -> 100%)
                        bar_val = np.interp(angle, (30, 160), (100, 0))
                        
                    # Lấy tọa độ, xử lí bài squat
                    elif mode == 'squat':
                        # Lấy điểm: Vai - Hông - Gối - Cổ chân (Trái)
                        shoulder = get_xy(self.mp_pose.PoseLandmark.LEFT_SHOULDER)
                        hip = get_xy(self.mp_pose.PoseLandmark.LEFT_HIP)
                        knee = get_xy(self.mp_pose.PoseLandmark.LEFT_KNEE)
                        ankle = get_xy(self.mp_pose.PoseLandmark.LEFT_ANKLE)
                        
                        # Góc đầu gối
                        angle = self.calculate_angle(hip, knee, ankle)
                        
                        # Góc lưng (So với trục dọc)
                        # Tạo một điểm ảo thẳng đứng trên hông để đo độ nghiêng
                        hip_vertical = [hip[0], hip[1] - 100] 
                        back_angle = self.calculate_angle(hip_vertical, hip, shoulder)
                        
                        # Phân tích lỗi
                        feedback, is_correct = self.analyze_form_quality(angle, back_angle, mode)
                        
                        # Logic đếm Rep (Squat)
                        if angle > 160: 
                            stage = "up" # Đang đứng
                        if angle < 90 and stage == 'up': # Đã ngồi xuống đủ sâu
                            # Chỉ đếm nếu lưng thẳng
                            if back_angle <= 35:
                                stage = "down"
                                counter += 1
                        
                        # Thanh hiển thị (170 độ là đứng, 80 độ là ngồi sâu)
                        bar_val = np.interp(angle, (80, 170), (100, 0))

                    # === CẬP NHẬT DỮ LIỆU ===
                    angle_history.append(angle)
                    max_angle = max(max_angle, angle)
                    min_angle = min(min_angle, angle)
                    
                    if not is_correct:
                        draw_color = (0, 0, 255) # Đỏ (Sai)
                        form_violations += 1

                    # === VẼ GIAO DIỆN LÊN CAMERA ===
                    # 1. Header nền tối
                    cv2.rectangle(image, (0,0), (w, 130), (44, 62, 80), -1)
                    
                    # 2. Tên bài tập
                    ex_title = "BICEP CURL" if mode == 'curl' else "SQUAT"
                    cv2.putText(image, ex_title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # 3. Số Reps
                    cv2.putText(image, f"Reps: {counter}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (46, 204, 113), 2)
                    
                    # 4. Góc khớp & ROM
                    cv2.putText(image, f"Goc: {int(angle)} do", (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    curr_rom = max_angle - min_angle
                    cv2.putText(image, f"ROM (Bien do): {int(curr_rom)}", (250, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    
                    if mode == 'squat':
                         cv2.putText(image, f"Lung: {int(back_angle)} do", (250, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    # 5. Đếm lỗi
                    cv2.putText(image, f"Loi: {form_violations}", (w-150, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # 6. Hộp Phản hồi (Feedback Box) ở dưới cùng
                    fb_color = (46, 204, 113) if is_correct else (0, 0, 255) # Xanh lá hoặc Đỏ
                    cv2.rectangle(image, (0, h-60), (w, h), fb_color, -1)
                    cv2.putText(image, feedback, (50, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # 7. Thanh Progress Bar (bên phải)
                    bar_x = w - 50
                    bar_h = int(np.interp(bar_val, (0, 100), (0, 300)))
                    cv2.rectangle(image, (bar_x, h-400), (bar_x+30, h-100), (70, 70, 70), -1) # Nền thanh
                    cv2.rectangle(image, (bar_x, h-100-bar_h), (bar_x+30, h-100), draw_color, -1) # Giá trị

                    # Vẽ khung xương
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                except Exception as e:
                    pass

                cv2.imshow('AI Rehabilitation Assistant', image)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'): # Thoát
                    break
                if key == ord('r'): # Reset
                    counter = 0
                    form_violations = 0
                    max_angle = 0
                    min_angle = 180

        cap.release()
        cv2.destroyAllWindows()
        
        # Trả về kết quả
        return {
            'reps': counter,
            'violations': form_violations,
            'rom': curr_rom if 'curr_rom' in locals() else 0
        }

# === PHẦN GIAO DIỆN NGƯỜI DÙNG (GUI) ===
class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Hỗ Trợ Phục Hồi Chức Năng AI")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")
        
        self.ai_trainer = AITrainer()
        
        # Font chữ tiếng Việt
        self.header_font = font.Font(family="Arial", size=28, weight="bold")
        self.sub_font = font.Font(family="Arial", size=12)
        self.btn_font = font.Font(family="Arial", size=14, weight="bold")

        self.main_frame = tk.Frame(root, bg="#2c3e50")
        self.main_frame.pack(fill="both", expand=True)
        
        self.show_home_screen()

    def show_home_screen(self):
        # Xóa màn hình cũ
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        # Logo
        tk.Label(self.main_frame, text="🏥", font=("Arial", 60), bg="#2c3e50").pack(pady=30)
        
        # Tiêu đề
        tk.Label(self.main_frame, text="TRỢ LÝ PHỤC HỒI AI", 
                 font=self.header_font, bg="#2c3e50", fg="#ecf0f1").pack()
        
        tk.Label(self.main_frame, text="Phân tích chuyển động & Giám sát tập luyện", 
                 font=self.sub_font, bg="#2c3e50", fg="#bdc3c7").pack(pady=5)
        
        # Nút Bắt đầu
        btn_start = tk.Button(self.main_frame, text="BẮT ĐẦU TẬP", 
                              font=self.btn_font, bg="#27ae60", fg="white",
                              width=20, height=2, cursor="hand2",
                              command=self.show_exercise_selection)
        btn_start.pack(pady=40)
        
        # Hướng dẫn
        tk.Label(self.main_frame, text="Nhấn 'Q' để thoát camera | Nhấn 'R' để đặt lại", 
                 font=("Arial", 10, "italic"), bg="#2c3e50", fg="#95a5a6").pack(side="bottom", pady=20)

    def show_exercise_selection(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        tk.Label(self.main_frame, text="CHỌN BÀI TẬP", 
                 font=self.header_font, bg="#2c3e50", fg="#ecf0f1").pack(pady=40)
        
        # Container cho các nút
        btn_container = tk.Frame(self.main_frame, bg="#2c3e50")
        btn_container.pack(pady=20)
        
        # Card SQUAT 
        f_squat = tk.Frame(btn_container, bg="#34495e", bd=2, relief="groove")
        f_squat.grid(row=0, column=0, padx=20, ipadx=20, ipady=20)
        
        tk.Label(f_squat, text="🦵", font=("Arial", 40), bg="#34495e").pack()
        tk.Label(f_squat, text="SQUAT (Gánh đùi)", font=("Arial", 14, "bold"), bg="#34495e", fg="white").pack(pady=10)
        tk.Button(f_squat, text="Chọn", bg="#e67e22", fg="white", font=("Arial", 12), width=10,
                  command=lambda: self.start_session('squat')).pack()

        # Card CURL 
        f_curl = tk.Frame(btn_container, bg="#34495e", bd=2, relief="groove")
        f_curl.grid(row=0, column=1, padx=20, ipadx=20, ipady=20)
        
        tk.Label(f_curl, text="💪", font=("Arial", 40), bg="#34495e").pack()
        tk.Label(f_curl, text="BICEP CURL (Tay)", font=("Arial", 14, "bold"), bg="#34495e", fg="white").pack(pady=10)
        tk.Button(f_curl, text="Chọn", bg="#3498db", fg="white", font=("Arial", 12), width=10,
                  command=lambda: self.start_session('curl')).pack()
        
        # Nút Quay lại
        tk.Button(self.main_frame, text="← Quay lại", bg="#7f8c8d", fg="white", font=("Arial", 10),
                  command=self.show_home_screen).pack(pady=40)

    def start_session(self, mode):
        # Ẩn GUI
        self.root.withdraw()
        
        # Chạy AI
        data = self.ai_trainer.run_exercise(mode)
        
        # Hiện lại GUI và hiện kết quả (có thể mở rộng thêm màn hình Report)
        self.root.deiconify()
        print(f"Kết quả buổi tập: {data}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()