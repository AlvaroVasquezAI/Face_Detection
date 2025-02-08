# src/gui/app.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import datetime
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.gui.Face_Detection_Tracker import FaceDetector
from src.feedback.collector import FeedbackCollector
from src.gui.drawing_canvas import DrawingCanvas

class FaceDetectionApp:
    def __init__(self):
        # Main window configuration
        self.root = ctk.CTk()
        self.root.title("Face Detection App - RLHF Mode")
        self.root.geometry("1400x900")
        
        # Theme configuration
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        self.threshold = 0.5
        self.current_prediction = None
        self.human_correction = None
        self.drawing_canvas = None
        self.original_image_size = None
        
        # Model variables
        self.MODEL_PATH = None
        self.model_name = None
        self.detector = None
        
        # Feedback collector
        self.feedback_collector = FeedbackCollector()
        self.feedback_mode = False
        
        # Create interface
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Left control frame
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Model selection button
        self.select_model_button = ctk.CTkButton(
            self.control_frame,
            text="Select Model",
            command=self.select_model
        )
        self.select_model_button.pack(pady=10, padx=10)
        
        # Model label
        self.model_label = ctk.CTkLabel(
            self.control_frame,
            text="Model: Not selected"
        )
        self.model_label.pack(pady=5)
        
        # Image selection button
        self.select_button = ctk.CTkButton(
            self.control_frame,
            text="Select Image",
            command=self.select_image,
            state="disabled"
        )
        self.select_button.pack(pady=10, padx=10)
        
        # Threshold control
        self.threshold_label = ctk.CTkLabel(
            self.control_frame,
            text="Detection Threshold"
        )
        self.threshold_label.pack(pady=(20,0))
        
        self.threshold_slider = ctk.CTkSlider(
            self.control_frame,
            from_=0,
            to=1,
            number_of_steps=100,
            command=self.update_threshold
        )
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack(pady=10, padx=10)
        
        # Results frame
        self.results_frame = ctk.CTkFrame(self.control_frame)
        self.results_frame.pack(fill="x", pady=20, padx=10)
        
        # Results labels
        self.face_detected_label = ctk.CTkLabel(
            self.results_frame,
            text="Face Detected: -"
        )
        self.face_detected_label.pack(pady=5)
        
        self.confidence_label = ctk.CTkLabel(
            self.results_frame,
            text="Confidence: -"
        )
        self.confidence_label.pack(pady=5)
        
        self.bbox_label = ctk.CTkLabel(
            self.results_frame,
            text="Bounding Box: -"
        )
        self.bbox_label.pack(pady=5)
        
        # Feedback frame
        self.create_feedback_widgets()
        
        # Image frame
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Image label/canvas container
        self.image_container = ctk.CTkFrame(self.image_frame)
        self.image_container.pack(fill="both", expand=True)
        
        # Image label
        self.image_label = ctk.CTkLabel(self.image_container, text="")
        self.image_label.pack(fill="both", expand=True)
        
    def create_feedback_widgets(self):
        # Feedback control frame
        self.feedback_frame = ctk.CTkFrame(self.control_frame)
        self.feedback_frame.pack(fill="x", pady=20, padx=10)
        
        # Feedback mode toggle
        self.feedback_mode_btn = ctk.CTkButton(
            self.feedback_frame,
            text="Enable Feedback Mode",
            command=self.toggle_feedback_mode
        )
        self.feedback_mode_btn.pack(pady=5)
        
        # Draw correction button
        self.draw_correction_btn = ctk.CTkButton(
            self.feedback_frame,
            text="Draw Correction",
            command=self.start_drawing_correction,
            state="disabled"
        )
        self.draw_correction_btn.pack(pady=5)
        
        # Rating frame
        self.rating_frame = ctk.CTkFrame(self.feedback_frame)
        self.rating_frame.pack(fill="x", pady=5)
        
        self.rating_label = ctk.CTkLabel(
            self.rating_frame,
            text="Rate Prediction (0-5):"
        )
        self.rating_label.pack(pady=2)
        
        self.rating_slider = ctk.CTkSlider(
            self.rating_frame,
            from_=0,
            to=5,
            number_of_steps=5
        )
        self.rating_slider.set(2.5)
        self.rating_slider.pack(pady=5)
        
        # Comments
        self.feedback_comment = ctk.CTkTextbox(
            self.feedback_frame,
            height=100
        )
        self.feedback_comment.pack(pady=5, fill="x")
        
        # Submit feedback button
        self.submit_feedback_btn = ctk.CTkButton(
            self.feedback_frame,
            text="Submit Feedback",
            command=self.submit_feedback,
            state="disabled"
        )
        self.submit_feedback_btn.pack(pady=5)
        
    def toggle_feedback_mode(self):
        self.feedback_mode = not self.feedback_mode
        if self.feedback_mode:
            self.feedback_mode_btn.configure(text="Disable Feedback Mode")
            if self.current_image is not None:
                self.draw_correction_btn.configure(state="normal")
        else:
            self.feedback_mode_btn.configure(text="Enable Feedback Mode")
            self.draw_correction_btn.configure(state="disabled")
            if self.drawing_canvas:
                self.drawing_canvas.destroy()
                self.drawing_canvas = None
                self.image_label.pack(fill="both", expand=True)

    def start_drawing_correction(self):
        if self.current_image is None or self.current_prediction is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
            
        self.image_label.pack_forget()
        
        if self.drawing_canvas is None:
            # Get the model's prediction bbox (already normalized)
            model_bbox = self.current_prediction['bbox'] if self.current_prediction['has_face'] else None
            
            print(f"Original image size: {self.original_image_size}")  # Debug print
            print(f"Current image size: {self.current_image.size}")    # Debug print
            print(f"Model bbox (normalized): {model_bbox}")           # Debug print
            
            self.drawing_canvas = DrawingCanvas(
                parent=self.image_container,
                image=self.current_image,
                model_bbox=model_bbox,  # Pass normalized coordinates
                image_size=self.original_image_size,
                callback=self.on_drawing_complete
            )
        else:
            self.drawing_canvas.reset(self.current_image, self.current_prediction)
        
        self.drawing_canvas.pack(fill="both", expand=True)

    def on_drawing_complete(self, pixel_bbox):
        """Handle completion of drawing"""
        try:
            print(f"Received pixel_bbox: {pixel_bbox}")  # Debug print
            
            # Convert pixel coordinates to normalized coordinates
            h, w = self.original_image_size
            print(f"Original image size: {w}x{h}")  # Debug print
            
            # Get current display size
            display_w, display_h = self.current_image.size
            print(f"Display size: {display_w}x{display_h}")  # Debug print
            
            # Convert display coordinates to original image coordinates first
            scale_w = w / display_w
            scale_h = h / display_h
            
            original_coords = [
                int(pixel_bbox[0] * scale_w),
                int(pixel_bbox[1] * scale_h),
                int(pixel_bbox[2] * scale_w),
                int(pixel_bbox[3] * scale_h)
            ]
            print(f"Original coordinates: {original_coords}")  # Debug print
            
            # Then normalize to 0-1 range
            self.human_correction = [
                original_coords[0] / w,
                original_coords[1] / h,
                original_coords[2] / w,
                original_coords[3] / h
            ]
            print(f"Normalized coordinates: {self.human_correction}")  # Debug print
            
            # Enable feedback submission
            self.submit_feedback_btn.configure(state="normal")
            
            # Update UI to guide user
            self.rating_label.configure(
                text="Please rate the model's prediction (0-5):"
            )
            
            # Show success message
            messagebox.showinfo(
                "Success", 
                "Correction recorded!\n\n"
                "Please:\n"
                "1. Rate the model's prediction (0-5)\n"
                "2. Add any comments (optional)\n"
                "3. Click 'Submit Feedback'"
            )
        except Exception as e:
            print(f"Error in on_drawing_complete: {e}")  # Debug print
            messagebox.showerror("Error", f"Error processing drawing: {e}")

    def submit_feedback(self):
        """Handle feedback submission"""
        if not self.human_correction:
            messagebox.showwarning("Warning", "Please draw a correction first")
            return
        
        # Collect all feedback data
        feedback_data = {
            'image_path': self.current_image_path,
            'model_name': self.model_name,
            'model_prediction': {
                'has_face': self.current_prediction['has_face'],
                'confidence': float(self.current_prediction['confidence']),
                'bbox': [float(x) for x in self.current_prediction['bbox']] if self.current_prediction['has_face'] else None
            },
            'human_correction': [float(x) for x in self.human_correction],
            'rating': float(self.rating_slider.get()),
            'comments': self.feedback_comment.get("1.0", "end-1c"),
            'timestamp': datetime.datetime.now().isoformat(),
            'image_size': self.original_image_size
        }
        
        # Save feedback
        self.feedback_collector.add_feedback(feedback_data)
        
        # Show success message
        messagebox.showinfo(
            "Success", 
            "Feedback submitted successfully!\n\n"
            f"Rating: {feedback_data['rating']}/5\n"
            f"Comments recorded: {'Yes' if feedback_data['comments'] else 'No'}"
        )
        
        # Reset UI
        self.reset_feedback_ui()

    def reset_feedback_ui(self):
        """Reset UI after feedback submission"""
        # Reset feedback components
        self.submit_feedback_btn.configure(state="disabled")
        self.human_correction = None
        self.feedback_comment.delete("1.0", "end")
        self.rating_slider.set(2.5)
        
        # Reset canvas
        if self.drawing_canvas:
            self.drawing_canvas.destroy()
            self.drawing_canvas = None
            self.image_label.pack(fill="both", expand=True)
        
        # Update labels
        self.rating_label.configure(text="Rate Prediction (0-5):")
        
        # Enable new correction
        self.draw_correction_btn.configure(state="normal")

    def select_model(self):
        model_dir = filedialog.askdirectory(
            title="Select Model Directory",
            initialdir="models/"
        )
        
        if model_dir:
            weights_path = os.path.join(model_dir, "best_weights.weights.h5")
            
            if os.path.exists(weights_path):
                try:
                    self.MODEL_PATH = weights_path
                    self.detector = FaceDetector(self.MODEL_PATH)
                    self.model_name = os.path.basename(model_dir)
                    self.model_label.configure(text=f"Model: {self.model_name}")
                    self.select_button.configure(state="normal")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Error loading model: {e}")
                    self.model_label.configure(text="Model: Error loading model")
            else:
                messagebox.showerror("Error", f"No weights file found in {model_dir}")
                self.model_label.configure(text="Model: No weights file found")

    def select_image(self):
        if self.detector is None:
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        # Open dialog to select image
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.process_image()

    def update_threshold(self, value):
        """
        Update detection threshold and reprocess current image if one exists
        
        Args:
            value: New threshold value from slider
        """
        self.threshold = float(value)
        if self.current_image_path:
            self.process_image()

    def process_image(self):
        if self.detector is None:
            messagebox.showwarning("Warning", "Please select a model first")
            return

        try:
            # Use FaceDetector's methods to process the image
            result = self.detector.detect_face(self.current_image_path, self.threshold)
            self.original_image_size = result['original_size']
            self.current_prediction = {
                'has_face': result['has_face'],
                'confidence': result['confidence'],
                'bbox': result['bbox'] if result['has_face'] else None
            }
            
            # Draw detection on image
            processed_image = self.detector.draw_detection(result['original_image'], result)
            
            # Convert for display
            image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            self.current_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Resize for display
            display_size = (800, 600)
            self.current_image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            # Update display
            photo = ctk.CTkImage(
                light_image=self.current_image,
                dark_image=self.current_image,
                size=(self.current_image.width, self.current_image.height)
            )
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            # Update results display
            self.update_results_display(result)
            
            # Enable feedback controls if in feedback mode
            if self.feedback_mode:
                self.draw_correction_btn.configure(state="normal")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {e}")

    def update_results_display(self, result):
        self.face_detected_label.configure(
            text=f"Face Detected: {result['has_face']}"
        )
        self.confidence_label.configure(
            text=f"Confidence: {result['confidence']:.2f}"
        )
        if result['has_face']:
            bbox = [f"{x:.2f}" for x in result['bbox']]
            self.bbox_label.configure(
                text=f"Bounding Box: [{', '.join(bbox)}]"
            )
        else:
            self.bbox_label.configure(text="Bounding Box: -")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceDetectionApp()
    app.run()