# src/gui/drawing_canvas.py
import customtkinter as ctk
from PIL import ImageTk

# src/gui/drawing_canvas.py
class DrawingCanvas(ctk.CTkFrame):
    def __init__(self, parent, image, model_bbox, image_size, callback=None):
        super().__init__(parent)
        
        self.callback = callback
        self.image = image
        self.model_bbox = model_bbox
        self.image_size = image_size
        self.current_rect = None
        self.start_x = None
        self.start_y = None
        self.rect_coords = None  # Add this to store final coordinates
        
        # Create canvas with image dimensions
        self.canvas = ctk.CTkCanvas(
            self,
            width=image.width,
            height=image.height,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Display image
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(
            0, 0,
            image=self.photo,
            anchor="nw"
        )
        
        # Draw model prediction if available
        if model_bbox is not None:
            self.draw_model_prediction()
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Create buttons
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.pack(pady=5)
        
        self.clear_button = ctk.CTkButton(
            self.buttons_frame,
            text="Clear",
            command=self.clear_drawing,
            width=100
        )
        self.clear_button.pack(side="left", padx=5)
        
        self.done_button = ctk.CTkButton(
            self.buttons_frame,
            text="Done",
            command=self.finish_drawing,
            width=100
        )
        self.done_button.pack(side="left", padx=5)

    def draw_model_prediction(self):
        """Draw model's prediction in blue"""
        if self.model_bbox:
            # Get current display dimensions
            display_w = self.image.width
            display_h = self.image.height
            
            # Get original image dimensions
            orig_h, orig_w = self.image_size
            
            # Calculate scaling factors
            scale_x = display_w / orig_w
            scale_y = display_h / orig_h
            
            # Scale the model bbox coordinates to display size
            x1 = self.model_bbox[0] * scale_x
            y1 = self.model_bbox[1] * scale_y
            x2 = self.model_bbox[2] * scale_x
            y2 = self.model_bbox[3] * scale_y
            
            print(f"Model bbox (original): {self.model_bbox}")  # Debug print
            print(f"Model bbox (scaled): [{x1}, {y1}, {x2}, {y2}]")  # Debug print
            
            # Draw the rectangle
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline='blue',
                width=2,
                dash=(5, 5),
                tags='model_pred'
            )

    def on_press(self, event):
        """Handle mouse press"""
        print(f"Mouse press at: ({event.x}, {event.y})")  # Debug print
        self.start_x = event.x
        self.start_y = event.y
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            self.rect_coords = None

    def on_drag(self, event):
        """Handle mouse drag"""
        if not self.start_x or not self.start_y:
            return
            
        print(f"Mouse drag to: ({event.x}, {event.y})")  # Debug print
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            event.x, event.y,
            outline='green',
            width=2
        )

    def on_release(self, event):
        """Handle mouse release"""
        if not self.start_x or not self.start_y:
            return
            
        print(f"Mouse release at: ({event.x}, {event.y})")  # Debug print
        
        # Ensure coordinates are in correct order
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Store the coordinates
        self.rect_coords = [x1, y1, x2, y2]
        print(f"Stored coordinates: {self.rect_coords}")  # Debug print
        
        # Update rectangle
        if self.current_rect:
            self.canvas.coords(self.current_rect, x1, y1, x2, y2)

    def finish_drawing(self):
        """Complete drawing and return coordinates"""
        if not self.current_rect or not self.rect_coords:
            print("No rectangle drawn")
            return
            
        try:
            print(f"Final coordinates: {self.rect_coords}")  # Debug print
            
            # Change rectangle color
            self.canvas.itemconfig(self.current_rect, outline='yellow', width=3)
            
            # Disable drawing
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.done_button.configure(state="disabled")
            
            # Call callback with coordinates
            if self.callback:
                self.callback(self.rect_coords)
                
        except Exception as e:
            print(f"Error in finish_drawing: {e}")
            import traceback
            traceback.print_exc()

    def clear_drawing(self):
        """Clear current drawing"""
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            self.rect_coords = None
        self.start_x = None
        self.start_y = None