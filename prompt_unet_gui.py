import os
import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from models.prompt_unet import PromptUNet 

# Global parameters.
DIM = 256
NUM_CLASSES = 3  # 0: background, 1: cat, 2: dog
WEIGHTS_PATH = './prompt_weights/prompt_unet_model_256_epochs_50.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the model.
model = PromptUNet(num_classes=NUM_CLASSES, in_channels=4).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

def preprocess_image_and_prompt(image, prompt_coords, dim, device, prompt_kernel=15):
    """
    Preprocesses an image and creates a prompt heat map.
    Returns input tensor (1,4,dim,dim) and the prompt heatmap (dim,dim) for visualization.
    """
    # Convert image to RGB and resize.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    resized_img = cv2.resize(image_rgb, (dim, dim))
    resized_img = resized_img.astype(np.float32) / 255.0
    # Convert to CHW.
    image_tensor = np.transpose(resized_img, (2, 0, 1))
    
    # Map prompt coordinate from original image to resized.
    scale_x = dim / orig_w
    scale_y = dim / orig_h
    prompt_x = int(prompt_coords[0] * scale_x)
    prompt_y = int(prompt_coords[1] * scale_y)
    prompt_x = np.clip(prompt_x, 0, dim-1)
    prompt_y = np.clip(prompt_y, 0, dim-1)
    
    # Create a blank heat map, set the prompt point.
    prompt_heatmap = np.zeros((dim, dim), dtype=np.float32)
    prompt_heatmap[prompt_y, prompt_x] = 1.0
    # Apply Gaussian blur.
    prompt_heatmap = cv2.GaussianBlur(prompt_heatmap, (prompt_kernel, prompt_kernel), 0)
    prompt_heatmap = prompt_heatmap / (prompt_heatmap.max() + 1e-8)
    
    # Stack image and prompt.
    input_array = np.concatenate([image_tensor, np.expand_dims(prompt_heatmap, axis=0)], axis=0)
    input_tensor = torch.from_numpy(input_array).float().unsqueeze(0).to(device)
    
    return input_tensor, prompt_heatmap

def postprocess_output(output, orig_dims):
    output = output.astype(np.uint8)
    output = cv2.resize(output, (orig_dims[1], orig_dims[0]), interpolation=cv2.INTER_NEAREST)
    return output

class PromptSegGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Prompt-based Segmentation")
        self.image = None         # original image (cv2 BGR)
        self.display_image = None # image to display in Tkinter (PIL Image)
        self.prompt_point = None  # (x, y) in original image coordinates

        # Create UI components.
        self.canvas = tk.Canvas(root, width=600, height=400, bg='gray')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5)
        segment_btn = tk.Button(btn_frame, text="Segment", command=self.run_segmentation)
        segment_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(root, text="Load an image to begin.", fg="blue")
        self.status_label.pack(pady=5)

    def load_image(self):
        # Let user select an image file.
        file_path = filedialog.askopenfilename(title="Select Image",
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("Error", "Could not load image.")
                return
            # Convert for display (RGB, then PIL).
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image = Image.fromarray(image_rgb)
            # Resize image for display (maintain aspect ratio).
            self.display_image = self.display_image.resize((600, 400))
            self.tk_image = ImageTk.PhotoImage(self.display_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.status_label.config(text="Click on the image to set a prompt point.")
            self.prompt_point = None

    def on_canvas_click(self, event):
        # When user clicks on the canvas, record the point.
        if self.display_image is None:
            return
        # Get the click coordinates in the displayed image.
        x, y = event.x, event.y
        self.prompt_point = (x, y)
        # Draw a small circle to indicate the prompt.
        self.canvas.delete("prompt")
        r = 5
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="yellow", outline="black", tags="prompt")
        self.status_label.config(text=f"Prompt point set at: ({x}, {y})")

    def run_segmentation(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return
        if self.prompt_point is None:
            messagebox.showwarning("Warning", "Please click on the image to set a prompt point!")
            return

        self.status_label.config(text="Running segmentation...", fg="red")
        self.root.update_idletasks()
        # The prompt_point is in display image coordinates; we need to map back to original.
        disp_w, disp_h = self.display_image.size
        orig_h, orig_w = self.image.shape[:2]
        scale_x = orig_w / disp_w
        scale_y = orig_h / disp_h
        orig_prompt = (int(self.prompt_point[0] * scale_x), int(self.prompt_point[1] * scale_y))

        # Preprocess image and prompt.
        input_tensor, prompt_heat = preprocess_image_and_prompt(self.image, orig_prompt, DIM, DEVICE)
        with torch.no_grad():
            output_logits = model(input_tensor)
            pred = torch.argmax(output_logits, dim=1).cpu().numpy().squeeze()
        seg_result = postprocess_output(pred, (orig_h, orig_w))

        # Create a color overlay for segmentation.
        # Define colors: background = black, cat = red, dog = blue.
        overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        overlay[seg_result == 0] = [0, 0, 0]
        overlay[seg_result == 1] = [255, 0, 0]
        overlay[seg_result == 2] = [0, 0, 255]
        # Blend with original image.
        blended = cv2.addWeighted(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 0.6, overlay, 0.4, 0)
        blended_pil = Image.fromarray(blended)
        blended_pil = blended_pil.resize((600, 400))
        self.tk_blended = ImageTk.PhotoImage(blended_pil)

        # Show the segmentation result in a new window.
        result_window = tk.Toplevel(self.root)
        result_window.title("Segmentation Result")
        canvas_result = tk.Canvas(result_window, width=600, height=400)
        canvas_result.pack()
        canvas_result.create_image(0, 0, anchor=tk.NW, image=self.tk_blended)
        self.status_label.config(text="Segmentation complete.", fg="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = PromptSegGUI(root)
    root.mainloop()
