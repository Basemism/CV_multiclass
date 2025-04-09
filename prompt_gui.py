import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from models.prompt_unet import PromptUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PromptUNet(num_classes=2, in_channels=4).to(device)
model_path = 'prompt_weights/prompt_unet_model_256_epochs_50.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------------------
# Helper Functions
# ---------------------------
def create_prompt_heatmap(img_shape, point, sigma=5):
    """Create a heatmap with a Gaussian centered at 'point'."""
    heatmap = np.zeros(img_shape, dtype=np.float32)
    heatmap[point[0], point[1]] = 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap

def preprocess_for_model(image, prompt_heat):
    """
    Given an image (H,W,3) in BGR and a prompt heatmap (H,W) in [0,1],
    convert image to RGB, normalize to [0,1], resize to (256,256),
    and concatenate the prompt as a fourth channel.
    """
    image = cv2.resize(image, (256,256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2,0,1))  # (3,256,256)
    prompt_heat = cv2.resize(prompt_heat, (256,256))
    prompt_heat = np.expand_dims(prompt_heat, axis=0)  # (1,256,256)
    input_tensor = np.concatenate([image, prompt_heat], axis=0)  # (4,256,256)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(device)
    return input_tensor

def postprocess_prediction(pred):
    """Convert model output logits to binary mask."""
    pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    return pred

# ---------------------------
# GUI Code using Tkinter
# ---------------------------
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class SegmentationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Prompt-based Segmentation")
        
        # Variables
        self.image = None          # Original image as numpy array (BGR)
        self.photo = None          # PhotoImage for display
        self.prompt_point = None   # (row, col) for the prompt point
        self.prompt_marker = None  # Canvas item ID for prompt marker
        
        # Create UI elements.
        self.btn_load = tk.Button(master, text="Load Image", command=self.load_image)
        self.btn_load.pack(pady=5)
        
        self.canvas = tk.Canvas(master, width=256, height=256, bg='gray')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.btn_segment = tk.Button(master, text="Segment", command=self.segment_image, state=tk.DISABLED)
        self.btn_segment.pack(pady=5)
        
        self.btn_save = tk.Button(master, text="Save Figure", command=self.save_figure, state=tk.DISABLED)
        self.btn_save.pack(pady=5)
        
    def load_image(self):
        # Reset prior state.
        self.prompt_point = None
        self.prompt_marker = None
        self.btn_segment.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.canvas.delete("all")
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        # Load image using OpenCV.
        self.image = cv2.imread(file_path)
        if self.image is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        # Convert for display.
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil = image_pil.resize((256,256))
        self.photo = ImageTk.PhotoImage(image_pil)
        self.canvas.config(width=256, height=256)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.master.title(f"Loaded: {file_path}")
    
    def on_click(self, event):
        if self.image is None:
            return
        # Clear any previous prompt marker.
        if self.prompt_marker is not None:
            self.canvas.delete(self.prompt_marker)
        # Save new prompt coordinate.
        self.prompt_point = (event.y, event.x)  # row, col (canvas coordinates)
        r = 4
        self.prompt_marker = self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, outline='red', width=2)
        self.btn_segment.config(state=tk.NORMAL)
    
    def segment_image(self):
        if self.image is None or self.prompt_point is None:
            messagebox.showwarning("Warning", "Please load an image and click on it first.")
            return
        # Create prompt heatmap.
        prompt_heat = create_prompt_heatmap((256,256), self.prompt_point, sigma=5)
        # Preprocess image with prompt.
        input_tensor = preprocess_for_model(self.image, prompt_heat)
        with torch.no_grad():
            output = model(input_tensor)
        pred_mask = postprocess_prediction(output)
        # print(np.unique(pred_mask))
        self.show_visualization(prompt_heat, pred_mask)
        self.btn_save.config(state=tk.NORMAL)
    
    def show_visualization(self, prompt_heat, pred_mask):
        # Prepare the original image.
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (256,256))
        # Create a Matplotlib figure.
        self.fig, axs = plt.subplots(1, 3, figsize=(12,4))
        axs[0].imshow(image_rgb)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(prompt_heat, cmap='hot')
        axs[1].set_title("Prompt Heat Map")
        axs[1].axis("off")
        axs[2].imshow(pred_mask, cmap='gray')
        axs[2].set_title("Segmentation")
        axs[2].axis("off")
        plt.tight_layout()
        plt.show()
    
    def save_figure(self):
        if self.fig is None:
            messagebox.showwarning("Warning", "No figure to save!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if file_path:
            self.fig.savefig(file_path)
            messagebox.showinfo("Saved", f"Figure saved to {file_path}")

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    root = tk.Tk()
    app = SegmentationGUI(root)
    root.mainloop()