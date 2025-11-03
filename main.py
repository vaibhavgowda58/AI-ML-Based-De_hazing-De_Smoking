import os
import tarfile
import cv2
import torch
import gradio as gr
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dotenv import load_dotenv
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim
from guided_filter_pytorch.guided_filter import FastGuidedFilter

# Load environment variables
load_dotenv()

# ----------------- 1. Extract Dataset -----------------
def extract_tar(file_path, extract_to):
    if os.path.exists(extract_to):
        print(f"{extract_to} already exists, skipping extraction.")
        return

    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(file_path, 'r') as tar:
        members = tar.getmembers()
        base_folder = members[0].name.split('/')[0] if '/' in members[0].name else members[0].name
        tar.extractall(path=extract_to)
        
        extracted_path = os.path.join(extract_to, base_folder)
        if os.path.exists(extracted_path) and extracted_path != extract_to:
            for item in os.listdir(extracted_path):
                os.rename(os.path.join(extracted_path, item), os.path.join(extract_to, item))
            os.rmdir(extracted_path)

    print(f"Extracted {file_path} to {extract_to}")

# Define paths
data_root = "./revide_inside"
train_tar = "C:/Users/darsh/Documents/dehazing/Train.tar"
test_tar = "C:/Users/darsh/Documents/dehazing/Test.tar"
train_dir = os.path.join(data_root, "Train", "Train")
test_dir = os.path.join(data_root, "Test", "Test")

# Extract datasets
extract_tar(train_tar, train_dir)
extract_tar(test_tar, test_dir)

# ----------------- 2. Define Dataset Class -----------------

class DehazingDataset(Dataset):
    def _init_(self, hazy_dir, clear_dir, transform=None):
        self.hazy_images = sorted(self._get_all_images(hazy_dir))
        self.clear_images = sorted(self._get_all_images(clear_dir))
        self.transform = transform

    def _get_all_images(self, root_dir):
        image_paths = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(folder_path, file))
        return image_paths

    def _len_(self):
        return min(len(self.hazy_images), len(self.clear_images))

    def _getitem_(self, idx):
        hazy_path = self.hazy_images[idx]
        clear_path = self.clear_images[idx]
        hazy = Image.open(hazy_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")

        if self.transform:
            hazy = self.transform(hazy)
            clear = self.transform(clear)

        return hazy, clear

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = DehazingDataset(os.path.join(train_dir, "hazy"), os.path.join(train_dir, "gt"), transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ----------------- 3. Define Advanced Model -----------------
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def apply_bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def preprocess_image(image):
    image = apply_clahe(image)
    image = apply_bilateral_filter(image)
    return image

class AdvancedDehazingNet(torch.nn.Module):
    def _init_(self):
        super(AdvancedDehazingNet, self)._init_()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-2])
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load model
model = AdvancedDehazingNet().to(device)
model.load_state_dict(torch.load("dehaze_gan.pth", map_location=device, weights_only=True))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
criterion = nn.L1Loss() 

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999)) 


# ----------------- 4. Train Model ----------------- 
def train_model(num_epochs=5): 
    model.train() 
    for epoch in range(num_epochs): 
        epoch_loss = 0 
        for hazy, clear in train_loader: 
            hazy, clear = hazy.to(device), clear.to(device) 
            optimizer.zero_grad() 
            output = model(hazy) 
            loss = criterion(output, clear) 
            loss.backward() 
            optimizer.step() 
            epoch_loss += loss.item() 
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}") 
    torch.save(model.state_dict(), "dehaze_gan.pth") 
 
# ----------------- 5. Load Model ----------------- 
def load_model(): 
    if os.path.exists("dehaze_gan.pth"): 
        print("Loading model weights...") 
        model.load_state_dict(torch.load("dehaze_gan.pth", map_location=device, weights_only=True), strict=False) 
        model.eval() 
        print("Model loaded successfully!") 
    else: 
        print("No model weights found! Training might be required.")

#-------findindg accuarcy using psnr,ssim,mae-------

from skimage.metrics import structural_similarity as ssim
import math

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Perfect match
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def evaluate_model():
    total_psnr, total_ssim, total_mae = 0, 0, 0
    num_samples = 0

    for hazy, clear in train_loader:
        hazy, clear = hazy.to(device), clear.to(device)

        with torch.no_grad():
            dehazed = model(hazy)

        hazy_np = hazy.cpu().numpy().transpose(0, 2, 3, 1)
        clear_np = clear.cpu().numpy().transpose(0, 2, 3, 1)
        dehazed_np = dehazed.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(hazy_np.shape[0]):
            psnr = calculate_psnr(dehazed_np[i], clear_np[i])
            ssim_value = ssim(dehazed_np[i], clear_np[i], channel_axis=-1, win_size=11, data_range=1.0)
            mae = np.mean(np.abs(dehazed_np[i] - clear_np[i]))

            total_psnr += psnr
            total_ssim += ssim_value
            total_mae += mae
            num_samples += 1

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_mae = total_mae / num_samples

    print(f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, MAE: {avg_mae:.4f}")
    return f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, MAE: {avg_mae:.4f}"


# Dehazing function
def process_dehazing(input_path, model):
    if input_path.lower().endswith(('png', 'jpg', 'jpeg')):
        return dehaze_image(input_path, model)
    elif input_path.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
        return dehaze_video(input_path, model)
    else:
        raise ValueError("Unsupported file format")

# High-pass sharpening
def apply_sharpening(image):
    gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened

# Dehaze single image
def dehaze_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu()
    output = torch.clamp(output, 0, 1)
    output_image = transforms.ToPILImage()(output)
    temp_path = "dehazed_temp.png"
    output_image.save(temp_path)

    # High-pass sharpening
    img = cv2.imread(temp_path)
    sharpened = apply_sharpening(img)
    final_output_path = "dehazed_final.png"
    cv2.imwrite(final_output_path, sharpened)
    return final_output_path

# Dehaze video
def dehaze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "dehazed_video.avi"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 256x256 for model input
        resized_frame = cv2.resize(frame, (256, 256))
        image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor).squeeze(0).cpu()

        output = torch.clamp(output, 0, 1)
        output_image = transforms.ToPILImage()(output)
        output_image = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)

        # Resize back to original dimensions and apply sharpening
        output_image = cv2.resize(output_image, (width, height), interpolation=cv2.INTER_LINEAR)
        sharpened_frame = apply_sharpening(output_image)
        out.write(sharpened_frame)

    cap.release()
    out.release()
    return output_path

# Gradio Interface
# if _name_ == "_main_":
#     gr.Interface(
#         fn=lambda input_path: process_dehazing(input_path, model),
#         inputs=gr.File(type="filepath"),
#         outputs=gr.File(type="filepath"),
#         title="AI-Powered Dehazing App",
#         description="Upload an image or video to remove haze using deep learning with high-pass sharpening.",
#     ).launch()

if _name_ == "_main_":
    # train_model()
    # evaluate_model()
    with gr.Blocks(css="body { font-family: Arial; background-color: #f4f4f4; text-align: center; } h1 { color: white; }") as demo:
        gr.Markdown("# <h1>AI-Powered Dehazing App</h1>")
        gr.Markdown("Upload an image or video, preview the dehazed output, and download it.")

        with gr.Row():
            input_file = gr.File(type="filepath", label="Upload Hazy Image/Video")

        with gr.Row():
            output_display = gr.Image(label="Dehazed Image", visible=False)
            video_display = gr.Video(label="Dehazed Video", visible=False)

        process_btn = gr.Button("Process Dehazing")
        download_btn = gr.File(label="Download Processed File", visible=False)

        def process_file(input_path):
            output_path = process_dehazing(input_path, model)
            if input_path.lower().endswith(('png', 'jpg', 'jpeg')):
                return output_path, output_path, None, gr.update(visible=True), gr.update(visible=False)
            else:
                return output_path, None, output_path, gr.update(visible=False), gr.update(visible=True)

        process_btn.click(process_file, inputs=input_file, outputs=[download_btn, output_display, video_display, output_display, video_display])

    demo.launch()
