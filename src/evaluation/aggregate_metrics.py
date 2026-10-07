import torch
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import shutil
from pytorch_fid import fid_score
from torchvision.models import inception_v3
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
import lpips
import pytesseract

# --- MODELE ---
class Generator(torch.nn.Module):
    def __init__(self, z_dim, g_hidden, image_channel):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_dim, g_hidden * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(g_hidden * 8),
            torch.nn.ReLU(inplace=False),
            torch.nn.ConvTranspose2d(g_hidden * 8, g_hidden * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(g_hidden * 4),
            torch.nn.ReLU(inplace=False),
            torch.nn.ConvTranspose2d(g_hidden * 4, g_hidden * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(g_hidden * 2),
            torch.nn.ReLU(inplace=False),
            torch.nn.ConvTranspose2d(g_hidden * 2, g_hidden, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(g_hidden),
            torch.nn.ReLU(inplace=False),
            torch.nn.ConvTranspose2d(g_hidden, image_channel, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(torch.nn.Module):
    def __init__(self, d_hidden, image_channel):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(image_channel, d_hidden, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(d_hidden, d_hidden * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(d_hidden * 2),
            torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(d_hidden * 2, d_hidden * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(d_hidden * 4),
            torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(d_hidden * 4, d_hidden * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(d_hidden * 8),
            torch.nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Conv2d(d_hidden * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# --- KONFIGURACJA ---
conf = {
    'z_dim': 100,
    'g_hidden': 64,
    'd_hidden': 64,
    'image_channel': 1,
    'batch_size': 256,
    'image_size': 64,
    'data_path': 'letters_new',
    'csv_file': 'letters_labels_cleaned.csv'
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_real_imgs(max_files=1000, img_size=64):
    real_imgs_dir = 'real_imgs'
    os.makedirs(real_imgs_dir, exist_ok=True)
    images_in_dir = [f for f in os.listdir(real_imgs_dir) if f.endswith('.png')]
    if len(images_in_dir) >= max_files:
        print("Folder real_imgs nie jest pusty – pomijam kopiowanie.")
        return
    df = pd.read_csv(conf['csv_file'])
    real_letters = df[df['label'] == 0].head(max_files)
    for i, row in real_letters.iterrows():
        src_path = os.path.join(conf['data_path'], row['filename'])
        dst_path = os.path.join(real_imgs_dir, row['filename'])
        if os.path.isfile(src_path):
            img = Image.open(src_path).convert('RGB')
            img = img.resize((img_size, img_size), Image.LANCZOS)
            img.save(dst_path)
    print(f'Skopiowano {len(real_letters)} prawdziwych liter do {real_imgs_dir}/ (wszystkie {img_size}x{img_size})')

def load_generator(ckpt_path):
    netG = Generator(conf['z_dim'], conf['g_hidden'], conf['image_channel']).to(device)
    netG.load_state_dict(torch.load(ckpt_path, map_location=device))
    netG.eval()
    return netG

def load_discriminator(ckpt_path):
    netD = Discriminator(conf['d_hidden'], conf['image_channel']).to(device)
    netD.load_state_dict(torch.load(ckpt_path, map_location=device))
    netD.eval()
    return netD

def generate_fake_imgs(netG, fake_imgs_dir='fake_imgs', img_size=64, total_imgs=1000):
    batch_size = conf['batch_size']
    os.makedirs(fake_imgs_dir, exist_ok=True)
    files = glob.glob(os.path.join(fake_imgs_dir, '*.png'))
    for f in files:
        os.remove(f)
    to_pil = transforms.ToPILImage()
    idx = 0
    n_batches = (total_imgs + batch_size - 1) // batch_size
    for b in range(n_batches):
        current_batch = min(batch_size, total_imgs - idx)
        noise = torch.randn(current_batch, conf['z_dim'], 1, 1, device=device)
        with torch.no_grad():
            fake_images = netG(noise).detach().cpu()
        for i, img in enumerate(fake_images):
            img = img.clamp(-1, 1) * 0.5 + 0.5
            img_pil = to_pil(img)
            img_pil = img_pil.convert('RGB')
            img_pil = img_pil.resize((img_size, img_size), Image.LANCZOS)
            img_pil.save(os.path.join(fake_imgs_dir, f'fake_{idx}.png'))
            idx += 1
    print(f"Wygenerowano {idx} obrazów do folderu {fake_imgs_dir}")
    return idx

def compute_fid():
    return fid_score.calculate_fid_given_paths(['real_imgs', 'fake_imgs'], batch_size=32, device=device, dims=2048)

def inception_score(fake_imgs_dir='fake_imgs', splits=10):
    fake_files = sorted(glob.glob(os.path.join(fake_imgs_dir, '*.png')))
    all_imgs = []
    tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    for f in fake_files:
        img = Image.open(f).convert('RGB')
        img = tf(img)
        all_imgs.append(img)
    imgs_tensor = torch.stack(all_imgs, dim=0)

    inception_model = inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(device)
    inception_model.eval()
    with torch.no_grad():
        preds = inception_model(imgs_tensor.to(device))
    preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
    split_scores = []
    N = preds.shape[0]
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * (np.log(pyx) - np.log(py))))
        split_scores.append(np.exp(np.mean(scores)))
    score = np.mean(split_scores)
    std = np.std(split_scores)
    return score, std

def mse_ssim_lpips(real_folder, fake_folder, n_pairs=50):
    mse_vals = []
    ssim_vals = []
    lpips_vals = []
    lpips_alex = lpips.LPIPS(net='alex').to(device)
    real_files = sorted(glob.glob(os.path.join(real_folder, '*.png')))
    fake_files = sorted(glob.glob(os.path.join(fake_folder, '*.png')))
    n = min(len(real_files), len(fake_files), n_pairs)
    for r, f in zip(real_files[:n], fake_files[:n]):
        img_real = np.array(Image.open(r).convert('RGB'))
        img_fake = np.array(Image.open(f).convert('RGB'))
        if img_real.shape != img_fake.shape:
            continue
        mse_vals.append(mse(img_real, img_fake))
        ssim_vals.append(ssim(img_real, img_fake, channel_axis=-1))
        img_real_t = transforms.ToTensor()(Image.open(r).convert('RGB')).unsqueeze(0).to(device)
        img_fake_t = transforms.ToTensor()(Image.open(f).convert('RGB')).unsqueeze(0).to(device)
        lpips_score = lpips_alex(img_real_t, img_fake_t).item()
        lpips_vals.append(lpips_score)
    return np.mean(mse_vals), np.mean(ssim_vals), np.mean(lpips_vals)

def hist_compare(real_folder, fake_folder):
    real_files = glob.glob(os.path.join(real_folder, '*.png'))
    fake_files = glob.glob(os.path.join(fake_folder, '*.png'))
    def get_hist(filelist):
        hists = []
        for f in filelist:
            img = Image.open(f).convert('L')
            img_np = np.array(img).flatten()
            hist, _ = np.histogram(img_np, bins=256, range=(0, 255))
            hists.append(hist)
        return np.mean(hists, axis=0)
    h_real = get_hist(real_files)
    h_fake = get_hist(fake_files)
    corr = np.corrcoef(h_real, h_fake)[0,1]
    return corr

def ocr_recognizability(fake_folder, n=100):
    fake_files = sorted(glob.glob(os.path.join(fake_folder, '*.png')))[:n]
    good = 0
    bad = 0
    recognized = []
    for f in fake_files:
        img = Image.open(f).convert('RGB')
        txt = pytesseract.image_to_string(img, config='--psm 10').strip()
        if len(txt) == 1 and txt.isalnum():
            good += 1
            recognized.append(txt)
        else:
            bad += 1
    total = good + bad
    percent = 100*good/total if total > 0 else 0
    return percent, recognized

def unique_letters_by_ocr(recognized_list):
    unique = set(recognized_list)
    return len(unique), unique

def discriminator_accuracy(disc_ckpt, fake_imgs_dir='fake_imgs', real_imgs_dir='real_imgs', n=100):
    netD = load_discriminator(disc_ckpt)
    real_files = sorted(glob.glob(os.path.join(real_imgs_dir, '*.png')))[:n]
    fake_files = sorted(glob.glob(os.path.join(fake_imgs_dir, '*.png')))[:n]
    transform = transforms.Compose([
        transforms.Resize((conf['image_size'], conf['image_size'])),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    all_imgs = []
    all_labels = []
    for f in real_files:
        img = transform(Image.open(f))
        all_imgs.append(img)
        all_labels.append(0)
    for f in fake_files:
        img = transform(Image.open(f))
        all_imgs.append(img)
        all_labels.append(1)
    imgs = torch.stack(all_imgs).to(device)
    labels = torch.tensor(all_labels).to(device)
    with torch.no_grad():
        preds = netD(imgs)
        preds_bin = (preds > 0.5).long()
    acc = (preds_bin == labels).float().mean().item()
    return acc

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    prepare_real_imgs(max_files=1000, img_size=64)
    run_dirs = sorted([d for d in os.listdir() if os.path.isdir(d) and d.startswith("run_")])
    all_metrics = []

    for run_dir in run_dirs:
        print(f"\n==================== {run_dir} ====================")
        # Automatyczna detekcja wszystkich generatorów (epoch_X_Y) w tym runie
        gen_ckpts = sorted(glob.glob(f"{run_dir}/generator_epoch_*_*.pth"),
                           key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[2]))
        disc_ckpts = sorted(glob.glob(f"{run_dir}/discriminator_epoch_*_*.pth"),
                           key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[2]))

        epochs = []
        fids = []
        is_means = []
        is_stds = []
        mses = []
        ssims = []
        lpips_scores = []
        hist_corrs = []
        ocr_percents = []
        ocr_uniques = []
        disc_accs = []

        for idx, (g_ckpt, d_ckpt) in enumerate(zip(gen_ckpts, disc_ckpts)):
            # Wyciągnij numer epoki z pliku, np. generator_epoch_10_3.pth
            epoch = int(os.path.splitext(os.path.basename(g_ckpt))[0].split('_')[2])
            print(f"\n==== Ewaluacja: {run_dir}, epoka {epoch} ====")
            netG = load_generator(g_ckpt)
            generate_fake_imgs(netG, fake_imgs_dir='fake_imgs', img_size=64, total_imgs=1000)
            fid = compute_fid()
            is_mean, is_std = inception_score('fake_imgs')
            mse_mean, ssim_mean, lpips_mean = mse_ssim_lpips('real_imgs', 'fake_imgs', n_pairs=50)
            hist_corr = hist_compare('real_imgs', 'fake_imgs')
            ocr_percent, recognized_list = ocr_recognizability('fake_imgs', n=100)
            unique_count, unique_letters = unique_letters_by_ocr(recognized_list)
            disc_acc = discriminator_accuracy(d_ckpt, 'fake_imgs', 'real_imgs', n=100)

            # Zbieranie wyników
            epochs.append(epoch)
            fids.append(fid)
            is_means.append(is_mean)
            is_stds.append(is_std)
            mses.append(mse_mean)
            ssims.append(ssim_mean)
            lpips_scores.append(lpips_mean)
            hist_corrs.append(hist_corr)
            ocr_percents.append(ocr_percent)
            ocr_uniques.append(unique_count)
            disc_accs.append(disc_acc)

            # Do pełnego zbioru (z info o runie)
            all_metrics.append({
                'run': run_dir,
                'epoch': epoch,
                'FID': fid,
                'IS_mean': is_mean,
                'IS_std': is_std,
                'MSE': mse_mean,
                'SSIM': ssim_mean,
                'LPIPS': lpips_mean,
                'HistCorr': hist_corr,
                'OCR%': ocr_percent,
                'UniqueOCR': unique_count,
                'DiscAcc': disc_acc
            })

        # --- Wykresy dla tej serii ---
        plt.figure(figsize=(14,8))
        plt.subplot(2,3,1)
        plt.plot(epochs, fids, marker='o'); plt.title(f'{run_dir}: FID'); plt.xlabel('Epoch'); plt.grid(True)
        plt.subplot(2,3,2)
        plt.errorbar(epochs, is_means, yerr=is_stds, marker='o', capsize=3); plt.title(f'{run_dir}: IS'); plt.xlabel('Epoch'); plt.grid(True)
        plt.subplot(2,3,3)
        plt.plot(epochs, mses, marker='o'); plt.title(f'{run_dir}: MSE'); plt.xlabel('Epoch'); plt.grid(True)
        plt.subplot(2,3,4)
        plt.plot(epochs, ssims, marker='o'); plt.title(f'{run_dir}: SSIM'); plt.xlabel('Epoch'); plt.grid(True)
        plt.subplot(2,3,5)
        plt.plot(epochs, lpips_scores, marker='o'); plt.title(f'{run_dir}: LPIPS'); plt.xlabel('Epoch'); plt.grid(True)
        plt.subplot(2,3,6)
        plt.plot(epochs, ocr_percents, marker='o'); plt.title(f'{run_dir}: OCR %'); plt.xlabel('Epoch'); plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"metrics_vs_epoch_{run_dir}.png")
        plt.close()

        # Zapis do CSV
        df_run = pd.DataFrame({
            'epoch': epochs,
            'FID': fids,
            'IS_mean': is_means,
            'IS_std': is_stds,
            'MSE': mses,
            'SSIM': ssims,
            'LPIPS': lpips_scores,
            'HistCorr': hist_corrs,
            'OCR%': ocr_percents,
            'UniqueOCR': ocr_uniques,
            'DiscAcc': disc_accs
        })
        df_run.to_csv(f"metrics_{run_dir}.csv", index=False)

    # --- Wszystkie serie razem: średnia i std ---
    df_all = pd.DataFrame(all_metrics)
    mean_df = df_all.groupby('epoch').mean(numeric_only=True).reset_index()
    std_df = df_all.groupby('epoch').std(numeric_only=True).reset_index()

    # --- Wykresy uśrednione ---
    plt.figure(figsize=(14,8))
    plt.subplot(2,3,1)
    plt.errorbar(mean_df['epoch'], mean_df['FID'], yerr=std_df['FID'], marker='o', capsize=3); plt.title('Średni FID'); plt.xlabel('Epoch'); plt.grid(True)
    plt.subplot(2,3,2)
    plt.errorbar(mean_df['epoch'], mean_df['IS_mean'], yerr=std_df['IS_mean'], marker='o', capsize=3); plt.title('Średni IS'); plt.xlabel('Epoch'); plt.grid(True)
    plt.subplot(2,3,3)
    plt.errorbar(mean_df['epoch'], mean_df['MSE'], yerr=std_df['MSE'], marker='o', capsize=3); plt.title('Średni MSE'); plt.xlabel('Epoch'); plt.grid(True)
    plt.subplot(2,3,4)
    plt.errorbar(mean_df['epoch'], mean_df['SSIM'], yerr=std_df['SSIM'], marker='o', capsize=3); plt.title('Średni SSIM'); plt.xlabel('Epoch'); plt.grid(True)
    plt.subplot(2,3,5)
    plt.errorbar(mean_df['epoch'], mean_df['LPIPS'], yerr=std_df['LPIPS'], marker='o', capsize=3); plt.title('Średni LPIPS'); plt.xlabel('Epoch'); plt.grid(True)
    plt.subplot(2,3,6)
    plt.errorbar(mean_df['epoch'], mean_df['OCR%'], yerr=std_df['OCR%'], marker='o', capsize=3); plt.title('Średni OCR %'); plt.xlabel('Epoch'); plt.grid(True)
    plt.tight_layout()
    plt.savefig("metrics_vs_epoch_mean.png")
    plt.close()

    # Zapis wszystkich wyników do CSV
    df_all.to_csv("all_metrics_all_runs.csv", index=False)
    mean_df.to_csv("mean_metrics_all_runs.csv", index=False)

    print("Gotowe! Wszystkie metryki, csv i wykresy zapisane.")

