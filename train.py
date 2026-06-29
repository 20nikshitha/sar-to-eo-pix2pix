{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SAR -> EO Image Translation \u2014 run on Kaggle (free GPU)\n",
    "\n",
    "This notebook trains the model, evaluates it, and produces every result you need\n",
    "for the GalaxEye submission. **You do not need to read or understand the code.**\n",
    "Just do the one-time setup below, then press **Run All** and wait.\n",
    "\n",
    "## One-time setup before you press Run All\n",
    "1. **Turn on the GPU.** Right panel -> Settings -> **Accelerator** -> **GPU** (T4 or P100).\n",
    "2. **Turn on internet.** Right panel -> Settings -> **Internet** -> **On**\n",
    "   (needs a phone-verified Kaggle account).\n",
    "3. **Add the dataset.** Right panel -> **+ Add Input** -> search\n",
    "   `sentinel12 image pairs segregated by terrain` -> **Add** the `requiemonk` one.\n",
    "4. **Add your code.** First upload `sar2eo.zip` as a dataset\n",
    "   (Create -> New Dataset -> drag the zip -> title it `sar2eo code` -> Create).\n",
    "   Then here: **+ Add Input** -> **Your Datasets** -> add `sar2eo code`.\n",
    "\n",
    "Then: top menu **Run -> Run all**. The training cell takes 1-3 hours and runs by\n",
    "itself. When it finishes, copy the printed numbers and download the files at the\n",
    "very bottom.\n"
   ],
   "id": "064dcf04"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Get the code ready *(just run it)*"
   ],
   "id": "50ef43eb"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import os, shutil, glob, zipfile\n",
    "src = None\n",
    "for root, _, files in os.walk('/kaggle/input'):\n",
    "    if 'train.py' in files and 'infer.py' in files:\n",
    "        src = root; break\n",
    "if src is None:                       # code still zipped? unzip it\n",
    "    for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):\n",
    "        zipfile.ZipFile(z).extractall('/kaggle/working/_code')\n",
    "    for root, _, files in os.walk('/kaggle/working/_code'):\n",
    "        if 'train.py' in files: src = root; break\n",
    "assert src, \"Code not found - did you add the 'sar2eo code' dataset as input?\"\n",
    "dst = '/kaggle/working/sar2eo'\n",
    "if os.path.abspath(src) != os.path.abspath(dst):\n",
    "    shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)\n",
    "os.chdir(dst)\n",
    "print(\"OK - working inside:\", os.getcwd())\n",
    "print(sorted(os.listdir('.')))"
   ],
   "id": "f84bd1f6"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Install two small tools *(needs Internet = On)*"
   ],
   "id": "c6ec2e2d"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!pip install -q lpips pytorch-fid\n",
    "print(\"done\")"
   ],
   "id": "fc039d80"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Find the image dataset *(just run it)*"
   ],
   "id": "615f1eaf"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import os\n",
    "data_src = None\n",
    "for root, dirs, _ in os.walk('/kaggle/input'):\n",
    "    if any(os.path.isdir(os.path.join(root,d,'s1')) and\n",
    "           os.path.isdir(os.path.join(root,d,'s2')) for d in dirs):\n",
    "        data_src = root; break\n",
    "assert data_src, \"Dataset not found - did you add the requiemonk dataset as input?\"\n",
    "print(\"Dataset is at:\", data_src)"
   ],
   "id": "889479dc"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Make the first run faster *(optional)*\n",
    "40 epochs gives a decent first result in ~1-3 h. Raise it later for better quality."
   ],
   "id": "c91984eb"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import re\n",
    "c = open('config.yaml').read()\n",
    "c = re.sub(r'epochs:\\s*\\d+', 'epochs: 40', c)\n",
    "open('config.yaml','w').write(c)\n",
    "print(\"epochs set to 40\")"
   ],
   "id": "817e3b69"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Prepare the data *(about 1-2 minutes)*"
   ],
   "id": "c48bd271"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!python scripts/prepare_dataset.py --src \"{data_src}\" --dst data/sen12 --val_frac 0.1 --max_per_terrain 2000"
   ],
   "id": "2a15e2bd"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Train the main model *(the long step: 1-3 hours)*\n",
    "Loss numbers tick down each epoch \u2014 that is what you want. Leave the tab open."
   ],
   "id": "b5fcbea5"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!python train.py --config config.yaml"
   ],
   "id": "dc65ba97"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Generate predictions on the validation images"
   ],
   "id": "f8fdd07d"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!python infer.py --input_dir data/sen12/val/sar --output_dir outputs/pred_val --weights outputs/pix2pix_sen12_baseline/best.pth"
   ],
   "id": "4e19f3a8"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Score the model  \u2b05\ufe0f  COPY THESE 4 NUMBERS\n",
    "These are your **baseline (L1+GAN)** results for the report."
   ],
   "id": "98ed7486"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!python eval.py --pred_dir outputs/pred_val --gt_dir data/sen12/val/eo"
   ],
   "id": "a96e7d11"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Make the before/after pictures"
   ],
   "id": "3b4f1484"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!python scripts/make_qualitative.py --weights outputs/pix2pix_sen12_baseline/best.pth --data_root data/sen12 --split val --out outputs/qualitative.png --n 6"
   ],
   "id": "e6eeb3a8"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. The required experiment: turn the GAN OFF and retrain *(slow again)*\n",
    "This is the ablation the assignment asks for (L1-only vs L1+GAN)."
   ],
   "id": "62b2cf95"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import re\n",
    "c = open('config.yaml').read()\n",
    "c = re.sub(r'use_gan:\\s*\\w+', 'use_gan: false', c)\n",
    "c = re.sub(r'experiment_name:.*', 'experiment_name: pix2pix_l1only', c)\n",
    "open('config.yaml','w').write(c)\n",
    "print(\"switched to L1-only mode\")\n",
    "!python train.py --config config.yaml"
   ],
   "id": "7ce18bd4"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Score the experiment  \u2b05\ufe0f  COPY THESE 4 NUMBERS\n",
    "These are your **L1-only (ablation)** results for the report."
   ],
   "id": "3dfd0ad4"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!python infer.py --input_dir data/sen12/val/sar --output_dir outputs/pred_val_l1only --weights outputs/pix2pix_l1only/best.pth\n",
    "!python eval.py --pred_dir outputs/pred_val_l1only --gt_dir data/sen12/val/eo"
   ],
   "id": "3dc20d0c"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Download everything you need\n",
    "Click each blue link to save it to your computer."
   ],
   "id": "d7f43814"
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from IPython.display import FileLink, display\n",
    "import os\n",
    "for f in ['outputs/pix2pix_sen12_baseline/best.pth',\n",
    "          'outputs/pix2pix_sen12_baseline/loss_curve.png',\n",
    "          'outputs/pix2pix_sen12_baseline/losses.csv',\n",
    "          'outputs/qualitative.png']:\n",
    "    print(f)\n",
    "    display(FileLink(f)) if os.path.exists(f) else print(\"   (missing)\")"
   ],
   "id": "5f5d1f38"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## After this notebook (done on your computer)\n",
    "1. **Report:** open `report/technical_report.md`, replace every `[TODO]`/`[FILL]`\n",
    "   with the 8 numbers (steps 8 & 11) and your time/machine info. Make a PDF at\n",
    "   markdowntopdf.com.\n",
    "2. **Weights:** upload `best.pth` to Google Drive -> Share -> *Anyone with the link*.\n",
    "   Put that link in the README and the submission form.\n",
    "3. **GitHub:** create a public repo, upload the code files (not `best.pth`).\n",
    "4. **Final ZIP:** zip the report PDF + time log + `loss_curve.png` + `qualitative.png`,\n",
    "   rename to `FirstName_LastName_GalaxEye.zip`.\n",
    "5. **Submit** the form with: GitHub link, Drive weights link, and the ZIP.\n"
   ],
   "id": "36ec66bc"
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  },
  "accelerator": "GPU"
 },
 "nbformat": 4,
 "nbformat_minor": 5
}