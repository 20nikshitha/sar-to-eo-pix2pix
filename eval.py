# data and large artefacts (cite + link instead of committing)
data/
outputs/**/*.pth
*.pth
*.ckpt

# python
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/

# os
.DS_Store

# keep the outputs folder, the loss logs, curves and figures
!outputs/.gitkeep
!outputs/**/losses.csv
!outputs/**/loss_curve.png
!outputs/*.png
