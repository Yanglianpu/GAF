# FlameGAF-GateNet (GAF + Gated Multimodal Fusion for Flame/Smoke Detection)

This repository provides the official implementation of **FlameGAF-GateNet**, a gated multimodal framework for flame and smoke detection that combines **RGB appearance** with **temporal dynamics encoded by Gramian Angular Fields (GAFs)**.

✅ **This codebase is directly associated with the manuscript currently submitted to _The Visual Computer_:**  
**“Enhanced Flame and Smoke Detection via Gated Multimodal Fusion with Gramian Angular Fields.”**

If you find this repository useful, please consider citing our manuscript (see [Citation](#citation)).

---

## Highlights
- **Dual-branch architecture**: RGB branch + GAF branch.
- **GAF temporal encoding**: converts lightweight temporal descriptors into image-like GAF representations.
- **Sample-adaptive gated fusion**: learns per-sample fusion weights for robust multimodal integration.
- Reproducible training/evaluation scripts.

---

## Repository Structure
> (Update these names if your folders differ.)


---

## Requirements

### Environment
- OS: Linux/Windows/macOS
- Python: **3.8+** (recommended: 3.9/3.10)
- PyTorch: **[your version]** (CUDA **[your CUDA version]** if using GPU)

### Install
```bash
git clone https://github.com/Yanglianpu/GAF.git
cd GAF

# Option 1: pip
pip install -r requirements.txt

# Option 2: conda (example)
# conda create -n flamegaf python=3.10 -y
# conda activate flamegaf
# pip install -r requirements.txt

data/
  train/
    fire/
    smoke/
    normal/
  val/
    fire/
    smoke/
    normal/
  test/
    fire/
    smoke/
    normal/

python train.py \
  --mode rgb \
  --data_root data \
  --backbone [resnet18|resnet50|efficientnet_b0] \
  --epochs  [EPOCHS] \
  --batch_size [BS] \
  --lr [LR] \
  --output_dir runs/rgb_baseline
python train.py \
  --mode gaf \
  --data_root data \
  --epochs  [EPOCHS] \
  --batch_size [BS] \
  --lr [LR] \
  --output_dir runs/gaf_baseline
python train.py \
  --mode fusion \
  --fusion_type gate \
  --data_root data \
  --rgb_backbone [resnet18|resnet50|efficientnet_b0] \
  --gaf_backbone [resnet18|resnet50] \
  --epochs  [EPOCHS] \
  --batch_size [BS] \
  --lr [LR] \
  --output_dir runs/flamegaf_gatenet
python eval.py \
  --mode fusion \
  --fusion_type gate \
  --data_root data \
  --ckpt_path runs/flamegaf_gatenet/best.pth
python train.py --mode fusion --fusion_type gate --data_root data --output_dir runs/flamegaf_gatenet
@article{FlameGAFGateNet2026,
  title   = {Enhanced Flame and Smoke Detection via Gated Multimodal Fusion with Gramian Angular Fields},
  author  = {Yang, Lianpu and [Coauthors]},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript submitted},
}

---

### 我建议你再补一个小文件（编辑很喜欢）
在仓库根目录加一个 `CITATION.cff` 或 `CITATION.md`，并在 README 的 “Citation” 里链接过去。这样“提醒读者引用稿件”的要求会更明显。

如果你把仓库的实际结构（比如 `train.py`/`eval.py` 参数名、数据目录真实长什么样）贴我一下，我可以把上面 README 里的命令行参数**改成完全匹配你代码的版本**，做到“复制就能跑”。
::contentReference[oaicite:0]{index=0}

