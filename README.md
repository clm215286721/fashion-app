# 服装以图搜款 - 服务代码

Gradio 以图搜款服务（搜图 + 商家入库），部署在天翼云电脑 ming（`C:\fashion-app`）。

## 更新流程（云电脑上）

```bat
C:\fashion-app\update.bat
```

自动：git pull 最新代码 → 覆盖 py 文件 → 重启服务。

## 本地开发

```bash
pip install -r requirements.txt
python ui.py
```

## 不在仓库里的东西（云电脑已有）

- `models/vision_fp16.onnx`（178MB，HF 官方版）
- `data/merchants/`（商家图片）
- `index/`（faiss 索引）
- `venv/`（Python 虚拟环境）
