import sys
import cv2
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from functools import partial

# === VideoMAE モデル用パスを追加 ===
sys.path.append('/home/ollo/VideoMAE')
sys.path.append('/home/ollo/VideoMAE/AVION')
from AVION.avion.models.transformer import VisionTransformer

# === カテゴリマップ（ID→ラベル） ===
id_to_label = {
    0: "r_写っている", 1: "r_tc_はい", 2: "r_切る", 3: "r_削る", 4: "r_掘る", 5: "r_描く", 6: "r_注ぐ", 7: "r_引く", 8: "r_押す",
    9: "r_置く", 10: "r_取る", 11: "r_押さえる", 12: "r_回す", 13: "r_ひっくり返す", 14: "r_落とす", 15: "r_叩く", 16: "r_縛る・折る",
    17: "r_締める", 18: "r_緩める", 19: "r_開ける", 20: "r_閉める", 21: "r_洗う", 22: "r_拭く", 23: "r_捨てる", 24: "r_投げる", 25: "r_混ぜる",
    26: "l_写っている", 27: "l_tc_はい", 28: "l_切る", 29: "l_削る", 30: "l_掘る", 31: "l_描く", 32: "l_注ぐ", 33: "l_引く", 34: "l_押す",
    35: "l_置く", 36: "l_取る", 37: "l_押さえる", 38: "l_回す", 39: "l_ひっくり返す", 40: "l_落とす", 41: "l_叩く", 42: "l_縛る・折る",
    43: "l_締める", 44: "l_緩める", 45: "l_開ける", 46: "l_閉める", 47: "l_洗う", 48: "l_拭く", 49: "l_捨てる", 50: "l_投げる", 51: "l_混ぜる",
    52: "c_右手→左手", 53: "c_左手→右手", 54: "s_触って調べる", 55: "s_読む・調べる", 56: "m_はい", 57: "t_はい"
}

# === モデル読み込み（安全にキーをフィルタリング） ===
def load_model_videomae(weight_path, device):
    model = VisionTransformer(
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        num_classes=58,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )
    ckpt = torch.load(weight_path, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    msg = model.load_state_dict(filtered_state, strict=False)
    print(f"[INFO] モデル一部読み込み完了: {msg}")
    model.to(device)
    model.eval()
    return model

# === テキスト描画 ===
def draw_japanese_text(image, text, x, y, font_path, font_size=24, color=(0, 0, 255)):
    b, g, r = color
    rgb_color = (r, g, b)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text((x, y), text, fill=rgb_color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# === 前処理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === 推論 ===
def predict(model, frame, device):
    image = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        topk = probs.argsort()[-3:][::-1]
        return [(id_to_label[i], probs[i]) for i in topk]

# === 可視化処理 ===
def visualize_video(video_path, output_path, model, font_path, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 動画が開けません: {video_path}")
        return

    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        preds = predict(model, frame, device)
        for i, (label, prob) in enumerate(preds):
            text = f"{label} {prob*100:.2f}%"
            frame = draw_japanese_text(frame, text, 10, 30 + i*30, font_path)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"[INFO] {frame_count} フレーム処理完了。出力: {output_path}")

# === 実行 ===
if __name__ == "__main__":
    video_path = "/home/ollo/Ollo_video/office_bike_namiki_sagyo.mp4"
    output_path = "output_vis.mp4"
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    weight_path = "/home/ollo/videomae_finetuned_interactive.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_videomae(weight_path, device)
    visualize_video(video_path, output_path, model, font_path, device)