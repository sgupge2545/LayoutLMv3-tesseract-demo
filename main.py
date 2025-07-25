from PIL import Image
import pytesseract
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from collections import defaultdict

# ---- ① 入力画像の読み込み ----
image_path = "TEST.JPG"
image = Image.open(image_path).convert("RGB")

# ---- ② Tesseract で OCR + bbox を取得 ----
ocr_data = pytesseract.image_to_data(
    image, output_type=pytesseract.Output.DICT, lang="jpn"
)

words = []
boxes = []

for i in range(len(ocr_data["text"])):
    if int(ocr_data["conf"][i]) > 0 and ocr_data["text"][i].strip() != "":
        words.append(ocr_data["text"][i])
        # 座標は 0-1000 にスケーリングして LayoutLMv3 に合わせる
        x, y, w, h = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )
        boxes.append(
            [
                int(x * 1000 / image.width),
                int(y * 1000 / image.height),
                int((x + w) * 1000 / image.width),
                int((y + h) * 1000 / image.height),
            ]
        )

# ---- ③ LayoutLMv3 でトークン分類 ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "impira/layoutlm-document-qa"
).to(device)

encoding = processor(
    image,
    words,
    boxes=boxes,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
).to(device)
outputs = model(**encoding)
pred_ids = outputs.logits.argmax(-1).squeeze().tolist()
tokens = processor.tokenizer.convert_ids_to_tokens(
    encoding.input_ids.squeeze().tolist()
)

# ---- ④ 結果の表示（全ラベルごとにまとめて表示）----
label_to_tokens = defaultdict(list)
for token, label_id in zip(tokens, pred_ids):
    label = model.config.id2label[label_id]
    label_to_tokens[label].append(token)

for label, token_list in label_to_tokens.items():
    print(f"{label}:")
    print("  " + ", ".join(token_list))

print(words)
