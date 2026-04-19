from flask import Flask, request, jsonify
from PIL import Image, ImageFile
import torch, torchvision, io, base64
from torchvision import transforms
from torch import nn
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

# ========== GAN DETECTOR ==========
gan_model = torchvision.models.efficientnet_b0(weights=None)
gan_model.classifier[1] = nn.Linear(1280, 2)
gan_model.load_state_dict(torch.load('gan_detector.pt', map_location='cpu'))
gan_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = gan_model(tensor)
            probs = torch.softmax(output, dim=1)[0]
        fake_score = float(probs[0])
        real_score = float(probs[1])
        is_ai = fake_score > 0.5
        return jsonify({
            "success": True,
            "is_ai_generated": is_ai,
            "ai_score": round(fake_score * 100, 2),
            "human_score": round(real_score * 100, 2),
            "verdict": "REJECT" if is_ai else "PASS"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ========== DAMAGE DETECTOR ==========
damage_model = YOLO('car_damage_best.pt')

@app.route('/damage', methods=['POST'])
def damage():
    try:
        data = request.json
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        results = damage_model(img)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": damage_model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 2),
                    "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
                })
        damage_found = len(detections) > 0
        return jsonify({
            "success": True,
            "damage_found": damage_found,
            "total_damages": len(detections),
            "detections": detections,
            "verdict": "CLAIM_VALID" if damage_found else "NO_DAMAGE"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)