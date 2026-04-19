from flask import Flask, request, jsonify
from PIL import Image
import torch, torchvision, io, base64
from torchvision import transforms
from torch import nn

# Model load
model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 2)
model.load_state_dict(torch.load('gan_detector.pt', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    print("API running on port 5000!")