from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io, base64, torch

model = YOLO('car_damage_best.pt')

app = Flask(__name__)

@app.route('/damage', methods=['POST'])
def damage():
    try:
        data = request.json
        img_bytes = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        results = model(img)
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
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
    app.run(host='0.0.0.0', port=5001, debug=False)