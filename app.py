import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Waste info dictionary 
waste_info = {
    "plastic": {
        "disposal": "Recycle at plastic collection centers.",
        "co2": 6.0,
        "biodegradable": "No",
        "decomposition_time": "500+ Years",
        "hazard": "Medium",
        "tip": "Avoid single-use plastics. Switch to reusable containers.",
        "category": "Dry Waste",
        "reusable": "Yes (Depends on item)",
        "recycling_rate": "≈9% Globally",
        "more_info": "https://www.epa.gov/plastics"
    },
    "glass": {
        "disposal": "Rinse and recycle in glass bins.",
        "co2": 1.3,
        "biodegradable": "No",
        "decomposition_time": "1 Million Years",
        "hazard": "Low",
        "tip": "Reuse glass jars; recycle to reduce raw material extraction.",
        "category": "Dry Waste",
        "reusable": "Yes",
        "recycling_rate": "≈27% Globally",
        "more_info": "https://www.epa.gov/smm/sustainable-management-glass"
    },
    "metal": {
        "disposal": "Recycle at metal collection points.",
        "co2": 8.0,
        "biodegradable": "No",
        "decomposition_time": "50–500 Years",
        "hazard": "Medium",
        "tip": "Crush cans to save space before recycling.",
        "category": "Dry Waste",
        "reusable": "Yes",
        "recycling_rate": "≈30% Globally",
        "more_info": "https://www.epa.gov/smm/sustainable-management-metal"
    },
    "paper": {
        "disposal": "Recycle clean paper. Compost if uncoated.",
        "co2": 0.9,
        "biodegradable": "Yes",
        "decomposition_time": "2–6 Weeks",
        "hazard": "Low",
        "tip": "Avoid glossy/multi-layered paper — hard to recycle.",
        "category": "Dry Waste",
        "reusable": "Yes (If not torn)",
        "recycling_rate": "≈68% (Highest among waste types)",
        "more_info": "https://www.epa.gov/smm/sustainable-management-paper"
    },
    "cardboard": {
        "disposal": "Recycle if clean and dry.",
        "co2": 1.1,
        "biodegradable": "Yes",
        "decomposition_time": "2 Months",
        "hazard": "Low",
        "tip": "Flatten boxes before recycling to save space.",
        "category": "Dry Waste",
        "reusable": "Yes (For packaging, crafts)",
        "recycling_rate": "≈89% In developed countries",
        "more_info": "https://earth911.com/recycling-guide/how-to-recycle-cardboard/"
    },
    "trash": {
        "disposal": "Dispose as general waste (landfill).",
        "co2": 2.5,
        "biodegradable": "Mixed",
        "decomposition_time": "Varies By Contents",
        "hazard": "Medium",
        "tip": "Reduce mixed trash by separating waste types.",
        "category": "General Waste",
        "reusable": "No",
        "recycling_rate": "Not Recyclable",
        "more_info": "https://www.epa.gov/recycle"
    },
    "battery": {
        "disposal": "Drop off at hazardous/e-waste facility.",
        "co2": 12.0,
        "biodegradable": "No",
        "decomposition_time": "100+ Years",
        "hazard": "High",
        "tip": "Never throw batteries in regular trash.",
        "category": "Hazardous Waste",
        "reusable": "No",
        "recycling_rate": "<5% Globally",
        "more_info": "https://www.epa.gov/recycle/used-household-batteries"
    },
    "clothes": {
        "disposal": "Donate, reuse, or recycle via textile centers.",
        "co2": 3.5,
        "biodegradable": "Sometimes (Natural fibers)",
        "decomposition_time": "5 Months – 40 Years",
        "hazard": "Low",
        "tip": "Buy secondhand or use clothes for rags.",
        "category": "Dry Waste / Textile",
        "reusable": "Yes",
        "recycling_rate": "≈15% Globally",
        "more_info": "https://www.epa.gov/facts-and-figures-about-materials-textiles"
    },
    "shoes": {
        "disposal": "Donate if wearable or recycle via shoe programs.",
        "co2": 4.0,
        "biodegradable": "No",
        "decomposition_time": "30–40 Years",
        "hazard": "Low",
        "tip": "Use shoe recycling programs (e.g., Nike Grind).",
        "category": "Dry Waste / Textile",
        "reusable": "Yes",
        "recycling_rate": "Very Low",
        "more_info": "https://www.nike.com/help/a/recycle-shoes"
    },
    "biological": {
        "disposal": "Compost or dispose in green waste bin.",
        "co2": 1.0,
        "biodegradable": "Yes",
        "decomposition_time": "2–8 Weeks",
        "hazard": "Low",
        "tip": "Use food scraps for home composting.",
        "category": "Wet Waste / Compostable",
        "reusable": "Compostable",
        "recycling_rate": "Composting ≈35% In Some Regions",
        "more_info": "https://www.epa.gov/recycle/composting-home"
    }
}


class_names = list(waste_info.keys())

# Load the model
model = tf.keras.models.load_model("mobilenetv2_waste_classifier_final.h5")

# Prediction function
def classify_waste(img):
    if img is None:
        return "Please upload an image.", None

    image = img.resize((224, 224)).convert("RGB")
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_class = class_names[pred_idx]
    confidence = preds[0][pred_idx] * 100

    info = waste_info.get(pred_class, {})
    result = f"### 🧠 Prediction: **{pred_class.capitalize()}** ({confidence:.2f}%)\n\n"
    result += f"**Disposal**: {info.get('disposal', 'N/A')}\n"
    result += f"**CO₂ Footprint**: {info.get('co2')} kg/kg\n"
    result += f"**Biodegradable**: {info.get('biodegradable')}\n"
    result += f"**Decomposition Time**: {info.get('decomposition_time')}\n"
    result += f"**Hazard Level**: {info.get('hazard')}\n"
    result += f"**Category**: {info.get('category')}\n"
    result += f"**Reusable**: {info.get('reusable')}\n"
    result += f"**Recycling Rate**: {info.get('recycling_rate')}\n"
    result += f"**Tip**: {info.get('tip')}\n\n"
    result += f"[🌍 More Info]({info.get('more_info')})"

    return result, img

# Gradio Interface
with gr.Blocks(theme=gr.themes.Base(), css="body { font-family: 'Segoe UI', sans-serif; }") as demo:
    gr.Markdown("# ♻️ Smart Waste Classifier")
    gr.Markdown(
        "Take or upload a photo of waste material to classify it and get safe disposal tips. "
        "Help reduce pollution and improve recycling habits!"
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                source="upload", 
                type="pil", 
                label="📷 Upload or Take a Photo", 
                tool="editor"
            )
            classify_btn = gr.Button("🔍 Classify Waste")
        with gr.Column(scale=1):
            image_output = gr.Image(label="🔎 Preview", show_label=True)
            text_output = gr.Markdown()

    classify_btn.click(fn=classify_waste, inputs=img_input, outputs=[text_output, image_output])

    gr.Markdown("---")
    gr.Markdown("👨‍💻 Built with MobileNetV2 • Trained on Kaggle Dataset • Accuracy: **93%**")

# Run
demo.launch()
