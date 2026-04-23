This Markdown file outlines the technical logic and structure for your **Fit-Aware Virtual Try-On** prototype. It uses **IDM-VTON** as the generative engine because it supports text-prompt conditioning, allowing us to "force" a fit (tight/loose) based on numerical calculations.

---

# Project: Fit-Aware VTO (IDM-VTON + Logic Layer)

## 1. Overview
This system bridges the gap between image-based try-ons and numerical measurements. It calculates the difference between user body dimensions and garment specifications to generate a **"Fit Prompt"** that guides the AI generation.

## 2. Technical Stack
*   **Frontend:** Streamlit
*   **Engine:** IDM-VTON (Inference via API or Local GPU)
*   **Logic:** Python-based Dimension Comparison
*   **Data:** Pre-defined Garment Catalog (JSON)

---

## 3. Garment Database (Sample Data)
The system stores 10 items with "Target Dimensions" (measured in inches).

| ID | Gender | Item Name | Type | Size | Chest/Bust | Waist |
|:---|:---|:---|:---|:---|:---|:---|
| M1 | Men | Slim Fit White Shirt | Top | M | 38" | 32" |
| M2 | Men | Heavy Oversized Hoodie | Top | L | 46" | 40" |
| M3 | Men | Casual Linen Shirt | Top | L | 42" | 36" |
| M4 | Men | Compression Gym Tee | Top | S | 34" | 30" |
| M5 | Men | Standard Denim Jacket | Top | M | 40" | 34" |
| W1 | Women | Silk Evening Blouse | Top | S | 32" | 26" |
| W2 | Women | Boxy Graphic Tee | Top | M | 40" | 38" |
| W3 | Women | Tailored Blazer | Top | M | 36" | 30" |
| W4 | Women | Summer Crop Top | Top | XS | 30" | 24" |
| W5 | Women | Relaxed Flannel | Top | L | 42" | 40" |

---

## 4. The "Fit Logic" Algorithm
We calculate a **Delta ($\Delta$)** between the Garment and the Person.

**Formula:** $\Delta = Garment\_Dimension - User\_Dimension$

| Delta Range | Fit Category | Prompt Modifier |
|:---|:---|:---|
| $\Delta < -1"$ | **Undersized** | "Extremely tight fit, stretched fabric, bodycon style" |
| $-1" \le \Delta \le 1"$ | **True to Size** | "Perfectly tailored fit, regular silhouette" |
| $1" < \Delta \le 4"$ | **Relaxed** | "Slightly loose, comfortable casual fit" |
| $\Delta > 4"$ | **Oversized** | "Very baggy, oversized aesthetic, dropped shoulders" |

---

## 5. Implementation (Streamlit Code)

```python
import streamlit as st
from PIL import Image
import pandas as pd

# 1. Mock Garment Database
GARMENTS = [
    {"id": "M1", "gender": "Men", "name": "Slim Fit White Shirt", "chest": 38, "img": "m1.jpg"},
    {"id": "M2", "gender": "Men", "name": "Heavy Oversized Hoodie", "chest": 46, "img": "m2.jpg"},
    # ... add all 10 garments here
]

def calculate_fit_prompt(user_chest, garment_chest):
    delta = garment_chest - user_chest
    if delta < -1:
        return "tight, body-hugging, stretched fit"
    elif -1 <= delta <= 2:
        return "perfect fit, tailored look"
    elif 2 < delta <= 5:
        return "relaxed, loose fit"
    else:
        return "very oversized, baggy, street-style fit"

# --- UI Layout ---
st.title("📏 Fit-Aware Virtual Try-On")
st.sidebar.header("User Dimensions")

user_gender = st.sidebar.selectbox("Gender", ["Men", "Women"])
user_chest = st.sidebar.number_input("Your Chest Circumference (inches)", value=38)
person_img = st.sidebar.file_uploader("Upload Your Photo", type=['jpg', 'png'])

st.header("Step 1: Select a Garment")
# Filter garments by gender
filtered_items = [g for g in GARMENTS if g['gender'] == user_gender]
selected_garment_name = st.selectbox("Choose a piece of clothing", [g['name'] for g in filtered_items])
selected_garment = next(item for item in filtered_items if item["name"] == selected_garment_name)

if st.button("Generate Try-On"):
    if person_img is not None:
        # 2. Run Logic
        fit_description = calculate_fit_prompt(user_chest, selected_garment['chest'])
        
        # 3. Construct Final Prompt for IDM-VTON
        final_prompt = f"A person wearing a {selected_garment_name}, {fit_description}, high quality, realistic."
        
        st.info(f"**System Logic:** {fit_description}")
        st.write(f"**AI Prompt:** {final_prompt}")
        
        # 4. API Call to IDM-VTON (Placeholder)
        with st.spinner("AI is tailoring the clothes to your measurements..."):
            # result = call_idm_vton_api(person_img, selected_garment['img'], final_prompt)
            st.warning("Note: Connect your IDM-VTON API/Model here to see final image.")
            
            # Displaying Mock Result
            st.image("https://via.placeholder.com/500x700.png?text=Generated+Try-On+Result", caption="Virtual Result")
    else:
        st.error("Please upload a person image first.")
```

---

## 6. Why this works for your project
1.  **Solves "Blind Fit":** Standard VTO models just "paste" the clothes. This logic ensures that if a user wears a size too small, the AI is explicitly told to render it "tight."
2.  **User Trust:** The frontend shows the user that the system is actually considering their measurements, which adds a "fitting room" feel.
3.  **Efficiency:** Instead of training a new model (which takes months), you are using **Prompt Engineering** to control an existing powerful model (IDM-VTON).

## 7. Next Steps for Implementation
1.  **Image Prep:** Create the 10 garment images (clear background).
2.  **Model Setup:** Deploy IDM-VTON on a GPU (e.g., via Hugging Face Inference Endpoints or a local ComfyUI/Gradio backend).
3.  **Refinement:** Add more dimensions (Waist, Arm Length) to the logic to handle trousers and long-sleeved items.