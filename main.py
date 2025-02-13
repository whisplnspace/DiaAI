import numpy as np
import pickle
import gradio as gr

# Load the trained model and scaler
try:
    with open('trained_model.sav', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('scaler.sav', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")


# Function to make predictions
def diabetes_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                        Age):
    try:
        input_data = np.array(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age],
            dtype=float)
        input_data_reshaped = input_data.reshape(1, -1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = loaded_model.predict(std_data)

        if prediction[0] == 1:
            advice = (
                "⚠️ Health Tips for Managing Diabetes:\n"
                "- 🍏 Eat a balanced diet (low sugar, high fiber, and whole foods).\n"
                "- 🚶 Stay physically active (walking, yoga, strength training, etc.).\n"
                "- 🔬 Monitor blood sugar levels regularly.\n"
                "- 💧 Drink plenty of water and avoid processed foods.\n"
                "- 🩺 Follow medical advice and take prescribed medications."
            )
            return "🩸 The person is diabetic. Please follow a healthy lifestyle and consult a doctor.\n\n" + advice
        else:
            return "✅ The person is not diabetic. Keep maintaining a healthy routine and spread awareness for a healthier world! 💙"
    except Exception as e:
        return f"⚠️ Error in prediction: {e}"


# Function to reset inputs
def reset_fields():
    return 2, 120, 80, 20, 85, 25.6, 0.5, 35, ""


# Gradio UI with an aesthetic layout
with gr.Blocks() as demo:
    gr.Markdown("## 🩺 **DiaGuard: AI-Powered Diabetes Prediction**")
    gr.Markdown(
        "🔬 **Empowering health with AI** — This tool helps predict diabetes risk with accuracy and care. Enter your details below to get an AI-powered assessment. 🏥")

    with gr.Row():
        gr.Image("https://t3.ftcdn.net/jpg/09/60/70/62/360_F_960706299_kKIBFBX4TAaGiTlgUNTNurWUs1Nm5WBO.jpg", width=100)

    gr.Markdown("### 📝 **Fill in the details for analysis**")

    with gr.Row():
        with gr.Column():
            Pregnancies = gr.Number(label="👶 Pregnancies", value=2)
            Glucose = gr.Number(label="🩸 Glucose Level (mg/dL)", value=120)
            BloodPressure = gr.Number(label="💓 Blood Pressure (mmHg)", value=80)
            SkinThickness = gr.Number(label="📏 Skin Thickness (mm)", value=20)
        with gr.Column():
            Insulin = gr.Number(label="💉 Insulin Level (IU/mL)", value=85)
            BMI = gr.Number(label="⚖️ BMI (kg/m²)", value=25.6)
            DiabetesPedigreeFunction = gr.Number(label="🧬 Diabetes Pedigree Function", value=0.5)
            Age = gr.Number(label="🎂 Age", value=35)

    output = gr.Textbox(label="🩺 Prediction Result", interactive=False)

    with gr.Row():
        predict_btn = gr.Button("🔍 Predict", variant="primary")
        clear_btn = gr.Button("🔄 Clear")

    predict_btn.click(diabetes_prediction,
                      inputs=[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                              DiabetesPedigreeFunction, Age],
                      outputs=output)

    clear_btn.click(reset_fields,
                    inputs=[],
                    outputs=[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                             Age, output])

    # Closing message
    gr.Markdown("""
        ❤️ **Made with love and for the betterment of humanity.**  
        Stay informed, stay healthy, and take care of each other! 🌍💙
    """)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
