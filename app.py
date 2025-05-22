from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Use a tiny model to fit within Render's free memory limits
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

# Set pad token to eos token to avoid attention mask warnings
tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT = (
    "You are Hikari, a supportive, empathetic mental health companion. "
    "Respond with warmth, validation, and practical mindfulness suggestions. "
    "The user says: "
)

@app.route("/", methods=["GET"])
def index():
    return "Hikari chatbot server is running. Use POST /predict to chat."

@app.route("/predict", methods=["POST"])
def predict():
    print("Received POST to /predict")  # Debug
    user_input = request.json.get("message", "")
    print("User input:", user_input)  # Debug
    prompt = SYSTEM_PROMPT + user_input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    print("Tokenized input")  # Debug
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,  # Keep this low to reduce memory and response time
        pad_token_id=tokenizer.eos_token_id
    )
    print("Model generated output")  # Debug
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(SYSTEM_PROMPT):].strip()
    print("Decoded response:", response)  # Debug
    return jsonify({"response": response})

# No app.run() needed; Gunicorn will serve the app
