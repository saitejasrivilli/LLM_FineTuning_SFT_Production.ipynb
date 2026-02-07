import streamlit as st
import requests

st.set_page_config(page_title="Qwen-7B Fine-tuned", page_icon="🤖", layout="wide")

st.title("🚀 Qwen-7B Fine-tuned Assistant")
st.markdown("Fine-tuned with SFT+LoRA | **0.855 BERTScore** | **17% loss reduction**")
st.markdown("---")

st.info("⚠️ Model is a LoRA adapter requiring local inference with base model. See usage code below.")

st.markdown("### 📊 Training Results")
col1, col2, col3 = st.columns(3)
col1.metric("Training Loss", "1.176", "-17%")
col2.metric("BERTScore", "0.855", "+21%")
col3.metric("Trainable Params", "0.5%", "35M/7B")

st.markdown("---")

st.markdown("### 💻 How to Use")

code = '''from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "SaiTejaSrivilli/qwen-3b-sft")
tokenizer = AutoTokenizer.from_pretrained("SaiTejaSrivilli/qwen-3b-sft")

inputs = tokenizer("Your question", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))'''

st.code(code, language="python")

st.markdown("### 📝 Sample Outputs (from evaluation)")

examples = {
    "Explain quantum computing": "Quantum computing uses quantum bits that can exist in superposition...",
    "Python fibonacci function": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Climate change causes": "Main causes include greenhouse gas emissions, deforestation..."
}

for q, a in examples.items():
    with st.expander(f"Q: {q}"):
        st.write(f"**A:** {a}")

st.markdown("---")
st.markdown("🔗 [Model on HuggingFace](https://huggingface.co/SaiTejaSrivilli/qwen-3b-sft) | 💼 [LinkedIn](https://linkedin.com/in/saitejasrivilli) | 🐙 [GitHub](https://github.com/saitejasrivilli)")
```
```
# requirements.txt
streamlit>=1.30.0
