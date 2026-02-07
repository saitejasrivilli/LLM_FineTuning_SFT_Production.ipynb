import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

st.set_page_config(page_title="Qwen-7B Fine-tuned", page_icon="🤖", layout="wide")

@st.cache_resource
def load_model():
    pipe = pipeline(
        "text-generation",
        model="SaiTejaSrivilli/qwen-7b-sft-merged",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipe

st.title("🚀 Qwen-7B Fine-tuned Assistant")
st.markdown("Fine-tuned with SFT+LoRA | **0.855 BERTScore** | **17% loss reduction**")
st.markdown("---")

with st.spinner("Loading model..."):
    pipe = load_model()

col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.text_area("💬 Your Question", placeholder="Ask me anything...", height=100)

with col2:
    max_tokens = st.slider("📏 Max Length", 50, 500, 200)
    temperature = st.slider("🌡️ Temperature", 0.1, 1.5, 0.7)

if st.button("✨ Generate Response", type="primary"):
    if prompt.strip():
        with st.spinner("Generating..."):
            result = pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
            response = result[0]['generated_text'][len(prompt):].strip()
        
        st.markdown("### 🤖 Response:")
        st.success(response)
    else:
        st.warning("Please enter a question")

st.markdown("---")
st.markdown("### 📝 Example Prompts:")

examples = [
    "Explain quantum computing in simple terms",
    "Write a Python function to calculate fibonacci",
    "What are the main causes of climate change?"
]

cols = st.columns(3)
for i, example in enumerate(examples):
    if cols[i].button(f"Try: {example[:30]}..."):
        st.rerun()

st.markdown("---")
st.markdown("🔗 [Model](https://huggingface.co/SaiTejaSrivilli/qwen-7b-sft-merged) | 💼 [LinkedIn](https://linkedin.com/in/saitejasrivilli) | 🐙 [GitHub](https://github.com/saitejasrivilli)")
