import streamlit as st
from huggingface_hub import InferenceClient
import time

st.set_page_config(page_title="Qwen-7B Fine-tuned", page_icon="🤖", layout="wide")

st.title("🚀 Qwen-7B Fine-tuned Assistant")
st.markdown("Fine-tuned with SFT+LoRA | **0.855 BERTScore** | **17% loss reduction**")
st.markdown("---")

client = InferenceClient(model="SaiTejaSrivilli/qwen-7b-sft-merged")

col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.text_area("💬 Your Question", placeholder="Ask me anything...", height=120)

with col2:
    max_tokens = st.slider("📏 Max Length", 50, 500, 200)
    temperature = st.slider("🌡️ Temperature", 0.1, 1.5, 0.7, step=0.1)

if st.button("✨ Generate Response", type="primary", use_container_width=True):
    if prompt.strip():
        with st.spinner("🤖 Generating..."):
            try:
                response = client.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9
                )
                st.markdown("### 🤖 Response:")
                st.success(response)
            except Exception as e:
                error_msg = str(e)
                if "loading" in error_msg.lower() or "not found" in error_msg.lower():
                    st.warning("⏳ Model is still loading (takes 5-10 min after upload). Please try again in a few minutes!")
                    st.info("Check model status: https://huggingface.co/SaiTejaSrivilli/qwen-7b-sft-merged")
                else:
                    st.error(f"Error: {error_msg}")
    else:
        st.warning("Please enter a question")

st.markdown("---")
st.markdown("### 📝 Try These Examples:")

examples = [
    "Explain quantum computing in simple terms",
    "Write a Python function to calculate fibonacci numbers",
    "What are the main causes of climate change?",
]

for ex in examples:
    if st.button(ex, key=ex):
        st.rerun()

st.markdown("---")

# Show metrics while waiting
st.markdown("### 📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("BERTScore", "0.855", "+21%")
col2.metric("Training Loss", "1.176", "-17%")
col3.metric("Trainable Params", "0.5%", "35M/7B")

st.markdown("---")
st.markdown("🔗 [Model](https://huggingface.co/SaiTejaSrivilli/qwen-7b-sft-merged) | [GitHub](https://github.com/saitejasrivilli) | [LinkedIn](https://linkedin.com/in/saitejasrivilli)")
