import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Qwen-7B Fine-tuned", page_icon="🤖", layout="wide")

st.title("🚀 Qwen-7B Fine-tuned Assistant")
st.markdown("Fine-tuned with SFT+LoRA | **0.855 BERTScore** | **17% loss reduction**")
st.markdown("---")

client = InferenceClient(model="SaiTejaSrivilli/qwen-7b-sft-merged")

col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.text_area("💬 Your Question", placeholder="Ask me anything about science, tech, programming...", height=120)

with col2:
    max_tokens = st.slider("📏 Max Length", 50, 500, 200)
    temperature = st.slider("🌡️ Temperature", 0.1, 1.5, 0.7, step=0.1)

if st.button("✨ Generate Response", type="primary", use_container_width=True):
    if prompt.strip():
        with st.spinner("🤖 Thinking..."):
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
                st.error(f"Error: {e}")
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
st.markdown("🔗 [Model](https://huggingface.co/SaiTejaSrivilli/qwen-7b-sft-merged) | [GitHub](https://github.com/saitejasrivilli) | [LinkedIn](https://linkedin.com/in/saitejasrivilli)")
```
```
# requirements.txt
huggingface_hub>=0.20.0
streamlit>=1.30.0
