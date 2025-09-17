import streamlit as st
import constants as c

st.set_page_config(
    page_title="MLops",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


margin_r, body, margin_l = st.columns([0.4, 3, 0.4])


with body:
    c.menu()

    with st.sidebar:
        st.success("Select a page above.")

    st.header("🔢 Primary Components", divider='rainbow')
    col1, col2, col3 = st.columns([1.3, 0.2, 1])
    with col1:
        st.markdown(f"✅ Tracking: {c.mlflow_facts['tracking']}")
        st.markdown(f"🧠 Model: {c.mlflow_facts['model']}")
        st.markdown(f"📦 Model Registry: {c.mlflow_facts['model_registry']}")
        st.markdown(
            f"🕵️ AI Agent Evaluation and Tracing: {c.mlflow_facts['ai_agent']}"
        )

    with col3:
        st.image("images/mlflow.png", width=360)

    st.subheader("🚀 Traditional ML and Deep Learning", divider='rainbow')
    st.markdown("###### Native MLflow ML Library Integrations")
    col1, col2, col3 = st.columns([1.3, 0.2, 1])
    with col1:
        st.image("images/scikit.png")
        st.image("images/xgboost.png")
        st.image("images/tensorflow.png")
    with col3:
        st.image("images/pytorch.png")
        st.image("images/keras.png", width=80)
        st.image("images/spark.png")
