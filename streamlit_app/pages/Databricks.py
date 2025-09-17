import streamlit as st
import constants as c

st.set_page_config(
    page_title="Databricks",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="collapsed"
)


margin_r, body, margin_l = st.columns([0.4, 3, 0.4])


with body:
    c.menu()

    with st.sidebar:
        st.success("Select a page above.")

    st.header("MLOps workflows on Databricks", divider='rainbow')

    col1, col2, col3 = st.columns([2, 0.2, 1.5])

    with col1:
        st.markdown(f"###### {c.info['processes']}")
        st.markdown(f"###### 🚀 DevOps: {c.databricks_facts['devops']}")
        st.markdown(f"###### 📚 DataOps: {c.databricks_facts['dataops']}")
        st.markdown(f"###### 🧠 ModelOps: {c.databricks_facts['modelops']}")

    with col3:
        st.image("images/mlops_processes.png", width=360)

    st.subheader("✅ Best Practices", divider='rainbow')
    st.markdown(f"{c.databricks_facts['best_practices']}")
