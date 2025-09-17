import streamlit as st
import constants as c


st.set_page_config(
    page_title="MLOps",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

margin_r, body, margin_l = st.columns([0.4, 3, 0.4])


with body:
    c.menu()

    with st.sidebar:
        st.success("Select a page above.")

    st.header("🔄 What is MLOps?", divider='rainbow')

    col1, col2, col3 = st.columns([2, 0.2, 1.5])

    with col1:
        st.write(c.info['brief'])
        st.markdown("📊 Data + 🧠 ML + ⚙️ DevOps = 🚀 MLOps")

    with col3:
        st.image("images/mlops_cycle.png", width=360)

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("🧑‍💻 Who is it for?", divider='rainbow')
    st.markdown(f"""{c.info['users']}""")

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("🎯 Why is it relevant?", divider='rainbow')
    st.markdown(
        f"""
        {c.info['relevance']}
        """
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("♾️ MLOps Tools", divider='rainbow')

    st.markdown(
        f"""
        - **MLflow**: {c.info['mlflow']}
        - **Kubeflow**: {c.info['kubeflow']}
        - **Airflow**: {c.info['airflow']}
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    show_real_benefit = st.toggle("Show Real Benefit", value=False)

    if show_real_benefit:
        st.subheader("📈 :blue[REAL benefits]", divider='rainbow')
        current_benefit = 'real_benefit'
    else:
        st.subheader("📈 :blue[Benefits]", divider='rainbow')
        current_benefit = 'benefit'

    def benefit_tab(benefits):
        rows = len(benefits)//c.benefit_col_size
        benefits_iter = iter(benefits)
        if len(benefits) % c.benefit_col_size != 0:
            rows += 1
        for x in range(rows):
            columns = st.columns(c.benefit_col_size)
            for index_ in range(c.benefit_col_size):
                try:
                    columns[index_].button(next(benefits_iter))
                except StopIteration:
                    break

    with st.spinner(text="Loading section..."):
        if current_benefit == 'benefit':
            benefit_tab(c.info['benefit'])
        else:
            benefit_tab(c.info['real_benefit'])
