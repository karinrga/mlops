import streamlit as st

benefit_col_size = 5


def menu():
    bar0, bar1, bar2, bar3 = st.columns([0.1, 1, 1, 1])
    bar1.page_link("Home.py", label="MLOps", icon="🔄")
    bar2.page_link("pages/Databricks.py", label="Databricks", icon="🧱")
    bar3.page_link("pages/MLflow.py", label="MLflow", icon="⚙️")
    st.write("")


info = {'brief':
        """
          MLOps is a set of processes and automated steps for managing code,
          data, and models to improve performance, stability,
          and long-term efficiency of ML systems.
        """,
        "processes": "It combines DevOps, DataOps, and ModelOps.",
        'experiment_tracking': "MLflow",
        'orchestration': "Airflow, Kubeflow",
        'platform': "Vertex AI, Databricks",
        'users': """
        - Data Scientists
        - ML Engineers
        - Data Engineers
        """,
        "relevance":
        """
        - Accelerate ML model deployment
        - Improve collaboration between teams
        - Ensure model reliability and scalability
        - Enhance monitoring and governance of ML models
        """,
        'mlflow': "Open-source platform to manage the ML lifecycle.",
        'kubeflow': "Machine learning toolkit for Kubernetes to deploy, "
        "orchestrate, and manage ML workflows.",
        'airflow': "Platform to programmatically author, schedule, and monitor"
        "workflows.",
        'benefit': ['Efficiency',
                    'Scalability',
                    'Risk Reduction',
                    ],
        'real_benefit': ['Drink Coffee',
                         'Take Naps',
                         'Listen to Podcasts',]
        }

databricks_facts = {
    'devops': "Access control and versioning",
    'dataops': "Store data in a lakehouse architecture using Delta tables",
    'modelops': "Use MLflow to track experiments, models, and deployments",
}

mlflow_facts = {
    'definition': "an open-source platform to manage the ML lifecycle.",
    'tracking': "Allows you to track experiments to record and compare "
    "parameters and results.",
    'model': "Allow you to manage and deploy models from various ML libraries"
    "to various model serving and inference platforms.",
    'model_registry': "Allows you to manage the model deployment process "
    "from staging to production, with model versioning and annotation.",
    'ai_agent': "Allows you to develop high-quality AI"
    " agents by helping you compare, evaluate, and troubleshoot agents."
}
