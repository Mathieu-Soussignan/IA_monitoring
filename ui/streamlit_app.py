"""
Streamlit MLOps control panel – Day 3
"""

import os, json, requests
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth

# ──────────────────────────────────────────────────────────────────────────────
# 1. Page & theme
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLOps Control Panel",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="expanded",
)
PRIMARY_COLOR = "#00c0ff"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Credentials file & authentication
# ──────────────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).with_name("_cred.json")

if not CFG_PATH.exists():
    user = os.getenv("UI_USER", "admin")
    pwd  = os.getenv("UI_PWD",  "admin")
    hashed = stauth.Hasher([pwd]).generate()[0]
    creds = {
        "usernames": {
            user: {
                "email": f"{user}@example.com",
                "name": "Admin",
                "password": hashed,
            }
        }
    }
    CFG_PATH.write_text(json.dumps(creds))
else:
    creds = json.loads(CFG_PATH.read_text())

auth = stauth.Authenticate(
    creds,
    "mlops_ui",
    "abcdef",
    cookie_expiry_days=1,
)
name, auth_status, _ = auth.login(location="main")
if not auth_status:
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Globals & sidebar
# ──────────────────────────────────────────────────────────────────────────────
API_ML = st.sidebar.text_input(
    "ML-API URL", value=os.getenv("ML_API_URL", "http://ml-api:8001")
)
API_FAST = os.getenv("FAST_API_URL", "http://fastapi-app:8000")
API_PREFECT = st.sidebar.text_input(
    "Prefect API URL", value=os.getenv("PREFECT_API_URL", "http://prefect:4200/api")
)
API_TOKEN = st.sidebar.text_input("Bearer token", value=os.getenv("API_TOKEN", ""))
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

# ──────────────────────────────────────────────────────────────────────────────
# 4. Helper
# ──────────────────────────────────────────────────────────────────────────────
def call_api(method: str, url: str, **kwargs):
    try:
        r = requests.request(method, url, headers=HEADERS, timeout=20, **kwargs)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.ok, r.json()
        return r.ok, r.text
    except Exception as exc:
        return False, str(exc)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Layout (tabs)
# ──────────────────────────────────────────────────────────────────────────────
tab_gen, tab_pred, tab_train, tab_health = st.tabs(
    ["Generate", "Predict", "Retrain", "Health"]
)

with tab_gen:
    st.subheader("🧬 Generate synthetic dataset")
    n = st.number_input("n_samples", 1000, 20000, 1500, step=500)
    if st.button("Generate"):
        ok, data = call_api("POST", f"{API_ML}/generate", params={"n_samples": n})
        st.json(data if ok else {"error": data})
        if ok:
            st.balloons()

with tab_pred:
    st.subheader("🔮 Predict on last sample")
    if st.button("Predict"):
        ok, data = call_api("GET", f"{API_ML}/predict")
        if ok:
            st.metric("Prediction", data.get("prediction"))
            st.metric("Confidence", f"{data.get('confidence', 0):.2%}")
            st.snow()
        else:
            st.error(data)

with tab_train:
    st.subheader("♻️ Retrain model (direct ML-API)")
    if st.button("Retrain"):
        ok, data = call_api("POST", f"{API_ML}/retrain")
        st.json(data if ok else {"error": data})

    st.subheader("📦 Trigger Prefect flow")
    if st.button("Trigger Prefect flow"):
        endpoint = f"{API_PREFECT}/deployments/name/continuous-retrain/default/create_flow_run"
        ok, data = call_api("POST", endpoint)
        if ok:
            st.success("✅ Flow triggered!")
            st.json(data)
        else:
            st.error(f"❌ Failed to trigger flow: {data}")

with tab_health:
    st.subheader("🚦 Service health checks")
    col1, col2 = st.columns(2)
    ok, _ = call_api("GET", f"{API_FAST}/health")
    col1.metric("FastAPI", "UP" if ok else "DOWN")
    ok, _ = call_api("GET", f"{API_ML}/health")
    col2.metric("ML-API", "UP" if ok else "DOWN")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Tiny theme override
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    :root {{ --primary: {PRIMARY_COLOR}; }}
    </style>
    """,
    unsafe_allow_html=True,
)