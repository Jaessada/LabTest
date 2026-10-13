from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "return_classifier.joblib"

st.set_page_config(
	page_title="Return Risk | Order Check",
	page_icon="R",
	layout="centered",
)

st.markdown(
	"""
	<style>
	.stApp {
		background: #f4f7f5;
		color: #18332c;
	}
	.block-container {
		max-width: 880px;
		padding-top: 2.5rem;
		padding-bottom: 3rem;
	}
	h1, h2, h3, p, label, [data-testid="stMetricValue"] {
		font-family: "Sarabun", "Noto Sans Thai", sans-serif;
	}
	div[data-testid="stForm"] {
		border: 1px solid #d8e3dc;
		border-radius: 8px;
		background: #ffffff;
		padding: 1.25rem 1.4rem;
	}
	div.stButton > button[kind="primary"] {
		background: #176b57;
		border: 0;
		color: white;
		min-height: 2.8rem;
		font-weight: 650;
	}
	div.stButton > button[kind="primary"]:hover {
		background: #105443;
		border: 0;
		color: white;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(model_path: str) -> dict:
	return joblib.load(model_path)


st.title("ตรวจความเสี่ยงการคืนสินค้า")
st.caption("กรอกข้อมูลคำสั่งซื้อเพื่อประเมินโอกาสการคืนสินค้าด้วยโมเดลที่ฝึกไว้")

if not MODEL_PATH.is_file():
	st.error(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}")
	st.stop()

try:
	model_bundle = load_model(str(MODEL_PATH))
except Exception as error:
	st.error(f"โหลดโมเดลไม่สำเร็จ: {error}")
	st.stop()

st.subheader("ข้อมูลคำสั่งซื้อ")
with st.form("order_prediction"):
	left_column, right_column = st.columns(2)

	with left_column:
		price = st.number_input(
			"ราคา (บาท)", min_value=1.0, max_value=100000.0, value=1295.0, step=100.0
		)
		delivery_days = st.number_input(
			"ระยะเวลาจัดส่ง (วัน)", min_value=1, max_value=365, value=7, step=1
		)
		rating = st.selectbox("คะแนนรีวิว", options=[1, 2, 3, 4, 5], index=2)

	with right_column:
		discount_percent = st.number_input(
			"ส่วนลด (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0
		)
		category = st.selectbox(
			"ประเภทสินค้า", options=["Fashion", "Home", "Electronics"]
		)
		repeat_customer = st.selectbox("ลูกค้าซื้อซ้ำ", options=["No", "Yes"])

	submitted = st.form_submit_button(
		"ประเมินการคืนสินค้า", type="primary", use_container_width=True
	)

if submitted:
	order = pd.DataFrame(
		[{
			"price": price,
			"delivery_days": delivery_days,
			"rating": rating,
			"discount_percent": discount_percent,
			"category": category,
			"repeat_customer": repeat_customer,
		}]
	)
	order = order[model_bundle["feature_columns"]]

	transformed_order = model_bundle["preprocessor"].transform(order)
	selected_order = model_bundle["feature_selector"].transform(transformed_order)
	model = model_bundle["model"]
	prediction = int(model.predict(selected_order)[0])
	probabilities = model.predict_proba(selected_order)[0]
	returned_index = list(model.classes_).index(1)
	return_probability = float(probabilities[returned_index])

	st.divider()
	st.subheader("ผลการประเมิน")
	result_label = model_bundle["class_labels"].get(
		prediction, "คืนสินค้า" if prediction == 1 else "ไม่คืนสินค้า"
	)

	if prediction == 1:
		st.warning(f"ผลทำนาย: {result_label}")
	else:
		st.success(f"ผลทำนาย: {result_label}")

	metric_column, model_column = st.columns(2)
	metric_column.metric("โอกาสคืนสินค้า", f"{return_probability:.1%}")
	model_column.metric("โมเดล", model_bundle["model_name"])
	st.progress(return_probability, text="ความน่าจะเป็นที่โมเดลประเมิน")
	st.caption("ผลนี้เป็นการประเมินจากข้อมูลและโมเดล ไม่ใช่ข้อสรุปเชิงเหตุและผล")

st.divider()
st.caption(f"Model artifact: {MODEL_PATH.name}")
