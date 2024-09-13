from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry
from snowflake.snowpark.functions import col, F_month

import altair as alt
import streamlit as st

# Get the current credentials
session = get_active_session()

# App Title and description
st.title("Predictions")
st.write("""Proof of concept Machine Learning app to predict categories.""")

# Get model registry
session.use_database("ml_database")
session.use_schema("ml_schema")
reg = Registry(session)

# Get model
model = reg.get_model("CATEGORY_1_MODEL")

# Get raw data for inference
raw_data_df = session.table("raw_data")

# Run inference
pred_df = model.run(raw_data_df, function_name="predict")
pred_jan = pred_df.filter(F_month(col("date")) == 1)
pred_jan = pred_df.to_pandas()

# Plot data
st.markdown(
    """
    # January 2024
    ## Category 1 Predictions
    """
)
chart = (
    alt.Chart(pred_jan)
    .mark_bar()
    .encode(
        x=alt.X("SITE:N", sort="-y"),
        xOffset="TYPE:N",
        y=alt.Y("VALUE:Q"),
        color=alt.Color("TYPE:N").legend(orient="left"),
    )
    .interactive()
)

st.altair_chart(chart, theme=None, use_container_width=True)
