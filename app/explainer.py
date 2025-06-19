import streamlit as st
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from loader import model  # Assumes model is already fitted Pipeline

def app(input_data):
    st.markdown("---")
    # Full transform
    sample_transformed = model.named_steps['feature_engineering'].transform(input_data)
    sample_woe = model.named_steps['woe_encoding'].transform(sample_transformed)
    sample_final = model.named_steps['column_selector'].transform(sample_woe)

    # Get training data for explainer
    # Ideally you save a sample during training and load here
    # For demo, we generate a dummy sample
    explainer_data = pd.DataFrame(np.random.rand(100, sample_final.shape[1]), columns=sample_final.columns)
    class_names = ['No Diabetes', 'Diabetes']

    # LIME explainer
    explainer = LimeTabularExplainer(
        training_data=explainer_data.values,
        feature_names=explainer_data.columns.tolist(),
        class_names=class_names,
        mode='classification'
    )

    # Explain the instance
    exp = explainer.explain_instance(
        sample_final.iloc[0].values,  
        lambda x: model.named_steps['model'].predict_proba(pd.DataFrame(x, columns=sample_final.columns)),
        num_features=10
    )
    # Stream inputs
    def stream_data():
        st.markdown("Your Inputs: ")
        text = f"""
`Pregnancies`: {float(input_data.at[0, 'Pregnancies'])}\n
`Glucose`: {float(input_data.at[0, 'Glucose'])}\n
`Insulin`: {float(input_data.at[0, 'Insulin'])}\n
`BMI`: {float(input_data.at[0, 'BMI'])}\n
`Age`: {float(input_data.at[0, 'Age'])}
        """
        for word in text.split(" "):
            yield word + "  "
            time.sleep(0.05)
        st.markdown(
            """
            ### Column Explanations
            - 🟡 **Input Streaming**: Displays real-time inputs.
            - 🟡 **LIME Plot**: Feature impact on the current prediction.
            """
        )

    # Layout
    cols = st.columns(2)


    # Column 1: Input streaming
    with cols[0]:
        st.markdown("### Input Streaming")
        for word in stream_data():
            st.write(word)

    # Column 2: LIME Explanation
    with cols[1]:
        st.markdown("### Feature Contributions (LIME)")
        st.markdown("""
##### 🟢 LIME Explanation Plot
This plot shows the **top 10 features** that influenced the model's prediction for your input:
- **🟩 Green bars**: indicate features pushing the prediction **toward** the predicted class (e.g., Diabetes).
- **🟥 Red bars**: indicate features pushing the prediction **against** the class.
- **📊 Bar length**: represents the **strength of influence**.
- It helps explain **why** the model predicted what it did, making the model more **interpretable and transparent**.
""")
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
        st.markdown("""
###### ⚠️ Note:
This is a machine learning model, not a medical diagnosis. Always consult a medical professional for accurate health advice.
""")





# import streamlit as st
# import shap
# import time
# from loader import model
# import matplotlib.pyplot as plt


# def app(input_data):
#     sample_transformed = model.named_steps['feature_engineering'].transform(input_data)
#     explainer = shap.TreeExplainer(model.named_steps['model'])
#     shap_values_single = explainer.shap_values(sample_transformed)

#     shap_values_class_1 = shap_values_single[0][:, 1]  


#     def stream_data():
#         text = f"""
# Your inputs:\n
# `Pregnancies`: {float(input_data.iloc[0]['Pregnancies'])}\n
# `Glucose`: {float(input_data.iloc[0]['Pregnancies'])}\n
# `Insulin`: {float(input_data.iloc[0]['Pregnancies'])}\n
# `BMI`: {float(input_data.iloc[0]['Pregnancies'])}\n
# `Age`: {float(input_data.iloc[0]['Pregnancies'])}
#                 """
#         for word in text.split(" "):
#             yield word + " "
#             time.sleep(0.05)

#     # Layout with two columns
#     cols = st.columns(2)

#     # Column 1: Stream user input
#     with cols[0]:
#         st.markdown("### Input Streaming")
#         st.markdown("#### See your inputs in real-time below!")
#         for word in stream_data():
#             st.write(word)

#     # SHAP Waterfall Plot
#     fig, ax = plt.subplots()
#     shap.plots.waterfall(
#         shap.Explanation(
#             values=shap_values_class_1,
#             base_values=explainer.expected_value[0],
#             data=sample_transformed.iloc[0],
#             feature_names=sample_transformed.columns.tolist()
#         ), show=False
#     )
#     fig.patch.set_facecolor("lightblue")
#     fig.patch.set_alpha(0.3)
#     ax.set_facecolor("#023047")
#     ax.patch.set_alpha(0.5)

#     # Column 2: SHAP Waterfall Plot
#     with cols[1]:
#         st.markdown("### SHAP Waterfall Plot")
#         st.markdown(
#             """
#             - 🟡 **Base Value**: Expected model prediction without considering input features.
#             - 🟡 **Feature Contributions**: Bars represent individual feature impact.
#             - 🟡 **Output Prediction**: Sum of base value and contributions gives final output.
#             """
#         )
#         st.pyplot(fig)

#     # SHAP Force Plot
#     force_plot_html = shap.force_plot(
#         base_value=explainer.expected_value[1],
#         shap_values=shap_values_single[0][:, 1],
#         features=sample_transformed.iloc[0],
#         feature_names=sample_transformed.columns.tolist()
#     )

#     # Explanation column
#     st.markdown(
#         """
#         ### Column Explanations
#         - 🟡 **Input Streaming**: Displays user inputs dynamically in real-time.
#         - 🟡 **SHAP Waterfall Plot**: Visualizes how each feature contributes to the model prediction.
#         - 🟡 **SHAP Force Plot**: Interactive plot showing positive/negative feature contributions.
#         \n\n\n\n""",
#         unsafe_allow_html=True,
#     )

#     # Add SHAP JS visualization
#     force_plot_html = f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>"
#     st.markdown("### SHAP Waterfall Plot")
#     st.components.v1.html(force_plot_html, height=400)