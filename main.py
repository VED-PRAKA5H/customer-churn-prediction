import pickle  # For loading serialized Python objects
import pandas as pd  # For data manipulation and analysis
import streamlit as st  # For creating web applications
import os  # For interacting with the operating system
from src.utils import extract_cat_columns, extract_num_columns, encode_transform, make_float, transform_column

# Custom CSS for aesthetics (optional)
page_bg_style = '''
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #190482, #7752FE, #8E8FFA, #C2D9FF);
        background-size: cover;
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.5);
    }
    h1, h2, h3 {
        color: white;
        text-shadow: 2px 2px 4px #000000;
    }
</style>
'''
# Apply custom CSS to the Streamlit app
st.markdown(page_bg_style, unsafe_allow_html=True)

# Define the path to the data directory
data_path = f"{os.path.dirname(os.path.realpath(os.path.abspath(__file__)))}/notebook/data/"

# Load the pre-trained XGBoost model from a pickle file
with open(f"{data_path}xgb_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Sample input data for the demo
demo_input = {
    "AccountAge": 20,
    "MonthlyCharges": 11.055215098286784,
    "TotalCharges": 221.10430196573566,
    "SubscriptionType": ["Standard", "Basic", "Premium"],
    "PaymentMethod": ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
    "PaperlessBilling": ["No", "Yes"],
    "ContentType": ["Both", "TV Shows", "Movies"],
    "MultiDeviceAccess": ["No", "Yes"],
    "DeviceRegistered": ["Computer", "Tablet", "Mobile", "TV"],
    "ViewingHoursPerWeek": 36.75810391025656,
    "AverageViewingDuration": 63.53137733399087,
    "ContentDownloadsPerMonth": 10,
    "GenrePreference": ["Comedy", "Fantasy", "Drama", "Action", "Sci-Fi"],
    "UserRating": 2.1764975145384615,
    "SupportTicketsPerMonth": 4,
    "Gender": ["Female", "Male"],
    "WatchlistSize": 3,
    "ParentalControl": ["Yes", "No"],
    "SubtitlesEnabled": ["Yes", "No"]
}

# Create a Pandas DataFrame from the demo input
df = pd.DataFrame([demo_input])

# Title of the Streamlit app
st.title("✨ Customer Churn Prediction ✨")

# Create two columns for side-by-side input fields
col1, col2 = st.columns(2)

# Loop through each column in the DataFrame to create input fields
for column in df.columns:
    if df[column].dtypes != 'object':  # Check if the column is numerical
        with col1:
            df[column] = st.number_input(f"Enter {column}:", value=demo_input[column])  # Numerical input field
    else:  # If the column is categorical
        with col2:
            df[column] = st.selectbox(f"Select {column}", demo_input[column])  # Dropdown selection

# Load encoders for categorical variables from a pickle file
with open(f"{data_path}encoders.pkl", 'rb') as file:
    encoders = pickle.load(file)

# Extract categorical columns from the DataFrame
cat_columns = extract_cat_columns(df)

# Apply encoding transformations on categorical columns using loaded encoders
df = encode_transform(df, cat_columns, encoders)

# Transform the 'TotalCharges' column to reduce skewness
df["TotalCharges"] = transform_column(df["TotalCharges"])

# Extract numerical columns from the DataFrame
num_columns = extract_num_columns(df)

# Convert all numerical columns to float type for consistency in predictions
df = make_float(df, num_columns)

# Button to trigger prediction calculation when clicked
if st.button("Show Result"):
    # Make prediction using the loaded model on the processed DataFrame
    result = model.predict(df)[0]

    # Get prediction probabilities for interpretation of results
    pred_prob = model.predict_proba(df)[0]

    # Determine prediction outcome based on model result (0 or 1)
    prediction = 'Churn' if result == 0 else 'Not Churn'

    # Display prediction results in the Streamlit app
    st.subheader("🔥 Prediction Result:")
    st.write(f"Based on your inputs, the customer will: **{prediction}**")
    st.write(f"With a 📊 prediction probability: {pred_prob}")
else:
    st.write("_📈 Click the button to see your prediction!_")  # Prompt user to click button for result
