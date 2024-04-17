import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('app/Train.csv')
test = pd.read_csv('app/Test.csv')

combined_data = pd.concat([train, test], ignore_index=True)

label_encoders = {}
category_encodings = {}

for col in combined_data.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    combined_data[col] = label_encoders[col].fit_transform(combined_data[col])
    if col == 'Item_Category':
        category_encodings = dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))

train_encoded = combined_data[:len(train)]
test_encoded = combined_data[len(train):]

X_train = train_encoded.drop(['Selling_Price', 'Date', 'Product', 'Product_Brand'], axis=1)
y_train = train_encoded['Selling_Price']

model = LinearRegression()
model.fit(X_train, y_train)

def get_subcategories_by_category(train_data, category):
    subcategories = train_data[train_data['Item_Category'] == category]['Subcategory_1'].unique()
    return sorted(subcategories)

def predict_selling_price(model, item_category, subcategory_1, subcategory_2, item_rating):
    item_category_encoded = category_encodings.get(item_category, -1)
    subcategory_1_encoded = label_encoders['Subcategory_1'].transform([subcategory_1])[0]
    subcategory_2_encoded = label_encoders['Subcategory_2'].transform([subcategory_2])[0]
    
    if item_category_encoded == -1:
        return "Invalid category"

    prediction = model.predict([[item_category_encoded, subcategory_1_encoded, subcategory_2_encoded, float(item_rating)]])
    return prediction


def main():
    st.title("Selling Price Prediction")
    st.markdown(
        """
        <style>
        .title {
            color: #7E8D85;
            text-align: center;
            font-size: 36px;
            margin-bottom: 20px;
        }
        .header {
            color: #5E5E5E;
            font-size: 24px;
            margin-top: 30px;
        }
        .button {
            background-color: #9CDBB3;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            margin-top: 20px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #7E8D85;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<p class='title'>Selling Price Prediction</p>", unsafe_allow_html=True)
    st.write("Please select item category, subcategory 1, subcategory 2, and enter item rating to predict selling price.")

    item_category_options = sorted(train['Item_Category'].unique())
    item_category = st.selectbox("Item Category", options=item_category_options, index=0)

    subcategory_1_options = get_subcategories_by_category(train, item_category)
    subcategory_1 = st.selectbox("Subcategory 1", options=subcategory_1_options, index=0)

    subcategory_2_options = sorted(train[train['Subcategory_1'] == subcategory_1]['Subcategory_2'].unique())
    subcategory_2 = st.selectbox("Subcategory 2", options=subcategory_2_options, index=0)

    item_rating = st.text_input("Item Rating", placeholder="Type Here")

    if st.button("Predict"):
        result = predict_selling_price(model, item_category, subcategory_1, subcategory_2, item_rating)
        st.success(f"The predicted selling price is: {result[0]:,.2f}")

if __name__ == "__main__":
    main()

