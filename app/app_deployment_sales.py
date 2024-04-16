import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('app/Train.csv')
test = pd.read_csv('app/Test.csv')

combined_data = pd.concat([train, test], ignore_index=True)

label_encoders = {}
category_encodings = {}
subcategory_1_encodings = {}
subcategory_2_encodings = {}

for col in combined_data.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    combined_data[col] = label_encoders[col].fit_transform(combined_data[col])
    if col == 'Item_Category':
        category_encodings = dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))
    elif col == 'Subcategory_1':
        subcategory_1_encodings = dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))
    elif col == 'Subcategory_2':
        subcategory_2_encodings = dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))

train_encoded = combined_data[:len(train)]
test_encoded = combined_data[len(train):]

X_train = train_encoded.drop(['Selling_Price', 'Date', 'Product', 'Product_Brand'], axis=1)
y_train = train_encoded['Selling_Price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_selling_price(model, item_category, subcategory_1, subcategory_2, item_rating):
    
    item_category_encoded = category_encodings.get(item_category, -1)
    subcategory_1_encoded = subcategory_1_encodings.get(subcategory_1, -1)
    subcategory_2_encoded = subcategory_2_encodings.get(subcategory_2, -1)
    
    if item_category_encoded == -1 or subcategory_1_encoded == -1 or subcategory_2_encoded == -1:
        return "Invalid category or subcategory"

    prediction = model.predict([[item_category_encoded, subcategory_1_encoded, subcategory_2_encoded, float(item_rating)]])
    return prediction

def main():
    st.title("Selling Price Prediction")
    st.write("Please select item category, subcategory 1, subcategory 2, and enter item rating to predict selling price.")

     item_category_options = sorted(train['Item_Category'].unique())
    item_category = st.selectbox("Item Category", 
                                 options=item_category_options, 
                                 format_func=lambda x: f"Select item category" if x == item_category_options[0] else f"{x} - {label_encoders['Item_Category'].transform([x])[0]}",
                                 index=0)


    subcategory_1_options = ['Women Clothing', 'Men Clothing', 'Kids Clothing']
    subcategory_1 = st.selectbox("Subcategory 1", 
                                  options=subcategory_1_options, 
                                  index=0)

    subcategory_2_options = ['Western Wear']
    subcategory_2 = st.selectbox("Subcategory 2", 
                                  options=subcategory_2_options, 
                                  index=0)

    item_rating = st.text_input("Item Rating", placeholder="Type Here")

    if st.button("Predict"):
        result = predict_selling_price(model, item_category, subcategory_1, subcategory_2, item_rating)
        if isinstance(result, str):
            st.error(result)
        else:
            st.success(f"The predicted selling price is: {result[0]:,.2f}")

if __name__ == "__main__":
    main()
