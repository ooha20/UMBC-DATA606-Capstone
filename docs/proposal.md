# E-commerce Sales Prediction

**Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaoji (Jay) Wang - SPRING 2024 Semester**

**Author: Ooha Reddy Birru**

GitHub:https://github.com/ooha20

LinkedIn:https://www.linkedin.com/in/ooha-reddy-755b36228?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app

## Background:

### 1. What is it about?

The project revolves around predicting the prices of products on an e-commerce platform. E-commerce platforms have become integral to modern commerce, with a significant impact on consumer behavior. As users increasingly rely on these platforms for purchasing goods, accurately predicting product prices becomes crucial for both consumers and sellers. The project focuses on leveraging machine learning techniques to build a model capable of predicting the selling prices of various products.

### 2. Why does it matter?

E-commerce platforms operate in a highly competitive environment where pricing plays a pivotal role. Predicting product prices accurately can benefit both consumers and sellers. For consumers, it ensures fair and transparent pricing, aiding in making informed purchasing decisions. For sellers, accurate price prediction facilitates optimal pricing strategies, inventory management, and overall business planning. Additionally, a reliable price prediction model contributes to enhanced user experience and customer satisfaction.

### 3. What are your research questions?

- **How do various factors influence the pricing of products on an e-commerce platform?** 
- **Can machine learning models effectively predict the selling prices of products on e-commerce platforms?** 
- **What impact does data preprocessing, feature engineering, and model selection have on the prediction accuracy?** 
- **How does the model performance compare to other regression algorithms or ensemble methods?** 

Addressing these research questions provides a comprehensive understanding of the dynamics influencing e-commerce product pricing and the effectiveness of the chosen machine learning model in this context.

## Data:
### Data Sources:

The datasets used for this project include two CSV files: [Train.csv](https://www.kaggle.com/code/venkatkrishnan/sales-prediction-regression-problem/input) and [Test.csv](https://www.kaggle.com/code/venkatkrishnan/sales-prediction-regression-problem/input), containing information about products on an e-commerce platform.


### Data Size:
- Train.csv: 188.36 kB
- Test.csv: 74.44 kB

### Data Shape:

- Train.csv: 2452 rows, 8 columns
- Test.csv: 1051 rows, 7 columns

### Each Row Represents:

Each row in the dataset represents a specific product on the e-commerce platform. The columns contain various attributes related to the product, such as its ID, brand, category, subcategories, item rating, date of listing, and the selling price.

### Data Dictionary:

Columns in both Train.csv and Test.csv:

1. **Product:** Product ID (Object)
2. **Product_Brand:** Brand of the product (Object)
3. **Item_Category:** Main category of the item (Object)
4. **Subcategory_1:** Subcategory 1 of the item (Object)
5. **Subcategory_2:** Subcategory 2 of the item (Object)
6. **Item_Rating:** Rating assigned to the item (Float)
7. **Date:** Date of listing (Object)
8. **Selling_Price:** Target variable - Price at which the product is being sold (Float)

### Potential Values:

- Categorical columns (Object type) like Product, Product_Brand, Item_Category, Subcategory_1, Subcategory_2 have unique values representing different categories.
- Item_Rating is a float column representing the rating of the product.
- Date is an object column representing the date of listing.
- Selling_Price is a float column representing the target variable.

### Target/Label:

The target variable for the machine learning model is "Selling_Price." The objective is to predict the selling price of a product based on the given features.

### Features/Predictors:

The potential features/predictors for the machine learning models include:

- Product_Brand
- Item_Category
- Subcategory_1
- Subcategory_2
- Item_Rating
- Date (after potential feature engineering)

It's important to note that further exploration and preprocessing may be required to extract meaningful information from the "Date" column for it to be used as a feature. Additionally, techniques like one-hot encoding may be applied to handle categorical variables appropriately.
