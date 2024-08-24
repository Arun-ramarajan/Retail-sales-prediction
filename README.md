Developed a predictive ANN model to forecast department-wide sales for each store over the years and analyzed the impact of markdowns on sales during holiday weeks from three csv files store, sales and features. Store files have details about the store number, store type and store size. Sales file have the details of Store number, department number, date, weekly sales, holiday indicator. Features file have Store number, date, temperature, fuel price, markdown data (MarkDown1-5), CPI, unemployment rate, holiday indicator, where markdowns shows the discounted rate a department or store offered in a week to imporve the sales. Combining all these files with respective columns gives us a detailed information about the performance of each department in the stores in a week for over years. From this we can analyze the weekly sales in each department, monthly sales rate  in a year where sales is affected and how markdown impacts in the weekly sales at holidays.


Feature Engineering: created new features from the data for better understanding about how markdowns in each stores are improving weekly sales


Data cleaning: used methods like mean, median and machine learning to fili the null values in the data set.

EDA: used matplotlib to visualize the data distribution

ANN: Tensorflow is used to create a predictive ANN model to forecast the sales

Web app: Streamlit is used to create web application for the ANN model

Deploy: Web app is deployedin AWS 

skills used:

- Time Series Analysis
- Feature Engineering
- Predictive Modeling
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Deep Learning Algorithms
- AWS Deployment
- Model Evaluation and Validation
- Data Visualization
- Tensorflow
- Python Programming

