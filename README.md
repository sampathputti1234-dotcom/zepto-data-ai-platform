# Zepto Data & AI Platform — Capstone Project

This is my Zepto Data & AI Platform capstone project. It contains three modules that cover data processing, analytics and machine learning, and a GenAI support assistant.

1. **Data Pipeline** ('/data_pipeline'): Scrapes book data, cleans it, converts prices to INR, stores it in SQLite, and runs SQL and pandas queries.

2. **Analytics & Modeling** ('/analytics'): Uses the Titanic dataset for data analysis, visualization, and machine learning models.

3. **GenAI Support Assistant** ('/support_assistant'): Answers Zepto policy-related questions using the provided policy documents and a retrieval-based approach.


## Module 1: Data Pipeline (task_1.py)
### Overview

In Module 1, I scraped book information from 'books.toscrape.com' using Python. I cleaned the scraped data, converted the prices from GBP to INR, and stored the final data in a SQLite database.

I also created separate tables for books and categories and used SQL queries to get different results from the database. Finally, I used pandas to reproduce the JOIN query and checked that both SQL and pandas gave the same result.

### Currency Conversion

I used a fixed conversion rate of **1 GBP = 105.50 INR** as given in the project instructions. I used this fixed rate instead of an external API so that the results stay consistent every time the pipeline is run.

### Missing & Corrupted Data Handling

While cleaning the scraped data, I checked for missing or incorrectly parsed values. If a row had corrupted data that could not be converted correctly, I removed that row instead of filling it with an estimated value.

For the three categories I scraped, all **69 books were parsed successfully**, so no rows had to be removed.

### SQLite Database

I stored the cleaned book data in a SQLite database using two tables: 'categories' and 'books'. The 'categories' table stores each category, while the 'books' table stores the book details and links to the category using 'category_id'.

I also enabled SQLite foreign key support so that the relationship between the two tables is properly maintained.

### SQL Queries

I created and ran five SQL queries to check the data and get useful results from the database.

1. **SELECT, WHERE, LIMIT** – Find in-stock books under £25.
2. **ORDER BY, LIMIT** – Find the 5 most expensive books based on INR price.
3. **DISTINCT** – Check the different star ratings available.
4. **IN, BETWEEN** – Find books with ratings of 4 or 5 and prices between ₹2,000 and ₹4,000.
5. **JOIN** – Combine book details with their category names and find 5-star books sorted by price.

These queries helped me verify that the data was stored correctly and could be retrieved from the database as expected.

### Pandas Verification

I used 'pd.read_sql_query()' to get the JOIN query result directly from the SQLite database. I then loaded the required tables into pandas and used 'pd.merge()' to create the same result in memory.

I compared both results using 'pandas.testing.assert_frame_equal()' to make sure the SQL and pandas results were equivalent.


## Module 2: Analytics & Modeling (`/analytics`)

### Overview

In Module 2, I worked with the Titanic dataset to understand the data, handle missing values, explore relationships between different features, and build machine learning models.

I used data visualization and statistical analysis during the exploration. After that, I trained classification models to predict passenger survival and also built a regression model to predict fare.

### Data Loading & Cleaning

I loaded the Titanic dataset using Seaborn and saved a copy as 'titanic.csv' so that the cleaned data could be reused for the remaining analysis and modeling work.

I checked the dataset information, shape, summary statistics, and missing values before cleaning it. For columns with missing values, I used row removal or median/missing-value handling depending on the amount and type of missing data.

### Exploratory Data Analysis

I explored the Titanic data using different charts and summary statistics. I looked at age and fare distributions, checked for outliers using the IQR method, and compared survival rates across gender and passenger class.

I also used correlation analysis to understand the relationships between the main numerical features. Different multivariate charts were used to look at patterns between variables and survival.

### Machine Learning & Preprocessing

Before training the models, I split the data into training and testing sets using stratification so that the survival class proportions stayed similar in both sets.

I used a preprocessing pipeline to handle missing values, encode categorical features, and scale numerical features. The preprocessing steps were fitted only on the training data to avoid data leakage.

I trained three classification models: **Logistic Regression, Decision Tree, and Random Forest**. I compared them using accuracy, precision, recall, F1 score, and ROC-AUC.

### Class Imbalance & Model Tuning

I checked the distribution of the survived and not-survived classes in the training data. To see how class imbalance affected the results, I compared the baseline model with 'class_weight='balanced'' and a model trained using SMOTE on the training data.

I also used 'GridSearchCV' to tune the Random Forest model by testing different values for the number of trees, maximum depth, and maximum features. The best parameters and the corresponding cross-validation and OOB scores were recorded.

### Regression & Model Comparison

I also built a linear regression model to predict 'fare' using the other available features. I evaluated it using MAE, RMSE, R², and Adjusted R², and used a residual plot to check the model's errors.

Finally, I put the classification model results together in one comparison table and kept the regression metrics separate because they measure a different type of prediction problem. The final model choice was based on the classification metrics and the results from the model evaluation.


## Module 3: GenAI Support Assistant (`/support_assistant`)

### Overview

In Module 3, I built a small support assistant that answers questions about Zepto policies. I used the provided policy documents as the knowledge source and stored their embeddings in ChromaDB for retrieval.

The assistant uses LangGraph to decide whether a question is related to a Zepto policy. Policy questions go through retrieval before generating an answer, while general questions are handled directly.

### Architecture & Data Flow

The support assistant follows this flow:

'Policy Documents → Chunking → Embeddings → ChromaDB → Retrieval → Answer Generation'

The policy documents are loaded and converted into embeddings using 'all-MiniLM-L6-v2'. The embeddings are stored in ChromaDB, and when a policy question is asked, the system retrieves the top 3 relevant documents before generating the response.

LangGraph controls the flow using three nodes: 'classify_intent', 'retrieve_and_answer', and 'direct_answer'. The 'classify_intent' node checks whether the question is a policy question. Policy questions go to retrieval, while other questions go directly to the general-answer node.

### Mock Mode

The project runs in mock mode by default using the 'MOCK_LLM=1' environment variable. In this mode, policy questions still use the real ChromaDB retrieval, but the final answer is generated using a fixed response format instead of an external LLM.

If "MOCK_LLM=0" is used, the code follows the optional real-LLM path. The routing between policy and general questions stays the same in both modes.

### API Usage

The support assistant provides a "POST /ask" endpoint.

**Example 1:**

'''json
{
  "query": "How long is a Zepto gift card valid?"
}
'''

**Example 2:**

'''json
{
  "query": "Can I cancel my order after it is packed?"
}
'''

The API returns a validated response containing the answer, source document IDs, and a confidence value.



## Setup & Running

### Requirements

* Python 3.10 or above
* Required Python packages from 'requirements.txt'

Install the packages with:

'''bash
pip install -r requirements.txt
'''

### Run Module 1

From the repository root:

'''bash
cd data_pipeline
python task_1.py
'''

This runs the scraping, cleaning, SQLite storage, SQL queries, and pandas verification.

### Run Module 2

Open the notebooks inside the '/analytics' folder and run them in order.

The analysis uses the saved 'titanic.csv' file so that the same cleaned data can be reused for the modeling steps.

### Run Module 3

Build the Docker image from the repository root:

'''bash
docker build -t zepto-support .
'''

Run the container:

'''bash
docker run -p 7860:7860 zepto-support
'''

The FastAPI support assistant will then be available locally on port '7860'.

The default mode is 'MOCK_LLM=1', so no external paid LLM service is required.
