#!/usr/bin/env python
# coding: utf-8

# ## data_analysis_logistic_regression
# 
# Data Science Degree Assignment for Data Analytics module. The notebook focusses on building a customer churn model for Claranet.
# 

# # **Data Analytics Assessment Project** 
# ## **Claranet Customer Churn Prediction using Logistic Regression and PySpark**
# 
# **Claranet is a global information technology service provider delivering consultancy and solutions across cloud, 
# cybersecurity, data, AI, applications, networks and workplace. The current use of data science and Machine Learning 
# in Claranet UK is limited. However, there are many opportunities for such projects to add value to the business.**
# 
# As a Business-to-Business supplier Customer Churn is always a concern. Several organisational restructures have 
# exacerbated  the issue and the topic is always discussed at the Financial review meetings. 
# There is an old Customer Churn predictive model which was developed several years ago by a member of staff 
# who later left the business. This model uses a rule-based analysis of data which is exported from the source systems 
# manually. No one really understands the logic behind this system and if it delivers anything of value to the business. 
# The success of the model is not measurable, and it is not managed or updated to reflect changes to the way data is 
# recorded or changes to business processes.
# 
# The analysis is performed using pyspark rather than pandas. Pandas is ideal for smaller datasets and offers great flexibility and ease of use for detailed data manipulation. However pyspark harnesses the power of distributed computing to efficiently process large-scale datasets across multiple machines. In this case the dataset is not particularly large and the overhead of managing distributed computing can outweigh the benefits for smaller datasets and pyspark may be slower than scala or java as the python code must be translated into Java Virtual Machine code instructions. However it is important to develop the skills and knowledge required to use pyspark as there are many examples of ML Ops using pandas and skikit learn and far fewer of pyspark and performance is not critical to this project. It would be easy to rewrite the code using pandas dataframes and skikit learn for a comparison at a later date.
# 

# #### **Import Python Modules and Functions**
# **Synapse Clusters have most of the Common Python libraries installed by default**

# In[12]:


#general
from pyspark.sql.session import SparkSession, SparkConf
from pyspark.sql.functions import col, when, lit, sum, count_distinct, date_add, datediff, current_date
from datetime import datetime
import numpy as np

#plotly express for creating charts in the notebook UI
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

#required for ML Ops
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RFormula, OneHotEncoder, StringIndexer, VectorIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, GBTClassifier, RandomForestClassifier
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator,  RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# #### **Define Data Analytics Functions**
# **Run data_science_analytics_functions notebook**

# In[13]:


get_ipython().run_line_magic('run', 'data-science-project/data_science_analytics_functions')


# ### **Load Data for Analysis into Spark Dataframes**
# ##### The data is sourced 2 databases within the uk-data-platform. Views of the data have been created in the analyticshub database datascience schema using cross database references. The data can be loaded directly using the JDBC driver for SQL Server. This uses the connection string in the synapse linked service which connects to the database from the workspace. The authentication is via the synapse workspace system managed identity
# 
# **The data is sourced from 4 views in the analyticshub database in the uk-data-platfrom**
# - **ServiceNow Incidents** - The number of incidents that the customer has reported on a priority scale of 1-5, 1 being the most critical within a specific time interval 
# - **ServiceNow Customer Service Tickets (Complaints)** - This is the number of complaints raised by the customer.
# - **ServiceNow Assessment Results (Customer Satisfaction Surveys)** - This provides the results of te Customer satisfaction Surveys which are sent out to customers
# - **Messina Order Atom Data** - This provides the Customer order cancellation indicator or churn
# 

# In[14]:


#create datframes from analyticshub views

database_name = "analyticshub"
linkedservice_name = "linkedservice_synondemandsql_analyticshub"

#define sql queries
sql_query_incident = "(select incident_number,incident_created,incident_priority,incident_customer_finance_ref,\
                       incident_customer_name from datascience.servicenow_incident) as servicenow_incident"

sql_query_orderatom = "(select order_orderid,order_orderdate,order_status,customer_custid,customer_name,\
                        customer_finance_ref,orderatom_orderatomid,orderatom_state,orderatom_dateinstalled,\
                        orderatom_canceldate,orderatom_cancelation_reason,orderatom_service_replace_type,\
                        orderatom_service_install_cost, orderatom_total_annual_recurr_cost,\
                        orderatom_total_annual_recurr_value from datascience.messina_order_atom) as messina_orderatom"

sql_query_ticket  = "(select csi_ticket_sys_id,csi_ticket_sys_created_on,csi_ticket_subcategory,csi_ticket_root_cause,\
                      csi_ticket_subcause,csi_ticket_root_cause_codes,csi_ticket_impact,csi_ticket_customer_finance_ref \
                      from datascience.servicenow_csi_tickets) as servicenow_csi_ticket"

sql_query_assessment = "(select asmt_result_sys_id,asmt_result_instance,asmt_result_sys_created_on,\
                         asmt_result_sys_updated_on,asmt_metric_type,asmt_result_actual_value,asmt_customer_finance_ref from \
                         datascience.servicenow_assessment_metric_results) as servicenow_assessment"

#load data into dataframes
df_incident = dataframe_load_from_synapse_database(linkedservice_name, database_name, sql_query_incident)

df_orderatom = dataframe_load_from_synapse_database(linkedservice_name, database_name, sql_query_orderatom)

df_ticket = dataframe_load_from_synapse_database(linkedservice_name, database_name, sql_query_ticket)

df_assessment =  dataframe_load_from_synapse_database(linkedservice_name, database_name, sql_query_assessment)


# ### **Data Cleaning and Preparation**
# ##### Perform data cleaning of dataframes using the dataframe_clean_data() function
# **Servicenow Incident**
# 1. Remove rows with null customer_finance_ref
# 2. Remove rows with null or invalid priority (priority should be integer range 1-5)
# 3. Remove duplicate rows based upon incident number
# 
# **Servicenow Ticket**
# 1. Remove rows with null csi_ticket_customer_finance_ref
# 2. Remove duplicates based on csi_ticket_sys_id
# 
# **Servicenow Assessment**
# 1. Filter on asmt_metric_type = 'Customer Relationship Survey' to obtain Customer Relationship Scores
# 2. Remove rows with null asmt_customer_finance_ref
# 3. Remove duplicates based on asmt_result_sys_id
# 
# **Messina Orderatom**
# 1. Remove rows with null customer_finance_ref
# 3. Remove rows with null order_orderid
# 3. Remove rows with null order_orderdate
# 4. Remove duplicate rows based upon orderatom_id
# 5. Remove rows where orderatom_cancelation_reason = 'Products test order - please ignore'

# In[15]:


#clean data servicenow_incident
null_col_list = ["incident_number","incident_created","incident_customer_finance_ref"]
df_incident = dataframe_clean_data(df_incident, "servicenow incident", null_col_list, ["incident_number"])

#clean data servicenow_ticket
null_col_list = ["csi_ticket_customer_finance_ref"]
df_ticket = dataframe_clean_data(df_ticket, "servicenow csi ticket", null_col_list, ["csi_ticket_sys_id"])

#clean data servicenow_assessment
null_col_list = ["asmt_customer_finance_ref"]
df_assessment = dataframe_clean_data(df_assessment, "servicenow assessment", null_col_list, ["asmt_result_sys_id"])
df_assessment = df_assessment.filter(col("asmt_metric_type")=="Customer Relationship Survey")
print(f"Number of rows servicenow assessment AFTER filtering on metric type Customer Relationship Survey\
= {dataframe_row_count(df_assessment)}")

#clean data messina_orderatom
null_col_list = ["order_orderid","order_orderdate","customer_finance_ref","orderatom_orderatomid"]
df_orderatom = dataframe_clean_data(df_orderatom, "messina orderatom", null_col_list, ["orderatom_orderatomid"])
df_orderatom = df_orderatom.filter(col("orderatom_cancelation_reason") != "Products test order - please ignore")
print(f"Number of rows messina orderatom AFTER removing Test orders = {dataframe_row_count(df_orderatom)}")


# ### **Feature Engineering**
# ##### **In this section we join the dataframes from the different sources to create a dataframe containing the features we would like to assess for their suitability.**
# 
# **1. Create a label column cancelled in the Order Atom Dataset**
# Add column cancel (int 0:1) where the orderatom_state = Cancelled and orderatom_canceldate is not null and **orderatom_cancelation_reason != Service Replaced** (This implies that the order has been replaced by a different product which could be an upgrade (or downgrade!) and is not a true cancellation)
# 
# **2. Create column days_installed** which is the number of days between the orderatom_dateinstalled and the analysis date (current_date)
# 
# **3. Create column number_complaints by LEFT OUTER joining df_complaint on csi_ticket_customer_finance_ref = customer_finance_ref**
# 
# **4. Create column crs_score by LEFT OUTER joining df_crs_score on asmt_customer_finance_ref = customer_finance_ref**
# 
# **5. Create column number_incidents by LEFT OUTER joining df_incident on incident_customer_finance_ref = customer_finance_ref**
# 
# **6. Create Columns for Number of Priority Incidents within Time frame in the Order Atom Dataset**
# - Incident must be created prior to the cancellation of the order since we are trying to establish if the incident is a contributing factor to cancellation of the order
# - Add separate columns for each incident priority classification between 1 and 5 - Priority 1 being the most severe
# - Experiment with different time intervals after the orderatom date installed (50, 100 and 200)
# - This results in the addition of 3 columns for each priority classification
# - Columns will be named following the convention p[priority]_[x] where priority = 1-5 and x=days (50, 100, 200 ) i.e. p1_50
#  
#  In order to add the incident count columns it is neccessary to join the two dataframes since udf cannot call another dataframe
# 
# - Join dataframes on customer_finance_ref
# - Use Filtering and Aggregation to determine number of incidents of given priority within timeframe after order installation date
# - Join dataframes back together using orderatom_orderatomid
# 

# In[16]:


#Create a binary column representing order cancellation in the Order Atom Dataset - Add column cancel (int 0:1) 
#where orderatom_state = Cancelled & orderatom_canceldate is not null and 
#orderatom_cancelation_reason != Service Replaced

#declare variables for incident count calculation 
days_from_installed = []
incident_priorities = []

#set variables for incident count calculation 
days_from_installed = [0,50,100,200]
incident_priorities = [1,2,3,4,5]

# 1. Create column cancelled
df_orderatom = df_orderatom.withColumn("cancelled", when((df_orderatom.orderatom_state == "cancelled") \
                                       & (df_orderatom.orderatom_cancelation_reason != "Service Replaced")
                                       & (df_orderatom.orderatom_canceldate.isNotNull()),lit(1)).otherwise(lit(0)))

#2. Create column days_installed which is the number of days between the orderatom_dateinstalled and the analysis date (current_date)
df_orderatom = df_orderatom.withColumn("days_installed", datediff(current_date(),col("orderatom_dateinstalled")))

#3. Create column number_complaints by joining df_complaint on csi_ticket_customer_finance_ref
df_ticket_number = df_ticket.groupBy("csi_ticket_customer_finance_ref").agg(count_distinct("csi_ticket_sys_id").alias("number_complaints"))
df_orderatom = dataframe_join(df_orderatom, df_ticket_number,["customer_finance_ref","csi_ticket_customer_finance_ref"], "left_outer")

# 4. Create column crs_score by joining df_assessment - the score has already been normalised
df_crs_score = df_assessment.groupBy("asmt_customer_finance_ref").agg(sum("asmt_result_actual_value").alias("crs_score"))
df_orderatom = dataframe_join(df_orderatom, df_crs_score,["customer_finance_ref","asmt_customer_finance_ref"], "left_outer")

# 5. Create columns for count of incidents
df_incident_count = df_incident.groupBy("incident_customer_finance_ref").agg(count_distinct("incident_number").alias("number_incidents"))
df_orderatom = dataframe_join(df_orderatom, df_incident_count,["customer_finance_ref","incident_customer_finance_ref"], "left_outer")
df_orderatom = df_orderatom.drop("incident_customer_finance_ref")

# 6. Create columns for count of incidents for priority and timeframe
#left join df_orderatom to df_incident based upon customer_finance_ref
df = dataframe_join(df_orderatom, df_incident, ["customer_finance_ref","incident_customer_finance_ref"], "left_outer")
df = df.select("customer_finance_ref","orderatom_orderatomid","orderatom_dateinstalled","incident_number","incident_created","incident_priority")

#create the dataframe for the analysis
df_analysis = df_orderatom

# Add incident priority counts
for priority in incident_priorities:
    for days in days_from_installed:
        df_inc_count = dataframe_incident_counts_priority_daterange(df, priority, days)     
        #join to df_orderatom,df
        df_analysis = dataframe_join(df_analysis,df_inc_count,["orderatom_orderatomid","orderatom_orderatomid"],"left_outer")


# #### **Features to Include/Engineer and Rationalisation** 
# 
# **Customer name and finance_ref SHOULD NOT BE INCLUDED for the following reasons:**
# - **Overfitting:** The customer ID is unique to the customer. If a specific customer ID happens to correlate highly with churn in the training data then the model might memorize this specific ID instead of learning general patterns that apply to new customers [1, 2]. This means the model will probably perform poorly when making predictions on real data.
# 
# - **Lack of Generalisability:** Customer IDs are identifiers not features that describe customer behavior or characteristics. A model needs features that have a consistent, generalisable relationship with the target variable (churn) across the entire dataset. [2].
# 
# - **High Cardinality and Model Complexity:** Including a unique ID as a categorical variable would create a vast number of levels making the model overly complex and computationally expensive [1, 2]. 
# 
# **Features to Include**
# 1. **Length of time between installation Date and Current Date (days_installed)** if a product or service has just been installed it may be easier to cancel it. Simularly if a product or service has been in use for a long time maybe the customer will consider that it has had its value. (Consider natural end of contract?)
# 2. **orderatom_service_install_cost** - If a product or service has a high installation cost then it may be less likely that it would be cancelled.
# 3. **orderatom_total_annual_reccurr_cost** - If a product or service has a high operational cost to the customer then then may be more likely to cancel it.
# 4. **number_incidents** - We have engineered columns representing different counts of the priority classification within different time intervals from the order installation date. We can experiment with different columns but start with count of incidents. (not within timeframe)
# 5. **number_complaints** - Number of complaints csi tickets logged by the customer. (not within timeframe)
# 6. **crs_score** - CRS Score for customer. (not within timeframe)
# 
# ```
# **feature_columns = ["days_installed","orderatom_service_install_cost","orderatom_total_annual_reccurr_cost","number_incidents","number_complaints","crs_score"]**
# ```
# 

# #### **Create Dataframe for Anaysis Containing Only Features and Target**
# **Fill null Numerical Column Values with Zero**

# In[17]:


df_feature_analysis = df_analysis.select("days_installed","orderatom_service_install_cost","orderatom_total_annual_recurr_cost",\
                                        "number_incidents","number_complaints","crs_score","cancelled")

                                        #Omit these columns for now - assess suitability of number_incidents
                                        #"number_p1_incidents","number_p2_incidents","number_p3_incidents",
                                        #"number_p4_incidents","number_p5_incidents", 
                                        
#fill nulls with zeros 
df_feature_analysis = df_feature_analysis.fillna(value=0)

print(f"Number of rows in feature analysis dataframe: {dataframe_row_count(df_feature_analysis)}")
print(f"Number of rows in feature analysis dataframe with cancelled = 1: {dataframe_row_count(df_feature_analysis.filter(col('cancelled')==lit(1)))}")
print(f"Number of rows in feature analysis dataframe with cancelled = 0: {dataframe_row_count(df_feature_analysis.filter(col('cancelled')==lit(0)))}")


# #### **Perform Univariant Analysis on Each Feature to Establish which Variables are Significant**
# 
# The conventional technique is to first run the univariate analyses and then use only those variables which meet a preset cutoff for significance to run a multivariable model. This cutoff is normally more liberal than the conventional cutoff for significance (e.g., P < 0.10, instead of the usual P < 0.05) since its purpose is to establish which variables shoud be used rather than to test a hypothesis.
# 
# ##### **Explanation of Model Metrics**
# **Receiver Operating Characteristic Area Under ROC:** The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Sensitivity) against the False Positive Rate (1-Specificity) for a binary classifier at various thresholds, while the Area Under the ROC Curve (AUC) summarizes its overall performance as a single score from 0 to 1, representing the probability that the model correctly ranks a random positive example higher than a random negative one. An AUC of 1.0 is perfect, 0.5 is random guessing, and a higher AUC indicates better class separation and diagnostic accuracy, making it a key tool in evaluating model discriminative power.
# 
# **Accuracy:** The fraction of total predictions that are correct. While simple to understand, it can be misleading in cases of imbalanced datasets where one class is much more frequent than others.
# 
# **Precision (Weighted):** Measures the exactness of the classifier. The weighted precision is calculated for each class (treating it as a binary problem) and then averaged, weighted by the number of instances in each class (support). It is useful when the cost of false positives is high.
# 
# **Recall (Weighted)**: Measures the completeness of the classifier (how many actual positive cases were correctly identified). The weighted recall is similarly averaged across all classes, weighted by each class's support. This is a crucial metric when false negatives are a higher concern than false positives, such as in medical diagnosis.
# 
# **F1 Score (Weighted):** The harmonic mean of precision and recall. The weighted F1 score provides a single balanced measure of performance, accounting for class imbalance through the weighting process.
# 

# In[43]:


label_column = "cancelled"
feature_list = dataframe_return_columns(df_feature_analysis, exclude_columns = [label_column])

train_ratio = 0.7
test_ratio = 1 - train_ratio

for feature in feature_list:

    #create new dataframe consisting of the label and single feature column
    df_single_feature_analysis = df_feature_analysis.select(col(feature),label_column)

    #vectorise single feature
    df_single_feature_analysis = dataframe_create_features_vector(df_single_feature_analysis, [feature], f"{feature}_vector", True)

    #Set the training and testing datframe rowcount ratios used to randomly split dataframe into 2
    seed = 2025

    # Split the Dataframe into training and test  
    df_training, df_testing = df_single_feature_analysis.randomSplit([train_ratio, test_ratio], seed=seed)

    ## Create the logistic regression object for the model to predict the cancelled column
    logistic_regression = LogisticRegression(featuresCol = f"{feature}_vector", labelCol = label_column)

    ## Perform training on the training dataset using logistic regression object to create a logistic regression model
    logistic_regression_model = logistic_regression.fit(df_training)

    df_predictions = logistic_regression_model.transform(df_testing)

    # AUC-ROC
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=label_column)
    auc = evaluator.evaluate(df_predictions)

    # Accuracy, Precision, and Recall
    multi_evaluator = MulticlassClassificationEvaluator(labelCol=label_column, predictionCol="prediction")

    accuracy = multi_evaluator.evaluate(df_predictions, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(df_predictions, {multi_evaluator.metricName: "weightedPrecision"})
    recall = multi_evaluator.evaluate(df_predictions, {multi_evaluator.metricName: "weightedRecall"})
    f1 = multi_evaluator.evaluate(df_predictions, {multi_evaluator.metricName: "f1"})

    ## Plot the ROC curve using model Summary object
    model_summary = logistic_regression_model.summary

    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(model_summary.roc.select('FPR').collect(),
         model_summary.roc.select('TPR').collect())
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Univariant Analysis - ROC Curve for Feature {feature}")
    plt.show()

    #print the metrics
    print("--------------------------------------------------------")
    print(f"{feature} AUC-ROC: {auc:.4f}")
    print(f"{feature} Accuracy: {accuracy:.4f}")
    print(f"{feature} Precision: {precision:.4f}")
    print(f"{feature} Recall: {recall:.4f}")
    print(f"{feature} F1: {f1:.4f}")


# #### **Select Features for Multi Variable Model**
# 
# **The results of the univariant analysis show that none of the features are significant and at this point we would probably look at alternatives. However
# time does not allow this so we will contine to analyse the Multi Variable model using ALL features**
# 
# There are several techniques to improve the performance of models:
# 1. **Vectorisation of features** - Vectorization can signifantly improve the performance of ML tasks reducing the need for iteration over multiple columns. The vector assembler may be used to concatenate all the features into a single vector.
# 2. **Normalisation of numeric features** - This is generally recommended in logistic regression when the features have vastly different scales. Although simple logistic regression without regularisation () does not require normalisation mathematically because the co-efficients adjust to the scale of the data.
# 
# Normalisation is recommended for the folowing reasons:
# - Faster Convergence: Algorithms that use gradient descent (common for training logistic regression models) converge much faster when features are on a similar scale.
# - Regularization: If using regularization (L1 or L2, also known as Lasso or Ridge regression), scaling is mandatory. Regularization penalizes large coefficients, and without scaling, features with larger natural scales would be unfairly penalized.
# - Interpretability: Scaling allows for a more meaningful comparison of the coefficients or the relative importance of the features as they are all on the same scale.
# - Numerical Stability: It helps with numerical stability during the computation process, especially with very large numbers, preventing potential issues during modelling. 
#   Pyspark Mllib has the **StandardScaler** function to normalise or scale numeric values. [https://spark.apache.org/docs/latest/mllib-feature-extraction.html](link-URL)
# 
# 3. **Hyper-Tuning Parameters** - These parameters can improve the performance of the model and should be adjusted once a satfisfactory outcome has been obtained to hypertune the model.
#   - maxIter - Sets the maximum number of iterations for the optimization solver (i.e L-BFGS) to converge.
#   - regParam - The regularization parameter (lambda) that controls the penalty strength. Higher values lead to more regularization, which helps prevent overfitting by shrinking coefficient values.
# - elasticNetParam - The ElasticNet mixing parameter (alpha), which balances L1 (Lasso) and L2 (Ridge) regularization:(alpha=0): Pure L2 regularization (Ridge)
# (alpha =1): Pure L1 regularization (Lasso) (alpha =0.8): A combination where 80% of the penalty is L1 and 20% is L2, making the model favor simpler solutions (those with less features) while still providing some coefficient shrinkage.
# 
# 4. **Use of a 5-Fold Cross Validator** - This technique involves partitioning the dataset randomly into five equal subsets so that 5 distinct iterations (or experiments) are performed on each subset. In each iteration: Four parts (80% of the data) are combined to train the machine learning model. The remaining (20% of the data) is held out and used as the validation/test set to evaluate the model's performance. The performance metrics(accuracy, precision etc) from all five evaluations are then averaged to produce a single, more robust estimate of the model's overall performance and stability. 
# Benefits
# Robust Evaluation: It provides a more reliable estimate of a model's true performance compared to a single train-test split, as it reduces the bias that might come from a particularly good or bad random split.
# Efficient Data Usage: Every data point gets to be in a training set (four times) and a test set (once), making maximum use of a limited dataset.
# Overfitting Prevention: It helps in detecting and mitigating overfitting by testing the model against various subsets of the data
# 
# 
# These techniques will be applied to the multi variable analysis.
# 

# In[19]:


label_column = "cancelled"
feature_column  = "feature_columns_vector"
df_feature_analysis = df_analysis.select("days_installed","orderatom_service_install_cost","orderatom_total_annual_recurr_cost",\
                                        "number_incidents","number_complaints","crs_score","cancelled")

#fill nulls with zeros 
df_feature_analysis = df_feature_analysis.fillna(value=0)

#vectorise the feature columns in df_feature_analysis - no need to scale
feature_columns = [x.name for x in df_feature_analysis.schema if x.name != label_column]
df_feature_analysis = dataframe_create_features_vector(df_feature_analysis, feature_columns, "feature_columns_vector", True)

df_feature_analysis = df_feature_analysis.select("feature_columns_vector","cancelled")

#Set the training and testing datframe rowcount ratios used to randomly split dataframe into 2
train_ratio = 0.8
test_ratio = 1 - train_ratio
seed = 2809

# Split the Dataframe into training and test  
df_training, df_testing = df_feature_analysis.randomSplit([train_ratio, test_ratio], seed=seed)


# ### **1.Pyspark Logistic Regression**

# In[ ]:


# Create the logistic regression object for the model to predict the cancelled column
logistic_regression = LogisticRegression(featuresCol = feature_column, labelCol = label_column)

# Create a parameter grid for hyper tuning the model
logreg_tuning_parameters = (ParamGridBuilder().addGrid(logistic_regression.regParam, [0.01, 0.1, 0.5, 1.0, 2.0])\
                                              .addGrid(logistic_regression.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0])\
                                              .addGrid(logistic_regression.maxIter, [1, 5, 10, 20, 50]).build())

# Define the Logistic Regression evaluator
logreg_evaluator = BinaryClassificationEvaluator(labelCol=label_column, rawPredictionCol="rawPrediction", metricName = "areaUnderROC")

# Create 5-fold CrossValidator
logreg_cross_validator = CrossValidator(estimator = logistic_regression, estimatorParamMaps = logreg_tuning_parameters,\
                                        evaluator = logreg_evaluator, numFolds = 5)

# Train the model on the training dataset
logistic_regression_model = logreg_cross_validator.fit(df_training)

# Predict on test data
df_prediction = logistic_regression_model.transform(df_testing)

# Compute the evaluation metric area under ROC
aroc = logreg_evaluator.evaluate(df_prediction)
print(f"Logistic Regression Area Under ROC on test data: {aroc:.4f}")

# Calculate evaluation metrics and confusion matrix
dataframe_evaluation_metrics(df_prediction,"prediction", label_column, ml_model=logistic_regression_model)

# Plot the ROC curve using model Summary object
model_summary = logistic_regression_model.bestModel.summary

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(model_summary.roc.select('FPR').collect(),
model_summary.roc.select('TPR').collect())
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f"ROC Curve")
plt.show()

# Display the model co-efficients and intercept
print("Coefficients (Weights):", logistic_regression_model.bestModel.coefficients)
print("Intercept:", logistic_regression_model.bestModel.intercept)

# Save the model
save_model_azure_storage(folder_name="pyspark-logistic-regression", ml_model=logistic_regression_model)


# ### **2. Pyspark Decision Tree Classifier**
# Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multi-class classification, do not require feature scaling, and are able to capture non-linearities and feature interactions.

# In[ ]:


# Create the decision_tree object for the model 
decision_tree_classifier = DecisionTreeClassifier(featuresCol=feature_column, labelCol=label_column)

# Create a parameter grid for hyper tuning the model
dt_tuning_parameters = (ParamGridBuilder().addGrid(decision_tree_classifier.maxDepth, [2, 5, 10])\
                                          .addGrid(decision_tree_classifier.maxBins, [10, 20]).build())
# Define the evaluator
dt_evaluator = BinaryClassificationEvaluator(labelCol=label_column, rawPredictionCol="rawPrediction", metricName = "areaUnderROC")

# Create 5-fold Cross Validator
dt_cross_validator = CrossValidator(estimator=decision_tree_classifier, estimatorParamMaps=dt_tuning_parameters,\
                                      evaluator=dt_evaluator, numFolds = 5)

# Train the model
decision_tree_model = dt_cross_validator.fit(df_training)

# Predict on test data
df_prediction = decision_tree_model.transform(df_testing)

# Compute the evaluation metric area under ROC
aroc = logreg_evaluator.evaluate(df_prediction)
print(f"Decision Trees Classifier Area Under ROC on test data: {aroc:.4f}")

# Calculate evaluation metrics and confusion matrix
dataframe_evaluation_metrics(df_prediction,"prediction", label_column, ml_model=decision_tree_model)

# Save the model azure storage
save_model_azure_storage(folder_name="pyspark-decision-trees-classifier", ml_model=gbdt_model)


# In[88]:


# Calculate evaluation metrics and confusion matrix
dataframe_evaluation_metrics(df_prediction,"prediction", label_column, ml_model=decision_tree_model)


# ### **3. Random Forest Classifier**
# 

# In[89]:


# Create the random Forest Classifier Object for the model
random_forest_classifier = RandomForestClassifier(labelCol=label_column, featuresCol=feature_column)

# Create a parameter grid for hyper tuning the model
rf_tuning_parameters = (ParamGridBuilder().addGrid(random_forest_classifier.maxDepth, [2, 5, 10])\
                                          .addGrid(random_forest_classifier.maxBins, [5, 10, 20])\
                                          .addGrid(random_forest_classifier.numTrees, [5, 20, 50]).build())

# Define the random forest evaluator
rf_evaluator = BinaryClassificationEvaluator(labelCol=label_column, rawPredictionCol="rawPrediction", metricName = "areaUnderROC")

# Create 5-fold Cross Validator
rf_cross_validator = CrossValidator(estimator=random_forest_classifier, estimatorParamMaps=rf_tuning_parameters,\
                                    evaluator=rf_evaluator, numFolds = 5)
# Train the model
random_forest_model = rf_cross_validator.fit(df_training)

# Predict on test data
df_prediction = random_forest_model.transform(df_testing)

# Compute the evaluation metric area under ROC
aroc = logreg_evaluator.evaluate(df_prediction)
print(f"Random Forest Classifier Area Under ROC on test data: {aroc:.4f}")

# Calculate evaluation metrics and confusion matrix
dataframe_evaluation_metrics(df_prediction,"prediction", label_column, ml_model=random_forest_model)

# Save the model azure storage
save_model_azure_storage(folder_name="pyspark-random-forest-classifier", ml_model=random_forest_model)


# ### **4. Pyspark Gradient Boosted Decision Trees Classifier**
# 
# **Decision trees** create a model that predicts the label by evaluating a tree of if-then-else true/false feature questions and estimating the minimum number of questions needed to assess the probability of making a correct decision. Decision trees can be used for classification to predict a category, or regression to predict a continuous numeric value.
# 
# **Random forest** uses a technique called bagging to build full decision trees in parallel from random bootstrap samples of the data set. The final prediction is an average of all of the decision tree predictions.
# 
# The term **gradient boosting** comes from the idea of boosting or improving a single weak model by combining it with a number of other weak models to generate a collectively strong model. Gradient boosting is an extension of boosting where the process of additively generating weak models is formalized as a gradient descent algorithm over an objective function. Gradient boosting sets targeted outcomes for the next model in an effort to minimize errors. Targeted outcomes for each case are based on the gradient of the error with respect to the prediction.
# 
# GBDTs iteratively train an ensemble of shallow decision trees, with each iteration using the error residuals of the previous model to fit the next model. The final prediction is a weighted sum of all of the tree predictions. Random forest bagging minimizes the variance and overfitting, while GBDT boosting minimizes the bias and underfitting.
# 
# XGBoost is a scalable and highly accurate implementation of gradient boosting that pushes the limits of computing power for boosted tree algorithms, being built largely for energizing machine learning model performance and computational speed. With XGBoost, trees are built in parallel, instead of sequentially like GBDT. It follows a level-wise strategy, scanning across gradient values and using these partial sums to evaluate the quality of splits at every possible split in the training set. 
# 

# In[20]:


# Create Gradient Boosted Decision Trees Classifier model
gbdt_classifier = GBTClassifier(labelCol=label_column, featuresCol=feature_column)

# Create a parameter grid for tuning the model
gbdt_tuning_parameters = (ParamGridBuilder().addGrid(gbdt_classifier.maxDepth, [2, 5, 10]).addGrid(gbdt_classifier.maxBins, [10, 20, 40])\
                                            .addGrid(gbdt_classifier.maxIter, [5, 10, 20]).build())

# Define the Gradient Boosted Decision Trees evaluator
gbdt_evaluator = BinaryClassificationEvaluator(labelCol=label_column, rawPredictionCol="rawPrediction", metricName = "areaUnderROC")

# Create 5-fold Cross Validator
gbdt_cross_validator = CrossValidator(estimator = gbdt_classifier, estimatorParamMaps = gbdt_tuning_parameters, evaluator = gbdt_evaluator, numFolds = 5)

# Train the model
gbdt_model = gbdt_cross_validator.fit(df_training)

# Predict on test data
df_prediction = gbdt_model.transform(df_testing)

# Compute the evaluation metric rmse
#rmse = evaluator.evaluate(df_prediction)
#print(f"Gradient Boosted Decision Trees Classifier Root Mean Squared Error (RMSE) on test data: {rmse:.4f}")

# Compute the evaluation metric area under ROC
aroc = gbdt_evaluator.evaluate(df_prediction)
print(f"Gradient Boosted Decision Trees Classifier Area Under ROC on test data: {aroc:.4f}")

# Calculate evaluation metrics and confusion matrix
dataframe_evaluation_metrics(df_prediction,"prediction", label_column, ml_model=gbdt_model)

# Save the model azure storage
save_model_azure_storage(folder_name="pyspark-gradient-boosted-decision-trees-classifier", ml_model=gbdt_model)

