#!/usr/bin/env python
# coding: utf-8

# ## data_science_analytics_functions
# 
# Data Science Degree Assignment for Data Analytics module. Data Analytics Functions Library
# 

# ### Functions
# **dataframe_load_from_synapse_database** - Loads data into a dataframe from synapse serverless sql database parameter using the sql query parameter. The JDBC driver connection utilises the synapse linked service and authenticates using the system managed identity
# 
# **dataframe_join** - Joins two spark dataframes based upon common key column.
# 
# **dataframe_drop_rows_null_value** - Drops rows from dataframe where column in list is null.
# 
# **dataframe_drop_duplicate_rows** - Drops duplicate rows from dataframe
# 
# **dataframe_row_count** - Returns rowcount of spark dataframe 
# 
# **dataframe_normalise_column_value** - Returns dataframe with new scaled column containing specified column normalised or scaled
# The naming format of the scaled column is [numeric_column_to_scale]_scaled. Optionally drops original column
# 
# **dataframe_incident_counts_priority_daterange** - Returns dataframe of incident counts for incident priority for orderatoms where the incident was created between orderatom installation date and installation date plus the number of days
# 
# **dataframe_create_features_vector** - Returns dataframe with specified numeric columns replaced by single column of vectorized features with the specified name which defaults to features_vector
# 
# **dataframe_normalise_column_value** - Returns dataframe with new scaled column containing specified column normalised or scaled. The naming format of the scaled column is [numeric_column_to_scale]_scaled
# 
# **dataframe_clean_data** - Returns cleaned dataframe. Drops rows where columns in null_columns list are null. Drops duplicated rows based upon duplicated_column list
# 
# **save_model_azure_storage** - Save the model to the synapse workspace default file system. This is the azure storage account sauks[env]ukdp where env in dev,pro. 
#                             Container ukdp_synapse. The root folder is ukdp-synapse/synapse/workspaces/syn-uks-dev-ukdp/machine-learning-models/[folder_name]
# 
# **dataframe_evaluation_metrics** - Function to evaluate the prediction dataframes confusion matrix metrics and plot the confusion matrix as a heatmap
# 

# In[ ]:


def dataframe_load_from_synapse_database(linkedservice_name, database_name, sql_query):
    """
    Load spark dataframe from synapse serverless sql endpoint using synapse linked service
    Return dataframe from sql query
    """

    server = "syn-uks-dev-ukdp-ondemand.sql.azuresynapse.net"
    port = 1433

    jdbc_url = f"jdbc:sqlserver://{server}:{port};databaseName={database_name};encrypt=true;\
                trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30"
    token = TokenLibrary.getConnectionString(linkedservice_name)

    connection_properties = {
        "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver",
        "accessToken" : token
    }

    df = spark.read.jdbc(url=jdbc_url, table=sql_query, properties=connection_properties)

    return(df)


# In[1]:


def dataframe_join(df1, df2, join_key_column: list, join_type: str = "inner"):
    """
    Join 2 spark dataframes based upon common join_key_key column parameter 
    Return joined dataframe
    """
    key1 = join_key_column[0]
    key2 = join_key_column[1]

    # Perform a join and drop column if key1=key2
    if key1 == key2:
        joined_df = df1.join(df2, on=key1, how=join_type).drop(*key1)
    else:
        joined_df = df1.join(df2, df1[key1] == df2[key2], how=join_type)
    
    return joined_df


# In[ ]:


def dataframe_row_count(df):
    """
    Return rowcount of spark dataframe
    """
    return df.count()


# In[ ]:


def dataframe_drop_rows_null_value(df, column_list: list):
    """
    Drops rows from dataframe where column in list is null
    Return dataframe with rows where columns in column_list are null
    """
    return df.dropna(subset=column_list)


# In[ ]:


def dataframe_drop_duplicate_rows(df, column_list: list):
    """
    Drops duplicate rows from dataframe using the column list to check for duplicates
    """
    return df.dropDuplicates(subset=column_list)


# In[ ]:


def dataframe_incident_counts_priority_daterange(df, incident_priority, number_days:int):
    """
    Return dataframe of incident counts for incident priority for orderatoms where the incident was created between 
    orderatom installation date and installation date + number of days
    """
    if number_days == 0:
        col_alias = f"number_p{incident_priority}_incidents"
        return_df = df.filter(col("incident_priority")==incident_priority).groupBy("orderatom_orderatomid")\
                            .agg(count_distinct("incident_number").alias(col_alias))
    else:
        col_alias = f"number_p{incident_priority}_incidents_{str(number_days)}_days"
        return_df = df.filter((col("incident_created") >= col("orderatom_dateinstalled"))\
                            & (col("incident_created")<=date_add(col("orderatom_dateinstalled"),number_days))\
                            & (col("incident_priority")==incident_priority)).groupBy("orderatom_orderatomid")\
                            .agg(count_distinct("incident_number").alias(col_alias))
          
    return return_df


# In[ ]:


def dataframe_clean_data(df, dataframe_name, null_columns: list, duplicated_columns: list):
    """
    Function to perform basic cleaning of dataframe 
    Drops rows where columns in null_columns list are null
    Drops duplicated rows based upon duplicated_column list
    Returns cleaned dataframe
    """
    print(f"Number of rows {dataframe_name} BEFORE removal of null columns = {dataframe_row_count(df)}")

    #drop rows having null value
    df = dataframe_drop_rows_null_value(df, null_columns)

    print(f"Number of rows {dataframe_name} AFTER removal of null columns = {dataframe_row_count(df)}")

    #drop duplicates
    df = dataframe_drop_duplicate_rows(df, duplicated_columns)

    print(f"Number of rows {dataframe_name} AFTER removal of duplicate columns = {dataframe_row_count(df)}")

    return df


# In[ ]:


def dataframe_create_features_vector(df, numeric_columns_to_vectorize: list, new_column_name: str = "features_vector",\
                                     drop_columns: bool = False):
    """
    Returns dataframe with specified numeric columns replaced by single column of vectorized features
    The new column is called features_vector by default
    """
    vector_assembler = VectorAssembler(inputCols = numeric_columns_to_vectorize,outputCol = new_column_name)
    return_df = vector_assembler.transform(df)
    
    if drop_columns:
        return_df.drop(*numeric_columns_to_vectorize) 
    
    return return_df


# In[ ]:


def dataframe_normalise_column_value(df, vectorised_column_to_scale):
    """
    Returns dataframe with new scaled column containing specified column normalised or scaled
    The naming format of the scaled column is <numeric_column_to_scale>_scaled
    """
    scaled_column_name = f"{vectorised_column_to_scale}_scaled"
    scaler = StandardScaler(inputCol=vectorised_column_to_scale, outputCol=scaled_column_name, withStd=True, withMean=False)
    scaler_model = scaler.fit(df.select(vectorised_column_to_scale))
    df_scaled_column = scaler_model.transform(df)

    return df_scaled_column


# In[1]:


def dataframe_return_columns(df, exclude_columns:list):
    """
    Returns the list of dataframe columns excluding the columns in exclude_columns
    """

    # get a list of feature column names
    column_names = [x.name for x in df.schema if x.name not in exclude_columns]

    return column_names


# In[2]:


def save_model_azure_storage(folder_name, ml_model):
    """
    The model can be saved for future analysis and re-use. To retrieve and use a saved Spark Logistic Regression model, 
    you use the LogisticRegressionModel.load() method in PySpark pointing to the directory where you saved it with model.save(), 
    then apply the loaded model to new data using .transform() to get the predictions.
    Save the model to the synapse workspace default file system. 
    This is the azure storage axcount sauks<env>ukdp where env in dev,pro. Container ukdp_synapse
    The root folder is ukdp-synapse/synapse/workspaces/syn-uks-dev-ukdp/machine-learning-models/
    """
    folder_name = f"/synapse/workspaces/syn-uks-dev-ukdp/machine-learning-models/{folder_name}/"
    date_stamp = datetime.utcnow().strftime('%Y-%m-%d-%s')
    file_name = f"{folder_name}/{folder_name}_{date_stamp}"

    #save model 
    ml_model.save(file_name)


# In[ ]:


def dataframe_evaluation_metrics(df, prediction_column_name, label_column_name, ml_model):
    """
    Function to evaluate the prediction dataframes confusion matrix metrics
    """
    # Calculate true positives, true negatives, false positives, false negatives
    tp = df_prediction.filter((col(label_column_name) == 1) & (col(prediction_column_name) == 1)).count()
    tn = df_prediction.filter((col(label_column_name) == 0) & (col(prediction_column_name) == 0)).count()
    fp = df_prediction.filter((col(label_column_name) == 0) & (col(prediction_column_name) == 1)).count()
    fn = df_prediction.filter((col(label_column_name) == 1) & (col(prediction_column_name) == 0)).count()

    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0  

    # Calculate recall
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0 

    # Calculate F1 measure
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    # Print confusion matrix metrics
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    # Create confusion matrix array
    confusion_matrix = np.array([[tn,fp],[fn,tp]])

    # Display Confusion matric=x using seaborn library
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(confusion_matrix, annot=labels, fmt="", cmap='Blues')

