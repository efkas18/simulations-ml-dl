# Banking Dataset - Marketing Targets

## Description
In this case study a dataset of term deposits of a Portuguese banking institute, is consists from observations of clients.  

Every ovservation has 17 Detail Columns representing the client's data.  
The original dataframe is consits from 45.000 observations of clients. Executions have been splitted based on whole dataframe and custom (balanced) dataframe.  

At the first situation, was selected randomly 20% of observations for testing (validation) and the residual for training models.  

At the second situation, was selected whole group of observations by the clients who accept the proposal of the bank's campaing (appox. 4.500, and randomly the same amount of observations selected from clients who declined the proposal. From the new custom dataframe 20% of cases selected randomly for evaluation and 80% for training.  

The flow of simulations starting by making some visualizations from original dataframe, plotting some figures to have an initial idea about the distribution of data.  
The following step is to reduce the dataframe by removing unnecessary columns from dataframe. Then, it follows the convert from string columns to numerical data. After that the separation on training and testing datasets and at the end the feature scaling, by normalizing the values.

### Detailed Column Descriptions
bank client data:

1 - age (numeric)
2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",  
"blue-collar","self-employed","retired","technician","services")  
3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)  
4 - education (categorical: "unknown","secondary","primary","tertiary")  
5 - default: has credit in default? (binary: "yes","no")  
6 - balance: average yearly balance, in euros (numeric)  
7 - housing: has housing loan? (binary: "yes","no")  
8 - loan: has personal loan? (binary: "yes","no")  
#### related with the last contact of the current campaign:  
9 - contact: contact communication type (categorical: "unknown","telephone","cellular")  
10 - day: last contact day of the month (numeric)  
11 - month: last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec")  
12 - duration: last contact duration, in seconds (numeric)  
#### other attributes:  
13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  
14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)  
15 - previous: number of contacts performed before this campaign and for this client (numeric)  
16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")  

Output variable (desired target):  
17 - y - has the client subscribed a term deposit? (binary: "yes","no")  

Missing Attribute Values: None


## Source
> [KAGGLE.COM -> DATASET](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)