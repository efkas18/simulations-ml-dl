# Medical Cost Personal Datasets

## Summary
A case study which refers to a regression approach. The dataset contains general information from personal of an insurance,
and based on features of each observation there is corresponding charge.

An ANN model was build for regression, contains 3 hidden layers with multiple units. As output there is a single unit,  because the goal is to predict a continuous number which represents the cost of charge.
There is a history plot of loss curve, which decreasing as the training of the model evolves. Additionally, at the training stage an early stopping callback was called, if there is not a significant improvement.  
 
The Mean Absolute Error as a validation metric was chosen, and after the testing completed the Mean Absolute Percentage Error was calculated to evaluate the actual success of neural network.

### Database of dataset:
> https://www.kaggle.com/datasets/mirichoi0218/insurance

### Raw URL of dataset:
> https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv
