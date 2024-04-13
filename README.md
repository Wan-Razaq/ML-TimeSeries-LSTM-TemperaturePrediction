
# Ladle Furnace Unit (LFU) Temperature Prediction - Steel Manufacturing

![Example of LFU](https://images.app.goo.gl/b3LnwrpnXyD956rSA)
*Source: [www.danieli.com]


## Project Domain

In the critical process of steel manufacturing, specifically at the ladle furnace unit (LFU), achieving the correct melt temperature is important for quality and efficiency. The LFU, a vital component in steel processing, faces challenges in accurately forecasting temperatures. Addressing this issue is essential for maintaining the delicate balance of the steel's chemical composition and ensuring optimal processing conditions. The goal of this project is to develop a predictive model that can accurately forecast the temperature measurements of the steel melt, which is a crucial step in improving the overall efficiency and output of the LFU operations.
## Business Understanding

The ladle furnace unit (LFU) is a critical component in steel processing that requires precise temperature control to ensure the quality of the final product. Inconsistent temperature readings due to incomplete data present significant challenges, directly impacting the steel’s chemical integrity and the efficiency of the LFU's operations. Our goal is to develop an accurate temperature prediction model to overcome these inconsistencies, ensuring that target temperatures are reliably achieved for each steel batch processed. To this end, we propose to explore a machine learning algorithms, such as LSTM, to identify the most effective solution for robust temperature forecasting in the LFU.
## Data Understanding

The project utilizes an extensive dataset from the ladle furnace unit (LFU), encompassing various aspects of the steel processing cycle. The dataset includes files such as data_gas.csv for purging gases, data_bulk.csv for bulk elements, data_wire.csv for wire supply, along with time-related data and crucial temperature measurements from data_temp.csv. With missing entries in temperature records posing a challenge, the dataset's robustness is vital for developing an accurate predictive model. Preliminary exploratory data analysis and visualization will be employed to assess data quality, uncover patterns, and understand the correlations between the different technological operations and the temperature outcomes.

The original source of data can be found through this link:
https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-ladlefurnace-unit/data


## Data Preparation

**1. Data Filtering**

I began by importing the **data_temp.csv** file and then focused specifically on the **time** and **Temperature** columns. 


**2. Handling Missing Values**

A check for missing values was conducted, and records containing null entries were removed. This is crucial because missing values can introduce bias or inaccuracies into the predictive model.

**3. Data Visualization**

We visualized the temperature over time using a line plot. This visualization helps in identifying patterns, trends, and potential outliers in the temperature data. 

**4. Data Splitting**

The dataset was split into training and validation datasets with an 80-20 split using the **train_test_split** method. This separation allows us to train the model on a large portion of the data while holding back a portion to validate the model’s performance. This wiill ensure that the model can generalize well to new, unseen data.

**5. Data Structuring for LSTM Model**

I prepared the data for the LSTM model by structuring it into sequences using a windowed dataset function. This function configures the data into batches of sequential time steps with a specified window size, which is critical for time-series forecasting where the sequence of values is important for prediction accuracy.

**6. Establishing Baseline Metrics**

A baseline metric for model evaluation was established by calculating 10% of the temperature scale, which serves as a target for the mean absolute error (MAE) of our model. This metric helps in setting a quantitative goal for model performance and ensures that the model's predictions are within an acceptable error range.




## Modelling

**1. Model Architecture**

I employed a deep learning approach using a Sequential model from TensorFlow's Keras API, designed specifically for time-series forecasting. The model consists of two LSTM layers followed by three Dense layers. The first LSTM layer has 60 units and returns sequences to provide a three-dimensional output that feeds into the next LSTM layer. The subsequent LSTM also has 60 units but does not return sequences, preparing the output for the dense layers. This architecture was chosen to capture the temporal dependencies in temperature changes effectively.

**2. Compilation and Optimization**

The model was compiled using the Huber loss function, which is less sensitive to outliers in data than traditional squared error loss. For optimization, I used SGD (Stochastic Gradient Descent) with a learning rate of 0.0001 and a momentum of 0.9. These parameters help in smoothing out the updates made to the weights and typically result in a more stable convergence.

**3. Model Training and Validation**

Training was conducted over 100 epochs with batch sizes of 100, utilizing both training and validation datasets prepared earlier. This process allows for periodic adjustments to the model's weights and biases and also optimize the prediction accuracy while validating against overfitting.

**4. Hyperparameter Tuning**

Given the complexity of the model and the dataset, tuning the hyperparameters (like learning rate and number of epochs) was crucial. I experimented with different values to find the optimal setting that minimizes the loss and improves the validation accuracy. This will ensure that the model is neither underfitting nor overfitting.

## Evaluation

**1. Evaluation Metrics**

The primary metric used to evaluate the model's performance was the Mean Absolute Error (MAE). MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It's particularly useful in this context because it quantifies the average error in predicted temperature against the actual values.

**2. Model Performance**

Throughout the training process, both training and validation MAE were tracked to gauge the model's performance. Initially, the model exhibited high error rates, which gradually decreased. This indicates that the model was learning from the data. By the end of 100 epochs, the validation MAE stabilized around 14 degrees Celsius. This suggests that, on average, the model's temperature predictions were within 14 degrees of the actual measurements.

**3. Analysis of Results**

The project results indicate that the model has achieved a reasonable level of accuracy in predicting furnace temperatures. However, the MAE of 14 degrees, while substantially improved from the initial stages, still suggests room for improvement given the precision required in industrial settings. The MAE target calculated from the data scale (10% of the temperature range, approximately 51.4 degrees) shows that our model's performance is well within this range, confirming the model's effectiveness but also highlighting potential areas for further refinement to reduce the error margin.






## Conclusion

The evaluation confirms that the LSTM model is capable of forecasting temperatures with an acceptable level of accuracy. Future improvements could include exploring more complex or different types of neural network architectures, incorporating additional features from other related datasets, or further tuning the model's hyperparameters. Additional evaluation metrics like the Root Mean Squared Error (RMSE) or Mean Absolute Percentage Error (MAPE) could also be considered to provide different perspectives on the model's accuracy and reliability.