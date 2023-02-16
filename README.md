![This is an image](https://assets1.ignimgs.com/thumbs/userUploaded/2019/5/29/dndmobilebr-1559158864269.jpg)

#  Predicting Wisdom in Dungeons and Dragons

# Project Goals

     - Find the best drivers for the Wisdom stat
     - Use the drivers for modeling

# Project Description

In real life there is no set way to predict if someone has wisdom. Wether it be age, experience, nothing is truly reliable. In this
project I will be using machine learning to try and predict the wisdom stat on DnD characters. I will do this by first finding what drives wisdom, then
using those drivers in my machine learning models.

# Initial Questions

 1. Do characters who have a higher stat of height or weight have a higher wisdom stat?
 2. Do characters having above average stats have higher wisdom?
 3. Do characters having both high intelligence and dexterity have greater wisdom?
 4. Does being a certain race mean higher wisdom?


# The Plan

 - Create README with project goals, project description, initial hypotheses, planning of project, data dictionary, and come up with recommedations/takeaways

### Acquire Data
 - Acquire data from data.world and create a function to later import the data into a juptyer notebook to run our notebook faster. (wrangle.py)

### Prepare Data
 - Clean and prepare the data creating a function that will give me data that is ready to be explored upon. Within this step we will also write a function to split our data into train, validate, and test. (wrangle) 
 
### Explore Data
- Create visuals on our data 

- Use clustering techniques to observe if there are any insights we can gather from our data

- Create at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, document any findings and takeaways that are observed.

### Feature Engineering:
 - Scale our data for our models
 
 - Create dummies to encode categorical variables for our models (if needed)

### Model Data 
 - Establish a baseline(mean or median of target variable)
 
 - Create, Fit, Predict on train subset on four regression models.
 
 - Evaluate models on train and validate datasets.
 
 - Evaluate which model performs the best and on that model use the test data subset.
 
### Delivery  
 - Create a Final Report Notebook to document conclusions, takeaways, and next steps in recommadations for predicitng wine quality. Also, inlcude visualizations to help explain why the model that was selected is the best to better help the viewer understand. 


## Data Dictionary


| Target Variable |     Definition     |
| --------------- | ------------------ |
|      Wisdom      | The wisdom level of a character in a range from 1-20  |

| Feature  | Definition |
| ------------- | ------------- |
| Race| The race of the character |
| Height| The height of a character in a measurment of inches |
| Weight | The weight of a character in a measurment of lbs. |
| Speed | The speed level of a character in a range from 1-20  |
| Strength| The strength level of a character in a range from 1-20  |
| Dexterity | The dexterity level of a character in a range from 1-20  | 
| Constitution| The constitution level of a character in a range from 1-20  |
| Wisdom| The wisdom level of a character in a range from 1-20  |
| Intelligene | The intelligence level of a character in a range from 1-20 |
| Charisma | The charisma level of a character in a range from 1-20 |


## Steps to Reproduce

- You will need to download the 1 csv's from data.world (https://data.world/greengabeles/dnd-stats)

- Clone my repo including the wrangle.py, explore.py, and model.py (make sure to create a .gitignore to hide your csv files since it will not be needed to upload those files to github)

- Put the data in a file containing the cloned repo

- Run notebook

## Conclusions
 
Wisdom predictions were used by minimizing RMSE within our models. Race and Intelligence have proven to be the most valuable, but there is still room for improvement.
 
Best Model's performance:

- Our best model was the baseline model

## Recommendations
- We would recommend adding more races to the dataset. 

- We would also recommend collecting more data on what other characters as a whole.

## Next Steps
- Remove outliers, and explore other features,  and use other types of models such as xgbost

- Consider adding different hyperparameters to models for better results. 
