import pandas as pd
from joblib import load
from models import annotate

image = r"the corresponding chicken image from the cropped folder"
# Create an Annotator object
annotator = annotate.Annotator(image)
# Perform annotation and masking
annotator.annotate_and_mask()
# Calculate the area of the mask
area = annotator.area()
# Calculate the average length of the top 50 longest horizontal lines
average_length = annotator.length()
# Calculate the average width of the top 50 longest vertical lines
average_width = annotator.width()
# Calculate the perimeter of the masked polygon
perimeter = annotator.perimeter()
# Load the trained model and scaler
model_scaler = load(r'models/regression_model_chicken.joblib')
# Create a DataFrame with scalar values and explicit index
new_data = pd.DataFrame({'area': [area], 'length': [average_length], 'width': [average_width], 'perimeter': [perimeter]}, index=[0])
# Predict using the loaded model and scaler
estimated_weight = model_scaler.predict(new_data)
print(estimated_weight[0])





import pandas as pd
from joblib import load
from models import annotate

image = r"the corresponding pig image from the cropped folder"
# Create an Annotator object
annotator = annotate.Annotator(image)
# Perform annotation and masking
annotator.annotate_and_mask()
# Calculate the area of the mask
area = annotator.area()
# Calculate the average length of the top 50 longest horizontal lines
average_length = annotator.length()
# Calculate the average width of the top 50 longest vertical lines
average_width = annotator.width()
# Calculate the perimeter of the masked polygon
perimeter = annotator.perimeter()
# Load the trained model and scaler
model_scaler = load(r'models/regression_model_pig.joblib')
# Create a DataFrame with scalar values and explicit index
new_data = pd.DataFrame({'area': [area], 'length': [average_length], 'width': [average_width], 'perimeter': [perimeter]}, index=[0])
# Predict using the loaded model and scaler
estimated_weight = model_scaler.predict(new_data)
print(estimated_weight[0])
