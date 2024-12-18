import pandas as pd

# Load project data
data = pd.read_csv('data/sample_project_data.csv')

# Clean data
data['Cost'] = data['Cost'].fillna(data['Cost'].mean())
data['Completion_Percentage'] = data['Completed_Days'] / data['Total_Days']

# Generate real-time project updates
progress_report = data[['Project_ID', 'Milestone', 'Completion_Percentage']].copy()
progress_report.to_csv('docs/progress_report_sample.csv', index=False)
print("Project progress report generated!")

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load project data
data = pd.read_csv('data/sample_project_data.csv')

# Prepare data for cost estimation
X = data[['Area', 'Material_Quantity', 'Labor_Hours']]
y = data['Cost']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict costs for new projects
predictions = model.predict(X_test)
pd.DataFrame({'Actual': y_test, 'Predicted': predictions}).to_excel('data/cost_estimation_model_results.xlsx')
print("Cost estimation results saved!")
