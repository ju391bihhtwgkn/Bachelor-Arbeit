import pandas as pd
import matplotlib.pyplot as plt
import glob

# Function to calculate the averages for each retriever from a CSV file
def calculate_average_from_csv(file_path):
    # Load CSV file
    df = pd.read_csv(file_path)
    
    # Select the columns for which you want to calculate the average
    score_columns = ['BERT_precision','BERT_recall','BERT_F1', 'BLEURT-Score', 'ragas_answer_correctness', 'ragas_answer_relevancy','ragas_context_precision','ragas_context_recall','ragas_faithfulness',
                      'llm_accuracy', 'llm_relevance','llm_completeness','llm_overall']
    
    # Calculate the average for the selected score columns
    averages = df[score_columns].mean()
    
    # Extract the retriever type from the 'typ' column (assuming it's the same for all rows)
    retriever_type = df['typ'].iloc[0]  # Assuming all rows belong to the same retriever in one CSV
    
    return retriever_type, averages

# List to store the averages of all retrievers
retriever_averages = {}

# Path to the folder where all your CSV files are stored
csv_folder_path = '../evaluation_results/*.csv'  # Replace with the actual folder path

# Load and process all CSV files
for file in glob.glob(csv_folder_path):
    retriever_type, averages = calculate_average_from_csv(file)
    retriever_averages[retriever_type] = averages

# Create a DataFrame from the retriever averages
retriever_df = pd.DataFrame(retriever_averages).T  # Transpose to make retrievers the index

# Plot the bar chart for the average scores of each retriever
retriever_df.plot(kind='bar', figsize=(12, 8))

plt.xticks(rotation=45, ha='right')

# Add labels and title
plt.title('')
plt.xlabel('Retriever')
plt.ylabel('Average Score')

# Display the plot
plt.legend(loc='upper right')
plt.show()