import os
import pandas as pd

import os
import pandas as pd

def Summary_Cross_Validation(path):
    '''
    This function reads the scores from the cross-validation and outputs a summary of the RMSE scores.
    
    The folders in the path should be named as follows:
    'Model_ModelType_DataType'
    
    Each folder should contain a file named 'Scores.csv' with the RMSE scores.
    This function is created by the 'Modules/Cross_Validation.py' script.

    Parameters:
    path (str): The path to the directory containing the cross-validation results.

    Returns:
    pd.DataFrame: A DataFrame summarizing the RMSE scores with columns 'Model', 'Data', and 'RMSE'.

    Output:
    A CSV file named 'Summary_Scores.csv' is saved in the provided path directory, containing the summarized RMSE scores.
    '''
    output_scores = {'Model':[], 'Flower':[], 'Data':[], 'RMSE':[]}

    # Traverse the directory structure starting from the provided path
    for folder in os.walk(path):
        try:
            folder_string = folder[0].split('/')[-1]  # Get the folder name
            model = folder_string.split('_')[1]  # Extract the model type
            flower = folder_string.split('_')[2]  # Extract
            data = ''.join(s for s in folder_string.split('_')[3:])  # Extract and concatenate the data type parts

            # Process each file in the current folder
            for file in folder[2]:
                if file.startswith('Scores') and file.endswith('.csv'):
                    # Read the scores from the CSV file
                    scores = pd.read_csv(os.path.join(folder[0], file))
                    # Append the scores to the output dictionary
                    for i in range(len(scores)):
                        output_scores['Model'].append(model)
                        output_scores['Flower'].append(flower)
                        output_scores['Data'].append(data)
                        output_scores['RMSE'].append(scores['RMSE'][i])
        except:
            # Skip the folder if any error occurs
            continue
    
    # Convert the output dictionary to a DataFrame
    output_scores = pd.DataFrame(output_scores)
    # Save the summary DataFrame to a CSV file
    output_scores.to_csv(os.path.join(path, 'Summary_Scores.csv'), index=False)
    
    return output_scores


if __name__ == '__main__':
   summary_scores = Summary_Cross_Validation(path='/home/u108-n256/PalmProject/CrossValidation')
   print('s_saved')
   print(summary_scores.groupby(['Model','Flower', 'Data']).mean().sort_values(by='RMSE'))