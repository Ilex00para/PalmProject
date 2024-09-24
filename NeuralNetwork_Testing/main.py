
import NN_Architectures.exclude_Flowers as arch_ex
import NN_Architectures.exclude_Flowers_Bunchload as arch_ex_bl
import NN_Architectures.full_data as arch_full
import torch
import traceback
import datetime
import os
from Functions.Main_Process import main





'''[arch_ex.Flowering_TransformerCNN.TransformerCNN(),
                  arch_ex.Flowering_Transformer.Transformer(),
                  arch_ex.Flowering_CNN.CNN(),
                  arch_ex_bl.Flowering_TransformerCNN.TransformerCNN(),
                  arch_ex_bl.Flowering_Transformer.Transformer(),
                  arch_ex_bl.Flowering_CNN.CNN(),
                  arch_full.Flowering_TransformerCNN.TransformerCNN(),
                  arch_full.Flowering_Transformer.Transformer(),
                  arch_full.Flowering_CNN.CNN()]'''


for flower in ['male']:
    for data_id, model in enumerate([arch_full.Flowering_TransformerCNN_L1.L_TCNN1()]):
        try:
            '''Model (imported from NN_Architectures as arch)'''
            model =  model
            flower = flower
            #Model to be trained


            #model = torch.load('/home/u108-n256/PalmProject/NeuralNetwork_Testing/Saved_Objects/BEST_MODELS/Male_Prediction/Model_Transformer_SHALLOW_Include/Long_Trained_Model.pt') #Load the model from a saved model


            """Hyperparameters"""
            #self explaining hyperparameters
            batch_size = 64
            epochs = 3000


            patience = 200 #Patience for early stopping, if improvement of validation loss is not seen for patience epochs, training is stopped
            best_mse = 1000000 #Initial value for validation loss


            weight_decay = 10e-3 #L2 regularization for the optimizer

            """Learning rate scheduler"""

            scheduler_type = 'cyclical' #Either 'plateau' or 'exponential' or 'cyclical'


            """Cyclical: Learning rate is cycled between a minimum and maximum value"""

            min_lr = 10e-9 #Minimum learning rate
            max_lr = 10e-8 #Maximum learning rate

            ''' 
            Step size for the learning rate cycle https://arxiv.org/abs/1506.01186; 

            samples training data set: 212 trees x 131 samples = 27.772,0 
            iterations = samples training data set/batch size
            recommended step size = (2,3 or 4) x iterations
            '''
            step_size =  int(25 * (27772/batch_size))

            mode = 'triangular2' #Mode for the learning rate cycle


            """Main function"""

            if __name__ == '__main__':

                #Creating teh folder to save outputs of the script
                script_path = os.path.abspath(__file__)
                script_dir = os.path.dirname(script_path)
                os.chdir(script_dir) #change dir to the dir where the file is located...

                main(model=model,
                    
                    epochs=epochs, 
                    batch_size=batch_size,

                    min_lr=min_lr,
                    max_lr=max_lr,
                    
                    mode=mode, 
                    step_size=step_size,
                    weight_decay=weight_decay,

                    scheduler_type=scheduler_type,
                    patience=patience,
                    flower=flower,
                    
                    data_id=data_id)
                
        except Exception as e:
            with open('error_log.txt', 'a') as f:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Time: {current_time}\n")
                f.write(f"Model: {model.__class__}\n")
                f.write(f"Flower: {flower}\n")
                f.write(f"Error message: {str(e)}\n")
                f.write("Full traceback:\n")
                f.write(traceback.format_exc())
                f.write("\n\n")  # Add some space between error logs
            continue