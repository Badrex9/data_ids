import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from pyraug.trainers.training_config import TrainingConfig
from pyraug.pipelines.training import TrainingPipeline

from pyraug.pipelines.generation import GenerationPipeline
from pyraug.models import RHVAE
import os


Y_label = 13

dataset_to_augment = np.load('./X_mixed_' + str(Y_label) +'.npy')

torch.cuda.is_available()

config = TrainingConfig(
    output_dir='my_model',
    train_early_stopping=50,
    learning_rate=1e-3,
    batch_size=200, # Set to 200 for demo purposes to speed up (default: 50)
    max_epochs=500 # Set to 500 for demo purposes. Augment this in your case to access to better generative model (default: 20000)
)
torch.manual_seed(8)

# This creates the Pipeline
pipeline = TrainingPipeline(training_config=config)

# This will launch the Pipeline on the data
pipeline(train_data=dataset_to_augment, log_output_dir='output_logs')

last_training = sorted(os.listdir('my_model'))[-1]

# reload the model
model = RHVAE.load_from_folder(os.path.join('my_model', last_training, 'final_model'))

torch.manual_seed(88)
torch.cuda.manual_seed(88)

# This creates the Pipeline
generation_pipe = GenerationPipeline(
    model=model
)

# This will launch the Pipeline
len_generation = 10000
generation_pipe(len_generation)

last_generation = sorted(os.listdir('dummy_output_dir'))[-1]
generated_data = torch.load(os.path.join('dummy_output_dir', last_generation, 'generated_data_500_0.pt'))

for i in range(1,20):
    generated_data = torch.cat((generated_data, torch.load(os.path.join('dummy_output_dir', last_generation, 'generated_data_500_' + str(i) + '.pt'))), 0)
    
np.save('./X_autoencoder_' + str(Y_label) +'.npy', generated_data.view(len_generation, 82, 20).cpu().numpy())
np.save('./Y_autoencoder_' + str(Y_label) +'.npy', [Y_label for i in range(len_generation)])