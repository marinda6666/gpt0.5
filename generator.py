import sys
import torch 
import random
from train import create_train_model

SAMPLE_LEN = 1000
SEED = random.randint(0, 100)

class GeneratorUsageException(Exception):
    def __init__(self, message='generator take two argument (checkpoint of the model and the file that the model was trained on)'):
        self.message = message
        super().__init__(message)

# run generation on Tolstoi text -> $ python3 generator.py models/tolstoi.pth data/tolstoi_l_n__voina_i_mir.txt
def main():
    argv = sys.argv 

    if (len(argv) != 3):
        raise GeneratorUsageException()

    checkpoint = argv[1]
    filename = argv[2]

    try:
        print(f'\nPreparing {filename}...')
        model, _, _ = create_train_model(filename=filename, seed=SEED)

        print(f'Loading {checkpoint} for model...\n')
        model.load_state_dict(torch.load(checkpoint, weights_only=True))

        model.eval()

        print(f'GPT 0.5 ({checkpoint}):\n')
        model.generate(len=SAMPLE_LEN).data[0]
        print('\n\n')
    except:
        raise GeneratorUsageException()
    
    

if __name__ == '__main__':
    main()