from parameters import *
from rnn import *

def get_weights(rnn, num_layer):
    return getattr(getattr(rnn, "rnn"+str(num_layer)), 'weight_ih_l0')

def save_state(model, filename):
    checkpoint = {
            'layers': model.cpu().layers,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': model.cpu().optimizer.state_dict()
            }
    torch.save(checkpoint, filename)
    if use_cuda:
        model = model.to("cuda")
    
def load_state(model, filename):
    checkpoint = torch.load(filename)
    
    model = RNN(nb_inputs = nb_input, layers = LAYERS, nb_outputs=nb_output, learning_rate=lr)
    model = model.cpu()
    
    model.layers = checkpoint['layers']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if use_cuda:
        model = model.to("cuda")
    model.eval()
    return model
