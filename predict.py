import argparse
import numpy
import torch
from torch import nn, tensor, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='flowers/test/16/image_06670.jpg', help='path to input image')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='trained model checkpoint image path')
parser.add_argument('--top_k', type=int, default=5, help='number of most likely results returned')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of output results to flower names')
parser.add_argument('--gpu', type=str, default='gpu', help='use GPU for inference')

in_arg = parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    transformations = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalization])
    image_tensor = transformations(pil_img)
    return image_tensor    
    
    
def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model = getattr(torchvision.models, checkpoint['model architecture'](pretrained = True))
    model.clasifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.model_state_dict = checkpoint['model_state_dict']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = check['model_class_to_idx']
    return model, optimizer
    
    
def predict(image_path, checkpoint_path, topk=5):
    cuda = torch.cuda.is_available()
    model, optimizer = load_checkpoint(checkpoint_path)
    image = process_image(image_path)
    if cuda:
        model.cuda()
        image = image.cuda()
    else:
        model.cpu()
    
    model.eval() # setting model to evaluation mode
    image_tensor = torch.from_nump(np.array([image])).float()
    output = model.forward(image_tensor)
    probabilities = torch.exp(output).data
    # getting top five probabilities and indexes
    top_probs, top_labels = probabilities.topk(top_k)
    top_probs = top_probs.tolist()[0]
    top_labels = top_probs.tolist()[0]
    return_classes = []
    for label in top_labels:
        return_classes.append(label)
        
    return top_probs, return_classes
    

def print_probabilities(probs, flowers, topk):
    for i, (flower_name, probability) in enumerate(zip(flowers, probs)):
        print("Rank {} out of {} Flower Name: {} Probability: {}".format(i, topk, flower_name, probability))
       

def main():
    output_probabilities, flower_classes = predict(in_arg.image, in_arg.checkpoint, in_arg.top_k)
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    flower_classes = [cat_to_name[i] for i in flower_classes]
    print_probabilities(output_probabilities, flower_classes, in_arg.topk)

if __name__ == '__main__':
    main()
    