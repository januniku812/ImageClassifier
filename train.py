import argparse
import numpy
import torch
from torch import nn, tensor, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import time
from PIL import Images

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='flowers', help='directory to flower images')
parser.add_argument('--checkpoint_save_dir', type=str, default='checkpoint.pth', help='trained model checkpoint')
parser.add_argument('--arch', type=str, default='vgg19', help='desired model architecture, choose from: vgg19, vgg16, vgg13, alexnet, densenet121')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--gpu', type=str, default='gpu', help='user GPU for inference and computations')
parser.add_argument('--hidden_units', type=int, default=1024, help='number of nuerons/units in hidden layer')
parser.add_argument('--learning_rate', type=int, default=0.001, help='learning rate of model')
in_arg = parser.parse_args()

def return_loaders_and_training_data(train_dir, valid_dir, test_dir):
   # TODO: Define your transforms for the training, validation, and testing sets
    normalization = transforms.Normalize([0,485, 0.456, 0.406], [0.229, 0.224, 0.225]) # creating a variable to hold transforms normalize
    # tranining data transforms - has random rotation and flipping to better train our model
    training_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalization]) 

    # validation data transforms - no need for rotation and flippimg
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               normalization])

    # testing data transforms - no need for rotation and flippimg
    testing_transforms = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               normalization])


    # TODO: Load the datasets with ImageFolder
    training_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_datasets = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(training_datasets, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(validation_datasets, batch_size=32)
    test_loader = torch.utils.data.DataLoader(testing_datasets, batch_size=32)
    return train_loader, valid_loader, test_loader, training_datasets

def save_checkpoint(model, optimizer, epochs, training_datasets, hidden_layer_units, arch, learning_rate, dropout, input_size, output_size, path='checkpoint.pth'):
    model.class_to_idx = training_datasets.class_to_idx
    torch.save({'input_size': input_size,
                'hidden layer': hidden_layer_units,
                 'output_size': output_size,
                'dropout': dropout,
                 'epochs': epochs,
                 'model architecture': arch,
                'optimizer': optimizer.state_dict(),
                 'learning_rate': learning_rate,
                 'model_state_dict': model.state_dict(),
                 'model_class_to_idx': model.class_to_idx}, path)
    

def create_model(arch='vgg19', hidden_units='1024', learnrate=0.001, device='gpu'):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True) # minimum input size is 32 x32 
        input_units = 25088
        input_units 
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True) # minimum input size is 32 x 32
        input_units = 25088
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True) # minimum input size is 32x32
        input_units = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_units = 9216 # found input size from pytorch documentation on github
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True) # minimum input size is 29x29
        input_units = 1024
    else:
        print("Unknown model: {}".format(arch))
    
    model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units), nn.Dropout(0.5), nn.ReLU(), nn.Linear(hidden_units, 10), nn.LogSoftmax())
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    device = torch.device('cuda' if torch.cuda.is_available() and device=='gpu' else 'cpu')
    model.to(device)
    return model, optimizer, input_units
   
      
    
def train(trainloader, validloader, epochs, model, criterion, optimizer, print_every=10):
    # actually training data
    epochs = 10
    cuda = torch.cuda.is_available()
    print("IS CUDA AVAILABLE : {}".format(cuda))
    if cuda:
        model.cuda()
    else:
        model.cpu()
    start = time.time()
    print("Training started at : {}".format(start))
    for e in range(epochs):
        print("Running Epoch {} out of {}".format(e+1, epochs))
        model.train() # telling the model we are training
        training_loss = 0
        training_accuracy = 0
        steps = 0
        for i, (images, labels) in enumerate(trainloader): # getting a batch of our training data at a time
            steps += 1
            print("Running Batch {} from training data".format(i+1))
            if cuda: # moving traingin data to cuda/GPU if GPU available
                images = images.to('cuda')
                labels = labels.to('cuda')

            #cleaning all previous gradient
            optimizer.zero_grad()
            # forward pass - should go through our sequential model's 3 layers
            outputs = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step() # taking a step "down" the error gradient
            training_loss += loss.item()
            
            # training validation every 10 steps
            if steps % print_every == 10:
                model.eval()
                # validation without gradient
                with torch.no_grad():
                    for images, labels in enumerate(validloader):
                        if cuda:
                            images, labels = images.to('cuda'), labels.to('cuda')
                        log_probabilities = model.forward(images)
                        loss = criterion(log_probabilities, labels)
                        validation_loss = loss.item()
                        probabilities = torch.exp(log_probabilities)
                        top_probability, top_class = probabilities.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor))
            print("Epoch {} out of {}, Training loss: {}, Validation Loss {}, Validation Accuracy: {}".format(e+1, epochs, training_loss, validation_loss, validation_accuracy))
                      
            probabilities = torch.exp(output).data # raising e to the outputs to get our actual output data
            equality = (labels.data == probabilities.max(1)[1])
            training_accuracy = equality.type_as(torch.cuda.FloatTensor().mean())
            print("Batch no: {}, Training Loss for epoch: {}, Training Accuracy for Epoch: {}".format(i, training_loss, training_accuracy))

    end_time = time.time()
    print("Training finished at: {}".format(end_time))
    print("Total time to run through {} epochs: {}".format(epochs, start_time-end_time))

def main():
    train_loader, valid_loader, test_loader, training_datasets = return_loaders_and_training_data(in_arg.data_dir + '/train', in_arg.data_dir + '/valid', in_arg.data_dir + '/test')
    model, optimizer, input_units = create_model(in_arg.arch, in_arg.hidden_units, in_arg.learning_rate, in_arg.gpu)
    criterion = nn.NLLLoss() # using negative log likelihood loss since we are dealing with probabilities
    train(train_loader, valid_loader, in_arg.epochs, model, criterion, optimizer, 10)
    save_checkpoint(model, optimizer, in_arg.epochs, training_datasets, in_arg.hidden_units, in_arg.arch, in_arg.learning_rate, 0.5, input_units, 10, in_arg.checkpoint_save_dir)

if __name__=='__main__':
    main()
    
    
