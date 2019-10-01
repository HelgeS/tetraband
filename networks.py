import copy
import time

import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


def get_model(model_name='squeezenet', dataset='cifar10'):
    if dataset == 'cifar10':
        num_classes = 10
        use_torch_pretrained = False
    elif dataset == 'imagenet':
        num_classes = 1000
        use_torch_pretrained = True
    else:
        raise NotImplementedError("Unknown dataset {}".format(dataset))

    model, input_size = initialize_model(model_name,
                                         num_classes,
                                         feature_extract=False,
                                         use_pretrained=use_torch_pretrained)

    if not use_torch_pretrained:
        model.load_state_dict(torch.load('models/model-{}-{}.pth'.format(model_name, dataset),
                                         map_location='cpu'))

    return model, input_size


def finetrain_model(model_name='squeezenet', dataset='cifar10', num_epochs=20,
                    feature_extract=False, batch_size=16):
    """
    Take a pretrained model (for Imagenet) and finetune it for a different dataset.
    Most code is taken from the Pytorch Finetuning tutorial.
    :param num_epochs:
    :param batch_size:
    :param feature_extract:
    :param model_name:
    :param dataset:
    :param epochs:
    :return:
    """
    if dataset == 'cifar10':
        num_classes = 10
    else:
        raise NotImplementedError("Unknown dataset {}".format(dataset))

    model, input_size = initialize_model(model_name,
                                         num_classes,
                                         feature_extract=False,
                                         use_pretrained=True)
    model = model.to(get_device())

    if dataset == 'cifar10':
        image_datasets = {
            'train': datasets.CIFAR10('cifar10',
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomResizedCrop(input_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                      ])),
            'val': datasets.CIFAR10('cifar10',
                                    train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(input_size),
                                        transforms.CenterCrop(input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])
                                    ]))
        }
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x],
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4) for x in ['train', 'val']
        }
    else:
        raise NotImplementedError("Unknown dataset {}".format(dataset))

    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    model, _ = train_model(model,
                           dataloaders_dict,
                           criterion,
                           optimizer_ft,
                           num_epochs=num_epochs)
    torch.save(model.state_dict(), 'model-{}-{}.pth'.format(model_name, dataset))


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 224

    if model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        if num_classes != 1000:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        if num_classes != 1000:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if num_classes != 1000:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if num_classes != 1000:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if num_classes != 1000:
            model_ft.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1),
                                                     stride=(1, 1))
            model_ft.num_classes = num_classes
    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if num_classes != 1000:
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(get_device())
                labels = labels.to(get_device())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
