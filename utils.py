import torch
import torch.nn.functional as F
from torchsummary import summary
from typing import List
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


#########################################################################################################################################################

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

#########################################################################################################################################################

def load_weights_from_path(model, path):
    """load weights from file

    Args:
        model (Net): Model instance
        path (str): Path to weights file

    Returns:
        Net: loaded model
    """
    model.load_state_dict(torch.load(path))
    return model

#########################################################################################################################################################

def get_incorrrect_predictions(model, loader, device="cpu", lightning_mode=False):
    """Get all incorrect predictions

    Args:
        model (Net): Trained model
        loader (DataLoader): instance of data loader
        device (str): Which device to use cuda/cpu

    Returns:
        list: list of all incorrect predictions and their corresponding details
    """
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            if not lightning_mode:
                data, target = data.to(device), target.to(device)
            output = model(data)
            #loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return incorrect

#########################################################################################################################################################

def get_all_predictions(model, loader, device):
    """Get All predictions for model

    Args:
        model (Net): Trained Model 
        loader (Dataloader): instance of dataloader
        device (str): Which device to use cuda/cpu

    Returns:
        tuple: all predicted values and their targets
    """
    model.eval()
    all_preds = torch.tensor([]).to(device)
    all_targets = torch.tensor([]).to(device)
    with torch.no_grad():
        for data, target in loader:
            data, targets = data.to(device), target.to(device)
            all_targets = torch.cat(
                (all_targets, targets),
                dim=0
            )
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds = torch.cat(
                (all_preds, preds),
                dim=0
            )

    return all_preds, all_targets

#########################################################################################################################################################

def get_device() -> tuple:
    """
    Get Device type

    Returns:
        tuple: cuda:bool, mps:bool device
    """
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    return (use_cuda, use_mps, device)

#########################################################################################################################################################

def model_summary(model, input_size=(3, 32, 32)):
    """
    Print Model summary

    Args:
        model (Net): Model Instance
        input_size (tuple, optional): Input size. Defaults to (3, 32, 32).
    """
    summary(model, input_size=input_size)

#########################################################################################################################################################

def save_best(valid_correct, test_loader, model, best_perc):
        """
        Save the best model based on validation accuracy

        Args:
            valid_correct (int): number of correct predictions
        """
        valid_perc = (100. * valid_correct / len(test_loader.dataset))

        if valid_perc >= best_perc:
            best_path = f'trained_model_{valid_perc:.2f}.pth'
            torch.save(model.state_dict(), best_path)
            return valid_perc

#########################################################################################################################################################

def compute_mean_std(training_data) -> List[tuple]:
    """
        Compute mean and std of training data. For Cifar10 which has 3 channels in input images.
        Cifar10 is a collection of pair of image (at 0 index) and labels (at 1 index)
    """
    imgs = [item[0] for item in training_data]
    imgs = torch.stack(imgs, dim=0).numpy()     # Stacks all images columnwise and then converts the data to numpy.

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()


    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()

    return [(mean_r,mean_g,mean_b,), (std_r,std_g,std_b,)]

#########################################################################################################################################################

def prepare_confusion_matrix(all_preds, all_targets, class_map):
    """Prepare Confusion matrix

    Args:
        all_preds (list): List of all predictions
        all_targets (list): List of all actule labels
        class_map (dict): Class names

    Returns:
        tensor: confusion matrix for size number of classes * number of classes
    """
    stacked = torch.stack((
        all_targets, all_preds
    ),
        dim=1
    ).type(torch.int64)

    no_classes = len(class_map)

    # Create temp confusion matrix
    confusion_matrix = torch.zeros(no_classes, no_classes, dtype=torch.int64)

    # Fill up confusion matrix with actual values
    for p in stacked:
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

    return confusion_matrix

#########################################################################################################################################################

def image_denormalizer(image, std_, mean_, permutation=(1,2,0)):
    """
        Function to Denormalize an image
        
        args:
        image : 
        std_ : Standard daviation of dataset used at the time of normalizing image
        mean_ : Mean of dataset used at the time of normalizing image
    """
    return image.permute(permutation) * np.array(std_) + np.array(mean_)

#########################################################################################################################################################

def get_GradCAM_heatmap(model, target_layers, use_cuda:bool, target, img:np.array, transparency, permutation=(2,0,1)):
    """
        To get heatmap of a layer's activation maps

        args:
        model: CNN Model to be used
        target_layers: Layer of which channels, gradient to be used
        use_cuda:bool
        target: target label
        img : Denormalized image, in numpy array format
    """
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(target)]  # Pass the label if incase you want to focus on a specific class. else pass None | I am passing label of image
    grayscale_cam = cam(input_tensor=img.permute(permutation).unsqueeze(dim=0).to(torch.float32), targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    return show_cam_on_image(img.to(torch.float32).numpy(), grayscale_cam, use_rgb=True, image_weight=transparency)

#########################################################################################################################################################