#==============================================
# Date: 20.09.2019
# Description: model for data which is like grayscale matrix (1X90X64x64)
#==============================================



import argparse
from datetime import datetime
import os
from tqdm import tqdm

#Pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import models, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary

# My own imports
from Dataloaders.spermFeatureDataLoaderNormalized_unsqueezed_stackedFrames import SpermFeatureDatasetNormalizedUnsqueezed as spvdn

#======================================
# Get and set all input parameters
#======================================

parser = argparse.ArgumentParser()

# Hardware
parser.add_argument("--device", default="gpu", help="Device to run the code")
parser.add_argument("--device_id", type=int, default=0, help="")

# Optional parameters to identify the experiments
parser.add_argument("--name", default="", type=str, help="A name to identify this test later")
parser.add_argument("--id", default=datetime.timestamp(datetime.now()), help="Generate ID from the timestamp")
parser.add_argument("--py_file",default=os.path.abspath(__file__)) # store current python file

# Directory and file handling
parser.add_argument("--gt_csv_file", 
                    default="/home/vajira/DL/Medicotask_2019/csv_files/semen_analysis_data.csv",
                    help="Semen analysis data (ground truth)")
parser.add_argument("--id_csv_file", 
                    default="/home/vajira/DL/Medicotask_2019/csv_files/videos_id.csv",
                    help="Video IDs")
parser.add_argument("--data_root", 
                    default="/work/vajira/data/stacked_original_frames_9x256x256",
                    help="Video data root with three subfolders (fold 1,2 and 3)")

parser.add_argument("--out_dir", 
                    default="/work/vajira/mediaeval_2019_output",
                    help="Main output dierectory")

parser.add_argument("--tensorboard_dir", 
                    default="/work/vajira/mediaeval_2019_output/tensorboard_out",
                    help="Folder to save output of tensorboard")

# columns to retrun from dataloader
parser.add_argument("--cols", default=["Progressive motility (%)", "Non progressive sperm motility (%)", "Immotile sperm (%)"])

# Hyper parameters

parser.add_argument("--bs", type=int, default=32, help="Mini batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
parser.add_argument("--num_workers", type=int, default=32, help="Number of workers in dataloader")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay of the optimizer")
parser.add_argument("--lr_sch_factor", type=float, default=0.1, help="Factor to reduce lr in the scheduler")
parser.add_argument("--lr_sch_patience", type=int, default=25, help="Num of epochs to be patience for updating lr")
parser.add_argument("--data_reset", type=int, default=10000, help="number of epochs to reset dataloader")

#AE
#parser.add_argument("--hidden_size", type=int, default=128, help="Number of hidden layers in LSTM")
#parser.add_argument("--num_layers_lstm", type=int, default=2, help="Number of layers in the LSTM")
parser.add_argument("--ae_checkpoint", default="/work/vajira/mediaeval_2019_output/4008_0_mt_fold_1_stackedImages_9x224x224_resnet34LSTM_nonNormalized_basecase_video_Adam.py/checkpoints/4008_0_mt_fold_1_stackedImages_9x224x224_resnet34LSTM_nonNormalized_basecase_video_Adam.py_epoch:2050.pt", help="Pre-trained AE path")

# Action handling 
parser.add_argument("--num_epochs", type=int, default=0, help="Numbe of epochs to train")
parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch in retraining")
parser.add_argument("action", type=str, help="Select an action to run", choices=["train", "retrain", "inference", "check"])
parser.add_argument("--checkpoint_interval", type=int, default=25, help="Interval to save checkpoint models")
parser.add_argument("--fold", type=str, default="fold_2", help="Select the validation fold", choices=["fold_1", "fold_2", "fold_3"])



opt = parser.parse_args()


    
#==========================================
# Device handling
#==========================================
torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#===========================================
# Folder handling
#===========================================

#make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)


# make subfolder in the output folder 
py_file_name = opt.py_file.split("/")[-1] # Get python file name (soruce code name)
checkpoint_dir = os.path.join(opt.out_dir, py_file_name + "/checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# make tensorboard subdirectory for the experiment
tensorboard_exp_dir = os.path.join(opt.tensorboard_dir, py_file_name)
os.makedirs( tensorboard_exp_dir, exist_ok=True)



#==========================================
# Tensorboard
#==========================================
# Initialize summary writer
writer = SummaryWriter(tensorboard_exp_dir)


#==========================================
# Prepare Data
#==========================================
def prepare_data():

    # Whole dataset
    dataset_all = {x: spvdn(opt.gt_csv_file, opt.id_csv_file,
                                                     os.path.join(opt.data_root, x),
                                                     opt.cols,
                                                     ) for x in ['fold_1', 'fold_2', 'fold_3']}

    # Use selected fold for validation
    train_folds = list(set(["fold_1", "fold_2", "fold_3"]) - set([opt.fold]))
    validation_fold = opt.fold

    # Train sub dataset from the whole dataset 
    # 2 folds to train 
    dataset_train = torch.utils.data.ConcatDataset([dataset_all["fold_2"], dataset_all["fold_3"]])
    
    # 1 fold to validation
    dataset_val = dataset_all["fold_1"]

    train_size = len(dataset_train)
    val_size = len(dataset_val)

    print("train dataset size =", train_size)
    print("validation dataset size=", val_size)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.bs,
                                                  shuffle=True, num_workers= opt.num_workers)
    
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.bs,
                                                  shuffle=True, num_workers= opt.num_workers)

    std = dataset_all['fold_1'].std[opt.cols].values.tolist()
    mean = dataset_all['fold_1'].mean[opt.cols].values.tolist()

    return {"train":dataloader_train, "val":dataloader_val, "dataset_size":{"train": train_size, "val":val_size}, "std": std, "mean": mean}

#================================================
# Train the model
#================================================

def train_model(model, model_ae,optimizer, criterion, criterion_validation, dataloaders: dict, scheduler):

    # std and mean, converted into tensors and transfered into device
    std = torch.FloatTensor(dataloaders["std"]).to(device, torch.float)
    mean = torch.FloatTensor(dataloaders["mean"]).to(device, torch.float)

    for epoch in tqdm(range(opt.start_epoch + 1, opt.start_epoch + opt.num_epochs + 1)):

        # reset dataloader after some epochs
        if epoch % opt.data_reset == 0:
            dataloaders = prepare_data()
            print("Dataloader reset...!!!")

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
                dataloader = dataloaders["train"]
            else:
                model.eval()
                dataloader = dataloaders["val"]

            running_loss = 0.0
            running_loss_real = 0.0
            
            for i, sample in tqdm(enumerate(dataloader, 0)):

                # handle input data
                input_img = sample["features"]
                input_img = input_img.to(device, torch.float)

                # Ground truth data
                # gt_normalized = sample["data_normalized"]
                gt_real = sample["data_non_normalized"]
                #gt_normalized = gt_normalized.to(device, torch.float)
                gt_real = gt_real.to(device, torch.float)

                #gt_normalized = sample["gt_normalized"]
                #gt_normalized = gt_normalized.to(device, torch.float)
                #gt_normalized = gt_normalized.to(device, torch.float)

                # get feature image
                feature_img, output_img= model_ae(input_img)
                

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(feature_img)
                    #outputs_real = outputs  # * std + mean

                    # Loss
                    loss = criterion(outputs, gt_real)
                    loss_real = criterion_validation(outputs , gt_real)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                
                # calculate running loss
                running_loss += loss.detach().item() * input_img.size(0)
                running_loss_real+= loss_real.detach().item() * input_img.size(0)

            epoch_loss = running_loss / dataloaders["dataset_size"][phase]
            epoch_loss_real  = running_loss_real / dataloaders["dataset_size"][phase]

            # update tensorboard writer
            writer.add_scalars("Loss", {phase:epoch_loss}, epoch)
            writer.add_scalars("Loss_real" , {phase:epoch_loss_real}, epoch)
            
            # update the lr based on the epoch loss
            if phase == "val": 
                # Get current lr
                lr = optimizer.param_groups[0]['lr']
                print("lr=", lr)
                writer.add_scalar("LR", lr, epoch)
                # scheduler.step(epoch_loss) 

                # save sample feature grid and image grid
                #save_image(feature_img, str(epoch) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                writer.add_images("input_one_channel", input_img[:, 0:1, :, :], epoch)

                writer.add_images("feature_img",feature_img, epoch)

                writer.add_images("output_one_channel", output_img[:, 0:1, :, :], epoch)
                
            # Print output
            print('Epoch:\t  %d |Phase: \t %s | Loss:\t\t %.4f | Loss-Real:\t %.4f '
                      % (epoch, phase, epoch_loss, epoch_loss_real))
        
        # Save model
        if epoch % opt.checkpoint_interval == 0:
            save_model(model, optimizer, epoch, loss) # loss = validation loss (because of phase=val at last)



#===============================================
# Prepare models
#===============================================

def prepare_ae():
    checkpoint_path = opt.ae_checkpoint
    model_ae = TubeEncoderDecoder()#CNNLSTM()
    checkpoint = torch.load(checkpoint_path)

    model_ae.load_state_dict(checkpoint["model_state_dict"])
    model_ae.eval()

    model_ae = model_ae.to(device)
    print("Pretrained AE successfully loaded")
    return model_ae


def prepare_model():
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(device)
    
    return model

class TubeEncoderDecoder(nn.Module):
    def __init__(self):
        super(TubeEncoderDecoder, self).__init__()
     

        self.encoder = nn.Sequential(
            nn.Conv2d(9, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2 ),
            nn.ReLU())
        
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 9, 2, stride=2 ),
            nn.ReLU())


    def forward(self, x):
        
        feature_img = self.encoder(x)
        output = self.decoder(feature_img)
                
        return feature_img, output






#====================================
# Run training process
#====================================
def run_train():
    model = prepare_model()
    model_ae = prepare_ae()
    dataloaders = prepare_data()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr , weight_decay=opt.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr )

    criterion =  nn.MSELoss() # backprop loss calculation
    criterion_validation = nn.L1Loss() # Absolute error for real loss calculations

    # LR shceduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=opt.lr_sch_factor, patience=opt.lr_sch_patience, verbose=True)

    # call main train loop
    train_model(model,model_ae, optimizer,criterion, criterion_validation, dataloaders, scheduler)

#====================================
# Re-train process
#====================================
def run_retrain():
    model = prepare_model()
    dataloaders = prepare_data()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr , weight_decay=opt.weight_decay)

    criterion =  nn.MSELoss() # backprop loss calculation
    criterion_validation = nn.L1Loss() # Absolute error for real loss calculations

    #Loading data from start epoch number 
    check_point_name = py_file_name + "_epoch:{}.pt".format(opt.start_epoch) # get code file name and make a name
    check_point_path = os.path.join(checkpoint_dir, check_point_name)

    checkpoint = torch.load(check_point_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # LR shceduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=opt.lr_sch_factor, patience=opt.lr_sch_patience, verbose=True)

    print("Models loaded successfully from checkpoint:\t {}".format(check_point_name))

    # call main train loop
    train_model(model,optimizer,criterion, criterion_validation, dataloaders, scheduler)


#=====================================
# Save models
#=====================================
def save_model(model, optimizer,  epoch,  validation_loss):
   
    check_point_name = py_file_name + "_epoch:{}.pt".format(epoch) # get code file name and make a name
    check_point_path = os.path.join(checkpoint_dir, check_point_name)
    # save torch model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "train_loss": train_loss,
        "val_loss": validation_loss
    }, check_point_path)

#=====================================
# Check model
#====================================
def check_model_graph():
    model = prepare_model()

    summary(model, (1, 256, 256)) # this run on GPU
    model = model.to('cpu')
    #dataloaders = prepare_data()
    #sample = next(iter(dataloaders["train"]))

    #inputs = sample["features"]
   # inputs = inputs.to(device, torch.float)
    #print(inputs.shape)
    print(model)
    dummy_input = Variable(torch.rand(13, 1, 256, 256))
    
    writer.add_graph(model, dummy_input) # this need the model on CPU


if __name__ == "__main__":

    data_loaders = prepare_data()
    print(vars(opt))
    print("Test OK")

    # Train or retrain or inference
    if opt.action == "train":
        print("Training process is strted..!")
        run_train()
        pass
    elif opt.action == "retrain":
        print("Retrainning process is strted..!")
        run_retrain()
        pass
    elif opt.action == "inference":
        print("Inference process is strted..!")
        pass
    elif opt.action == "check":
        check_model_graph()
        print("Check pass")

    # Finish tensorboard writer
    writer.close()
    