# this is the python code to execute the entire program to train on custom object detector

from config import DEVICE, NUM_CLASSESS, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from model import create_model
from utils import Averager
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader

import torch
import matplotlib.pyplot as plt; plt.style.use("ggplot")
import time

# we are going to need two functions; training_fn and validation_fn

# TRAIN_FN
def train(train_data_loader, model):
    print("Training...")
    global train_itr
    global train_loss_list

    # init progress bar to visualize the training state
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images,targets)

        losses = sum (loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1

        # update loss value beside the progress bar for each iter
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_loss_list

# VALID_FN: almost same as train() but there is no back-prop
def validate(valid_data_loader, model):
    print("Validating...")
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_loss_list

# MAIN CODE
if __name__ == "__main__":
    # init model and select the computation hw: GPU/CPU
    model = create_model(num_classes=NUM_CLASSESS)
    model = model.to(DEVICE)

    # get model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # define optimizer
    optimizer = torch.optim.SGD(params, lr=10e-3, momentum=0.9, weight_decay=5e-4)

    # init the averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()

    # starting iter
    train_itr = 1
    val_itr = 1

    # return of train and valid list and will plot the entire training log
    train_loss_list = []
    val_loss_list = []

    MODEL_NAME = "test_1"

    # whether to show transformed imgs from data loader or not?
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_transform_image
        show_transform_image(train_loader)

    # start training
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # reset train/valid history for current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # create variables for plotting values
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        # start timer in train/valid session
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)

        # if training/validation session is up to the specified interval, will plot the data
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')
        
        if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...')
        
        if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
        
        plt.close('all')