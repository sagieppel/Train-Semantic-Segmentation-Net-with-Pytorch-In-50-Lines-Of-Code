import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf

TrainImagesFolder=r"Trans10k/train/images//"
TrainAnnFolder="Trans10k/train/masks//"

Learning_Rate=1e-5
Weight_Decay = 4e-5

width=height=900 # image width and height
batchSize=3

ListImages=[]
ListImages=os.listdir(TrainImagesFolder) # Create list of images
#----------------------------------------------Transform image----------------------------------------------------------------------------------------------------------------------------
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])#tf.Resize((300,600)),tf.RandomRotation(145)])#
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor()])
#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(Hb=0,Wb=0): # First lets load random image and  the corresponding annotation
    idx=np.random.randint(0,len(ListImages)) # Select random image
    Img= cv2.imread(os.path.join(TrainImagesFolder, ListImages[idx]))
    AnnMap=cv2.imread(os.path.join(TrainAnnFolder, ListImages[idx].replace(".jpg", "_mask.png")))
    AnnMap = (AnnMap[:, :,2] > 0).astype(np.float32) + (AnnMap[ :, :,1] > 0).astype(np.float32) # Convert from RGB to one channel format
    Img=transformImg(Img)
    AnnMap=transformAnn(AnnMap)
    return Img,AnnMap
#--------------Load batch of images-----------------------------------------------------
def LoadBatch(): # Load batch of images
    images = torch.zeros([batchSize,3,height,width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i],ann[i]=ReadRandomImage()
    return images, ann

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes
Net=Net.to(device)
criterion = torch.nn.CrossEntropyLoss() # Set loss function
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer

for itr in range(20000): # Training loop
   images,ann=LoadBatch() # Load taining batch
   images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
   ann = torch.autograd.Variable(ann, requires_grad=False).to(device) # Load annotation
   Pred=Net(images)['out'] # make prediction
   Loss=criterion(Pred,ann.long()) # Calculate cross entropy loss
   Loss.backward() # Backpropogate loss
   optimizer.step() # Apply gradient descent change to weight
   print(itr,") Loss=",Loss.data.cpu().numpy())
   if itr % 1000 == 0: #Save model weight once every 60k steps permenant file
        print("Saving Model" +str(itr) + ".torch")
        torch.save(Net.state_dict(),   str(itr) + ".torch")
