transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])#tf.Resize((300,600)),tf.RandomRotation(145)])#


def infer(modelPath,imagePath):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # Check if there is GPU if not set trainning to CPU (very slow)
    Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
    Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes
    Net=Net.to(device) # Set net to GPU or CPU
    Net.load_state_dict(torch.load(modePath)) 
    Net.eval()
    Img = cv2.imread(imagePath)
    plt.imshow(Img) # Show image
    plt.show()
    Img=transformImg(Img) # Transform to pytorch
    Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
    with torch.no_grad():
       Prd=Net(Img)['out'] # Run net
       
    seg=torch.argmax(Prd[0],0).cpu().detach().numpy() # Get classes
    plt.imshow(seg) # display image
    plt.show()

modelPath="16000.torch" # Path to trained model
imagePath="test.jpg" # Test image
infer(modelPath,imagePath)
