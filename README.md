# Image-Denoising-AutoEncoder-Using-CNN
# Data Set:
We are going to work on MNIST DATASET it consists of handwritten digits(0-9) 60,000 training samples and 10,000 training samples, shape of each image is (1,28,28).
![](https://i.imgur.com/UvjsypP.png)
# Architecture:
  **Encoder**:
  
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)) 
  
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) 
  
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  
  (l1): Linear(in_features=1024, out_features=256, bias=True)
  
  **Decoder**:
  
  (l2): Linear(in_features=256, out_features=1024, bias=True)
  
  (conv4): ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(3, 3), padding=(2, 2))
  
  (conv5): ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(3, 3), padding=(4, 4))
  
  (conv6): ConvTranspose2d(16, 1, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))
| Layer  | Input Dimensions | OutPut dimensions |
| -----  | ---------------- | ----------------- |
| conv1  |  (m,1,28,28)     |   (m,16,16,16)    |
| conv2  |  (m,16,16,16)    |   (m,32,8,8)      |
| conv3  |  (m,32,8,8)      |   (m,64,4,4)      |
| L1     |  (m,1024)        |   (m,256)         |
| L2     |  (m,256)         |   (m,1024)        |
| conv4  |  (m,64,4,4)      |   (m,32,8,8)      |
| conv5  |  (m,32,8,8)      |   (m,16,16,16)    |
| conv6  |  (m,16,16,16)    |   (m,1,28,28)     |


# Activation Functions:
ReLU is used in conv layers to learn non linearities in data and last layer is activated with sigmoid t scale the values to lie between (0,1)
# Hyper Parameters 

| Hyper Perameter | Values |
| --------------- | ------ |
| Learning Rate   | 0.001  |
| Batch Size      | 64     |
| Epochs          | 10     |
|Loss Function    |MSE Loss|
|Optimiser        |Adam    |
| Beta1           | 0.9    |
| Beta2           | 0.999  |
| Weight Deacy    | 1e-5   |

# Noises Added:
In this we have added gausian noises using torch.randn().We can mask some values of inputs randomly using mask function in code and we can add some other noises.
# Training
Noisy images are feeded into the nueral network these are compresed to a latent space of lesser dimensions than input using encoder(conv2d layers) and then decoder(convTranspose2d layers) takes the latent features as input and reconstructs image.Now loss(reconstruction error) between the reconstructed image and the original image is computed using MSE loss function and this losses are backpropagated.Using ADAM algorithm model updates the value of parameters in order to reduce the reconstruction error. In this way Denoising AE learns to remove noise.

![](https://i.imgur.com/2ZyUplQ.jpg)


# Testing 
After the whole training process is completed we now test our model on unseen data.We will feed a noisy input image and by seeing the reconstruction error we can decide how well our model is performing.
# Results
**Plot Loss vs Iters:**

![](https://i.imgur.com/1JIUoti.png)


**OutPut:**

When Gausian Noise is added

![](https://i.imgur.com/xvrH9q5.png)








