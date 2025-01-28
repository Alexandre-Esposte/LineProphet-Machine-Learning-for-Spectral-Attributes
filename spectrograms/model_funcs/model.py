
from torch import nn


class ConvolutionalNet(nn.Module):

    def __init__(self):
        
        super().__init__()
        self.convolutional = nn.Sequential(
        
                
                nn.Conv2d(1,8, kernel_size = (2,2)),
                nn.MaxPool2d((2,2)),
                nn.ReLU(),
                

                #nn.Conv2d(8,8,8),
                #nn.Conv2d(8,8,4),
                nn.Conv2d(8,8,2),
                nn.AdaptiveAvgPool2d((20,20)),
                nn.ReLU(),
                
                
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*20*20,2),
            nn.Sigmoid()
            
        )


    def forward(self,X):
        
        features = self.convolutional(X)

        y = self.dense(features)

        return y