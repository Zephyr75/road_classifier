import torch
import torch.nn as nn

def expand(zipped_mask, width):
    # (desired) width==height in our cases
    nb_samples = zipped_mask.size(dim=0)
    expanded= torch.zeros((nb_samples, 1, width, width), device=zipped_mask.device)
    for k in range(nb_samples):
        for i in range(0, width, 16):
            for j in range(0, width, 16):
                expanded[k,0,i:i+16,j:j+16] = zipped_mask[k,0,i//16,j//16]
    return expanded


class build_cnn2(nn.Module):
    def __init__(self):
        super().__init__()
                                #channels
        self.conv1 = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv1x1_1 = nn.Conv2d(16, 4, kernel_size=1)
        self.bn1x1_1 = nn.BatchNorm2d(4)

        self.conv1x1_2 = nn.Conv2d(4, 1, kernel_size=1)

        #following each Batch Normalization 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        #self.exp = staticmethod(expand)
        
    def forward(self, inputs):
        #print(111111111111)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        #print("size at first conv: ", x.size())
        # x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv1x1_1(x)
        x = self.bn1x1_1(x)
        x = self.relu(x)
        #print("size at second conv: ", x.size())
        #x = self.dropout(x)

        x = self.conv1x1_2(x)
        #print("size at third conv: ", x.size())

                #input size instead of 400
        x = expand(x, inputs.size(dim=2))
        #print("size of expanded: ", x.size())
        return x

if __name__ == "__main__":
    x = torch.randn((2, 3, 608, 608))
    f = build_cnn2()
    y = f(x)
    print(y.shape)
