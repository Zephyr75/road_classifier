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


class build_cnn16(nn.Module):
    def __init__(self):
        super().__init__()
                                #channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=16, stride=16)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv1x1_1 = nn.Conv2d(16, 4, kernel_size=1, padding="same")
        self.bn1x1_1 = nn.BatchNorm2d(4)

        self.conv1x1_2 = nn.Conv2d(4, 1, kernel_size=1, padding="same")

        #following each Batch Normalization 
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.25)
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.conv1x1_1(x)
        x = self.bn1x1_1(x)
        x = self.relu(x)
        #x = self.dropout(x)

        x = self.conv1x1_2(x)

        x = expand(x, inputs.size(dim=2))
        return x

# if __name__ == "__main__":
#     x = torch.randn((2, 3, 400, 400))
#     f = build_cnn16()
#     y = f(x)
#     print(y.shape)
