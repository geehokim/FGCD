from torch import nn


class linear_model(nn.Module):
    def __init__(self, num_classes):
        super(linear_model, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(60, num_classes),
            #nn.GroupNorm(2, mlp_hidden_size),
            #nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)



def leaf_synthetic(num_classes=10, l2_norm=False, use_pretrained = False, transfer_learning = True, use_bn = False, use_pre_fc = False, use_bn_layer = False):
    return linear_model(num_classes)
