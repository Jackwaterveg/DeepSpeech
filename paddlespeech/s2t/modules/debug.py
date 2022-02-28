from paddle import nn


class PassLayer(nn.Layer):
    def __init__(self):
        super(PassLayer, self).__init__()
        return

    def forward(self, x):
        return x
