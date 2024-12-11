import torch

class CNNModel(torch.nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()

        # Second convolution layer with 16 out channels, 1 padding
        self.cnn1 = torch.nn.Conv2d(in_channels=1, out_channels=16,
                              kernel_size=3, stride=1, padding=1)

        # normalization for statbility
        self.batchnorm1 = torch.nn.BatchNorm2d(16)

        # size 2 kernel for maxpool
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolution layer with 32 out channels, 1 padding
        self.cnn2 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=3, stride=1, padding=1)

        # normalization for statbility
        self.batchnorm2 = torch.nn.BatchNorm2d(32)

        # size 2 kernel for maxpool
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        # Third convolution layer with 64 out channels, 1 padding
        self.cnn3 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                              kernel_size=3, stride=1, padding=1)

        # normalization for statbility
        self.batchnorm3 = torch.nn.BatchNorm2d(64)

        # size 2 kernel for maxpool
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        # Fully connected layer that takes the flattened output
        self.fc2 = torch.nn.Linear(64 * 16 * 16, 4)

    def forward(self, x):

        # YOUR CODE HERE

        # input image -> conv1 -> relu -> batchnorm -> maxpool1
        conv1_out = torch.nn.functional.relu(self.cnn1(x))
        pool1_out = self.maxpool1(self.batchnorm1(conv1_out))

        # maxpool1 output -> conv2 -> relu -> batchnorm -> maxpool2
        conv2_out = torch.nn.functional.relu(self.cnn2(pool1_out))
        pool2_out = self.maxpool2(self.batchnorm2(conv2_out))

        # maxpool2 output -> conv3 -> relu -> batchnorm -> maxpool3
        conv3_out = torch.nn.functional.relu(self.cnn3(pool2_out))
        pool3_out = self.maxpool3(self.batchnorm3(conv3_out))

        # flatten the maxpool3 output to be used as input into FCN layer
        fcn_input = pool3_out.view(pool3_out.size(0), -1)

        # Use the raw output of the fully connected layer as the final output
        out = self.fc2(fcn_input)

        return out