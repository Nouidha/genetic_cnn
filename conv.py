import torch.nn as nn



class CNN(nn.Module):
    """
    Class for Convolutional Neural Network
    """
    def __init__(self, num_classes, img_rows, img_cols, img_channels=1, num_conv_layers=2, conv_dropout=0.25, classifier_dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels

        assert num_conv_layers in [1, 2,3], "Number of convolutional layers must be equal to 1, 2 or 3"

        self.model = nn.Sequential(
        nn.Conv2d(in_channels=self.img_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(conv_dropout))

        in_channels = 64
        for _ in range(1, num_conv_layers):
            previous_in_channels = in_channels
            in_channels *= 2
            self.model = nn.Sequential(*list(self.model.children()),
                                       nn.Conv2d(in_channels=previous_in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2, 2),
                                       nn.Dropout2d(conv_dropout))

        self.model = nn.Sequential(*list(self.model.children()), nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(in_channels, self.num_classes)
        self.classifier_dropout = nn.Dropout(classifier_dropout)

    def __str__(self):
        return f"CNN with:\nConvolution:\n{self.model}\nClassifier:\n{self.classifier}"


    def forward(self, x):
        x_out = self.model(x)
        x_flat = x_out.view(x_out.size(0), -1)

        x_flat = self.classifier_dropout(x_flat)
        x_logis = self.classifier(x_flat)
        return x_logis