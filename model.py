
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes_ja = ["Tシャツ/トップ", "ズボン", "プルオーバー", "ドレス", "コート", "サンダル", "ワイシャツ", "スニーカー", "バッグ", "アンクルブーツ"]
classes_en = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

img_size=28

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features=64*4*4, out_features=256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = Net()

# モデルのパラメータを読み込む。deviceをcpuに選択
net.load_state_dict(
    torch.load(
        'model_state.pth', map_location=torch.device('cpu')
    )
)

def predict(img):

    # アップロードされた画像の前処理
    img = img.convert('L')
    img = img.resize((img_size, img_size))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.0), (1.0))
        ]
    )
    img = transform(img)

    # (サンプル数,チャネル数,横,縦)に変換
    x = img.reshape(1, 1, img_size, img_size)
    
    net.eval()
    y = net(x)

    y_prob = torch.nn.functional.softmax(torch.squeeze(y))
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)
    return [(classes_ja[index], classes_en[index], prob.item()) for index, prob in zip(sorted_indices, sorted_prob)]
