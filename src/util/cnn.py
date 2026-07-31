import torch
from torch import Tensor, nn


def configure_cuda() -> None:
	"""Enable CUDA performance optimizations."""
	torch.backends.cudnn.benchmark = True
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True


class BinaryCNN(nn.Module):
	"""Four-block convolutional network for binary classification."""

	def __init__(self, device: torch.device | None = None) -> None:
		"""Initialize the convolutional and fully connected blocks."""
		super().__init__()
		self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.fc_layers = nn.Sequential(
			nn.Linear(256 * 8 * 8, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Return one raw binary-classification logit per image."""
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = x.contiguous().view(-1, 256 * 8 * 8)
		return self.fc_layers(x)
