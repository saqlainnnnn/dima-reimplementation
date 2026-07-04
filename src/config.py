from dataclasses import dataclass

@dataclass
class Config:
    image_size = 512
    batch_size = 8
    epochs = 20
    lr = 1e-4

    backbone = "resnet34"

cfg = Config()