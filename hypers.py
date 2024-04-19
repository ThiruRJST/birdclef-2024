from dataclasses import dataclass

@dataclass
class HyperParams():
    Arch: str = "Resnet"
    lr: float = 0.01
    optimizer:str = "Adam"
    test_size:float = 0.2