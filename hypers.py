from dataclasses import dataclass

@dataclass
class HyperParams():
    Arch: str = "resnet34"
    root_path:str = "/home/tensorthiru/CLEF/birdclef-2024/melspecs"
    lr: float = 0.01
    optimizer:str = "Adam"
    test_size:float = 0.2
    num_classes:int = 182