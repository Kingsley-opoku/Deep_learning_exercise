from model import CovNetMNIST
from handwritten_pred import NumberPredictor
import torch
import click





@click.command()
@click.argument('path')
def cli(path):
    model=CovNetMNIST()
    model = torch.load('mymodel1.pth')
    numbers=NumberPredictor(model, img_path=path)
    numbers.show_digits()
    


if __name__=='__main__':
    cli()