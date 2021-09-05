""" main.py """
#internal
from configs.config import CFG
from models.mlp_model import MLPModel


def run():
    """Builds model, loads data, trains and evaluates"""
    print('*-----------------------------------*')
    print('Running main.py ...')
    model = MLPModel(CFG, name='tfio_tryout')
    print('* Model defined')
    model.load_data(method='tfio')
    print('* Data Loaded')
    model.build()
    model.train()
    model.evaluate()
    model.save()




if __name__ == '__main__':
    run()
