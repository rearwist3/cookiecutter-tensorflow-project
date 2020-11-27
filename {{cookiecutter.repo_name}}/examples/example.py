import tensorflow as tf

from data_loader.data_loader import Dataset
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

@hydra.main(config_path='config.yaml')
def main(cfg):
    print(cfg)
    # capture the config path from the run arguments
    # then process the json configuration file

    # try:
    #     args = get_args()
    #     config = process_config(args.config)

    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    # create the experiments dirs
    create_dirs([cfg.summary_dir, cfg.checkpoint_dir])
    # create your data generator
    dataset = Dataset(cfg)
    
    # create an instance of the model you want
    model = ExampleModel(cfg)
    # create tensorboard logger
    logger = Logger(cfg)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(model, dataset, cfg, logger)
    #load model if exists
    model.load()
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
