from JDSH import JDSH
from DJSRH import DJSRH
from utils import logger
from args import config


def main():
    # log
    log = logger()

    if config.MODEL == "DJSRH":
        Model = DJSRH(log, config)
    else:
        Model = JDSH(log, config)

    if not config.TRAIN:
        Model.load_checkpoints(config.CHECKPOINT)
        Model.eval()

    else:
        for epoch in range(config.NUM_EPOCH):
            Model.train(epoch)
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                Model.eval()
            # save the model
            if epoch + 1 == config.NUM_EPOCH:
                Model.save_checkpoints()
                Model.training_coplete()


if __name__ == '__main__':
    main()
