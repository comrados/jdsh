from JDSH import JDSH
from DJSRH import DJSRH
from utils import logger
from args import cfg


def main():
    # log
    log = logger()

    if cfg.MODEL == "DJSRH":
        model = DJSRH(log, cfg)
    else:
        model = JDSH(log, cfg)

    if cfg.TEST:
        model.load_checkpoints('best.pth')
        model.test()
    else:
        for epoch in range(cfg.NUM_EPOCH):
            model.train(epoch)
            if (epoch + 1) % cfg.EVAL_INTERVAL == 0:
                model.eval()
            # save the model
            if epoch + 1 == cfg.NUM_EPOCH:
                model.save_checkpoints()
                model.training_coplete()


if __name__ == '__main__':
    main()
