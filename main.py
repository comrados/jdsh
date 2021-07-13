from joint_controller import JointController
from utils import logger
from args import cfg


def main():
    # log
    log = logger()
    mode = 'TEST' if cfg.TEST else 'TRAIN'
    log.info('Initialization: {}, {}, {} bits, {}'.format(cfg.MODEL, cfg.DATASET, cfg.HASH_BIT, mode))

    model = JointController(log, cfg)

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
                model.training_complete()


if __name__ == '__main__':
    main()
