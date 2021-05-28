import argparse
import logging
import os
import sys
from datetime import datetime
from gln.common.cmd_args import cmd_args as gln_args
from models.gln_model.gln_trainer import GLNTrainer
try:
    from models.transformer_model.transformer_trainer import TransformerTrainer
    from onmt.bin.train import _get_parser
except Exception as e:
    print(e)
from rdkit import RDLogger


def parse_args():
    parser = argparse.ArgumentParser("train.py")
    parser.add_argument("--do_train", help="whether to do training (it's possible to only test)", action="store_true")
    parser.add_argument("--do_test", help="whether to do testing (only if implemented)", action="store_true")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--data_name", help="name of dataset, for easier reference", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--config_file", help="model config file (optional)", type=str, default="")
    parser.add_argument("--train_file", help="train SMILES file", type=str, default="")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str, default="")
    parser.add_argument("--model_path", help="model output path", type=str, default="")
    parser.add_argument("--model_seed", help="model seed", type=int, default=0)

    return parser.parse_known_args()


def train_main(args):
    if args.model_name == "gln":
        gln_args.seed = args.model_seed

        logging.info('gln args')
        logging.info(f'{gln_args}')
        trainer = GLNTrainer(
            model_name="gln",
            model_args=gln_args,
            model_config={},
            data_name=args.data_name,
            raw_data_files=[args.train_file],
            processed_data_path=args.processed_data_path,
            model_path=args.model_path
        )
    elif args.model_name == "transformer":
        # adapted from onmt.bin.train.main()
        parser = _get_parser()
        opt, _unknown = parser.parse_known_args()

        # update runtime args
        opt.config = args.config_file
        opt.log_file = args.log_file

        trainer = TransformerTrainer(
            model_name="transformer",
            model_args=opt,
            model_config={},
            data_name=args.data_name,
            raw_data_files=[],
            processed_data_path=args.processed_data_path,
            model_path=args.model_path
        )

    else:
        raise ValueError(f"Model {args.model_name} not supported!")

    logging.info("Building train model")
    trainer.build_train_model()

    if args.do_train:
        logging.info("Start training")
        trainer.train()
        logging.info("Finished training")
    if args.do_test:
        logging.info("Start testing")
        trainer.test()
        logging.info("Finished testing")

    sys.exit()

if __name__ == "__main__":
    args, unknown = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs("./logs/train", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/train/{args.log_file}.{dt}"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # train interface
    train_main(args)
