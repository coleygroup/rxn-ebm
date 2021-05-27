import logging
import os
import random
import traceback
from datetime import date
from pathlib import Path
from typing import Optional, Union

import numpy as np
import requests
import torch
import torch.nn as nn
from rxnebm.model import FF, G2E, S2E, model_utils

def setup_paths(
    load_trained: Optional[bool] = False,
    date_trained: Optional[str] = None,
    ckpt_root: Optional[Union[str, bytes, os.PathLike]] = None,
) -> Union[str, bytes, os.PathLike]:
    """
    Parameters
    ----------
    root : Union[str, bytes, os.PathLike] (Default = None)
        path to the root folder where checkpoints will be stored
        If None, this is set to full/path/to/rxnebm/checkpoints/
    """
    if load_trained:
        if date_trained is None:
            raise ValueError("Please provide date_trained as DD_MM_YYYY")
    else:
        date_trained = date.today().strftime("%d_%m_%Y")

    if ckpt_root is None:
        ckpt_root = Path(__file__).resolve().parents[1] / "checkpoints"
    else:
        ckpt_root = Path(ckpt_root)
    checkpoint_folder = ckpt_root / date_trained
    os.makedirs(checkpoint_folder, exist_ok=True)
    print(f"created checkpoint_folder: {checkpoint_folder}")
    return checkpoint_folder

def load_or_create_vocab(args):
    """Currently only supports loading. The vocab is small enough that a single universal vocab suffices"""
    root = Path(__file__).resolve().parents[1] / "data" / "cleaned_data"

    vocab = {}
    with open(root / args.vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip()
            vocab[token] = i

    return vocab

def load_model_and_opt(
    args,
    checkpoint_folder: Union[str, bytes, os.PathLike],
    optimizer_name: str = "Adam",
):
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_filename = f'{args.model_name}_{args.old_expt_name}_checkpoint.pth.tar'

    checkpoint = torch.load(
        Path(checkpoint_folder) / checkpoint_filename,
        map_location=torch.device(curr_device),
    )
    print(f"loaded checkpoint from {Path(checkpoint_folder) / checkpoint_filename}")
    
    begin_epoch = checkpoint["epoch"] + 1

    if args.model_name == "FeedforwardEBM":
        saved_model = FF.FeedforwardEBM(args)
    elif args.model_name == "GraphEBM_1MPN":        # Graph to energy, project both reactants & products w/ dot product output
        saved_model = G2E.GraphEBM_1MPN(args)
    elif args.model_name == "GraphEBM_2MPN":        # Graph to energy, separate encoders + projections, feedforward output
        saved_model = G2E.GraphEBM_2MPN(args)         
    elif args.model_name == "TransformerEBM":
        assert args.vocab_file is not None, "Please provide precomputed --vocab_file!"
        vocab = load_or_create_vocab(args)
        saved_model = S2E.TransformerEBM(args, vocab)
    else:
        raise ValueError("Unrecognized model name")

    saved_optimizer = model_utils.get_optimizer(optimizer_name)(
        saved_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    for key in list(checkpoint["state_dict"].keys()):
        if 'module.' in key:
            checkpoint["state_dict"][key.replace('module.', '')] = checkpoint["state_dict"][key]
            del checkpoint["state_dict"][key]
    saved_model.load_state_dict(checkpoint["state_dict"])
    saved_optimizer.load_state_dict(checkpoint["optimizer"])
    print('Loaded model and optimizer state dicts')

    if torch.cuda.is_available() and not args.ddp: # if ddp, need to move within each process  
        # move optimizer tensors to gpu  https://github.com/pytorch/pytorch/issues/2830
        for state in saved_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    return saved_model, saved_optimizer, begin_epoch

def send_message(msg, chat_id, bot_token):
    """
    params:
    -------
    msg: message you want to receive
    chat_id: CHAT_ID
    bot_token: API_KEY of your bot
    """

    url  = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    data = {'chat_id': str(chat_id), 'text': f'{msg}'}
    requests.post(url, data)
