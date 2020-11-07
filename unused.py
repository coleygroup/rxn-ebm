def resume(args):
    """resume training from saved checkpoint. Highly similar to trainEBM() except in
    loading saved model, optimizer & stats (& inferring existing model_args from saved stats file)
    """
    expt_name = "rdm_0_cos_0_bit_5_3"  # USER INPUT
    precomp_file_prefix = (
        "50k_" + expt_name
    )  # USER INPUT, expt.py will add f'_{phase}.npz'
    augmentations = {  # USER INPUT
        "rdm": {"num_neg": 0},
        "cos": {"num_neg": 0},
        "bit": {"num_neg": 5, "num_bits": 3},
    }

    #######################################################
    ##################### PRECOMPUTE ######################
    #######################################################
    lookup_dict_filename = "50k_mol_smi_to_sparse_fp_idx.pickle"
    mol_fps_filename = "50k_count_mol_fps.npz"
    search_index_filename = "50k_cosine_count.bin"
    mut_smis_filename = "50k_neg150_rad2_maxsize3_mutprodsmis.pickle"
    augmented_data = dataset.AugmentedDataFingerprints(
        augmentations,
        lookup_dict_filename,
        mol_fps_filename,
        search_index_filename,
        mut_smis_filename,
        seed=random_seed,
    )

    rxn_smis_file_prefix = "50k_clean_rxnsmi_noreagent"
    for phase in ["train", "valid", "test"]:
        augmented_data.precompute(
            output_filename=precomp_file_prefix + f"_{phase}.npz",
            rxn_smis=rxn_smis_file_prefix + f"_{phase}.pickle",
            distributed=False,
            parallel=False,
        )

    #######################################################
    ################ LOAD SAVED FILES #####################
    #######################################################
    optimizer_name = "Adam"  # USER INPUT, TODO: infer this from saved_stats
    model_name = "FeedforwardEBM"  # USER INPUT
    date_trained = "01_10_2020"  # USER INPUT
    old_expt_name = "" # USER INPUT
    saved_stats_filename = f'{model_name}_{old_expt_name}_stats.pkl'
    checkpoint_folder = expt_utils.setup_paths(
        "LOCAL", load_trained=True, date_trained=date_trained
    )
    saved_model, saved_optimizer, saved_stats = expt_utils.load_model_opt_and_stats(
        saved_stats_filename, checkpoint_folder, model_name, optimizer_name
    )

    #######################################################
    ##################### TRAINING ########################
    #######################################################
    # if using all same stats as before, just use: saved_stats['model_args'], ['train_args'], ['fp_args']
    # as parameters into Experiment (with **dictionary unpacking), otherwise, define again below:
    train_args = {
        "batch_size": 4096,
        "learning_rate": 5e-3,
        "optimizer": None, # load saved_optimizer
        "epochs": 5,
        "early_stop": True,
        "min_delta": 1e-4,
        "patience": 1,
        "num_workers": 0,
        "checkpoint": True,
        "random_seed": 0,
        "precomp_file_prefix": precomp_file_prefix,
        "checkpoint_folder": checkpoint_folder,
        "expt_name": expt_name,
    }

    experiment = expt.Experiment(
        saved_model,
        saved_stats["model_args"],
        augmentations=augmentations,
        **train_args,
        **saved_stats["fp_args"],
        load_checkpoint=load_trained,
        saved_optimizer=saved_optimizer,
        saved_stats=saved_stats,
        saved_stats_filename=saved_stats_filename,
        begin_epoch=saved_stats["best_epoch"] + 1,
    )

    experiment.train()
    experiment.test()
    scores_test = experiment.get_topk_acc(phase="test", k=1)
    scores_train = experiment.get_topk_acc(phase="train", k=1)