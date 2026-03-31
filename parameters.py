import argparse


def get_params():
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10 / Names")

    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset",   choices=["mnist", "cifar10", "names"], default="mnist")
    parser.add_argument("--model",     choices=["mlp", "cnn", "vgg", "resnet", "mobilenet", "rnn", "lstm"], default="mlp")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--batch_size",type=int,   default=64)
    # VGG-specific
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    # ResNet-specific
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)")
    # RNN/LSTM-specific
    parser.add_argument("--data_dir",  type=str,   default="data/names",
                        help="Directory with *.txt language files (names dataset)")
    parser.add_argument("--hidden",    type=int,   default=128,
                        help="Hidden size for RNN/LSTM (default: 128)")

    args = parser.parse_args()

    # Dataset-dependent settings
    if args.dataset == "mnist":
        input_size = 784
        mean, std  = (0.1307,), (0.3081,)
        num_classes = 10
    elif args.dataset == "cifar10":
        input_size = 3072
        mean       = (0.4914, 0.4822, 0.4465)
        std        = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    else:  # names
        input_size  = None   # determined at runtime from N_LETTERS
        mean, std   = None, None
        num_classes = None   # determined at runtime from dataset

    return {
        # Data
        "dataset":      args.dataset,
        "data_dir":     args.data_dir,
        "num_workers":  2,
        "mean":         mean,
        "std":          std,

        # Model
        "model":        args.model,
        "input_size":   input_size,
        "hidden_sizes": [512, 256, 128],
        "num_classes":  num_classes,
        "dropout":      0.3,
        "vgg_depth":    args.vgg_depth,
        "resnet_layers": args.resnet_layers,
        "hidden":       args.hidden,

        # Training
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  1e-4,

        # Misc
        "seed":         42,
        "device":       args.device,
        "save_path":    "best_model.pth",
        "log_interval": 100,

        # CLI
        "mode":         args.mode,
    }