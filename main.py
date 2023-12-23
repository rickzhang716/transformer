from training.basic_example.train import example_simple_model
from training.train_model import train_single_model
from training.config import config
from training.check_output import run_example
import argparse





parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("model_name", default="________")
args = parser.parse_args()
print(args.command)

config["file_name"] += args.model_name + "_"
match args.command:
    case "train":
        train_single_model("cpu", config)
    case "example_model":
        example_simple_model()
    case "test":
        run_example(1000)




