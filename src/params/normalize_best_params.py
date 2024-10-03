from json import dumps
from pprint import pprint

import torch
from numpy import arange
from pynimbar import loading_animation

from util import NormalizeTransform, evaluate_model, get_model_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for mean in arange(0.15, 1, 0.15):
        for std in arange(0.15, 1, 0.15):
            mean = round(mean, 2)
            std = round(std, 2)

            with loading_animation(
                f"Running with mean: {mean}, std: {std}...",
                break_on_error=True,
                verbose_errors=True,
                time_it_live=True,
            ):
                (
                    normalize_train_loader,
                    normalize_test_loader,
                    normalize_validation_loader,
                ) = get_model_data([NormalizeTransform(mean, std)])

                normalize_precision = evaluate_model(
                    device,
                    normalize_train_loader,
                    normalize_test_loader,
                    normalize_validation_loader,
                    verbose=False,
                )

                result = {"mean": mean, "std": std, "precision": normalize_precision}
                results.append(result)

    results.sort(key=lambda x: x["precision"], reverse=True)
    pprint(results)

    with open("./src/params/normalize.json", "w") as f:
        f.write(dumps(results))


if __name__ == "__main__":
    main()
