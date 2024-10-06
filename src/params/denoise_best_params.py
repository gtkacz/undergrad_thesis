import itertools
from json import dumps
from pprint import pprint

import torch
from pynimbar import loading_animation

from util import DenoiseTransform, evaluate_model, get_model_data


def process_combination(template_window_size, search_window_size, device):
    (
        denoise_train_loader,
        denoise_test_loader,
        denoise_validation_loader,
    ) = get_model_data([DenoiseTransform(template_window_size, search_window_size)])

    denoise_precision = evaluate_model(
        device,
        denoise_train_loader,
        denoise_test_loader,
        denoise_validation_loader,
        verbose=False,
    )

    # Free up memory
    del denoise_train_loader
    del denoise_test_loader
    del denoise_validation_loader
    torch.cuda.empty_cache()

    return {
        "template_window_size": template_window_size,
        "search_window_size": search_window_size,
        "precision": denoise_precision,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    combinations = list(itertools.product(range(9, 10), range(20, 24)))

    for t, s in combinations:
        with loading_animation(
            f"Running with template_window_size: {t}, search_window_size: {s}...",
            break_on_error=True,
            verbose_errors=True,
            time_it_live=True,
        ):
            result = process_combination(t, s, device)
            results.append(result)

    results.sort(key=lambda x: x["precision"], reverse=True)
    pprint(results)

    with open("./src/params/denoise.json", "w+") as f:
        f.write(dumps(results))


if __name__ == "__main__":
    main()
