import itertools
from json import dumps
from pprint import pprint

import torch
from numpy import arange
from pynimbar import loading_animation

from util import DenoiseTransform, evaluate_model, get_model_data


def process_combination(mean, std, device):
	(
		denoise_train_loader,
		denoise_test_loader,
		denoise_validation_loader,
	) = get_model_data([DenoiseTransform(mean, std)])

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
		"mean": mean,
		"std": std,
		"precision": denoise_precision,
	}


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	results = []

	combinations = list(itertools.product(arange(0.15, 1, 0.15), arange(0.15, 1, 0.15)))

	for mean, std in combinations:
		mean = round(mean, 2)
		std = round(std, 2)

		with loading_animation(
			f"Running with mean: {mean}, std: {std}...",
			break_on_error=True,
			verbose_errors=True,
			time_it_live=True,
		):
			result = process_combination(t, s, device)
			results.append(result)

	results.sort(key=lambda x: x["precision"], reverse=True)
	pprint(results)

	with open("./src/params/denoise_high_con.json", "w+") as f:
		f.write(dumps(results))


if __name__ == "__main__":
	main()
