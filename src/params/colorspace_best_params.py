from json import dumps
from pprint import pprint

import torch
from pynimbar import loading_animation

from util import ColorSpaceTransform, evaluate_model, get_model_data


def process_combination(target_space, device):
	(
		colorspace_train_loader,
		colorspace_test_loader,
		colorspace_validation_loader,
	) = get_model_data([ColorSpaceTransform(source_space="RGB", target_space=target_space)])

	colorspace_precision = evaluate_model(
		device,
		colorspace_train_loader,
		colorspace_test_loader,
		colorspace_validation_loader,
		verbose=False,
	)

	# Free up memory
	del colorspace_train_loader
	del colorspace_test_loader
	del colorspace_validation_loader
	torch.cuda.empty_cache()

	return {
		"target_space": target_space,
		"precision": colorspace_precision,
	}


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	results = []

	target_spaces = ["BGR", "HSV", "LAB", "YUV"]

	for target_space in target_spaces:
		with loading_animation(
			f"Running with target_space: {target_space}...",
			break_on_error=True,
			verbose_errors=True,
			time_it_live=True,
		):
			result = process_combination(target_space, device)
			results.append(result)

	results.sort(key=lambda x: x["precision"], reverse=True)
	pprint(results)

	with open("./src/params/colorspace.json", "w+") as f:
		f.write(dumps(results))


if __name__ == "__main__":
	main()
