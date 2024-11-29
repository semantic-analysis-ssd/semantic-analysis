# import argparse
# import pandas as pd
# from pathlib import Path
# from clone_classifier import CloneClassifier


# ROOT = Path(__file__).parent


# def main():

#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument(
#         "--max_token_size", type=int, metavar="", default=512, help="Max token size"
#     )
#     parser.add_argument("--fp16", action="store_true", help="Enable half-precision")
#     parser.add_argument(
#         "-i",
#         "--input",
#         type=str,
#         metavar="",
#         default=f"{ROOT / 'examples/c4.csv'}",
#         help="Input file path",
#     )
#     parser.add_argument(
#         "-o",
#         "--output",
#         type=str,
#         metavar="",
#         default=f"{ROOT / 'results/res.csv'}",
#         help="Output file path",
#     )
#     parser.add_argument(
#         "--per_device_eval_batch_size",
#         type=int,
#         metavar="",
#         default=32,
#         help="Batch size per device for evaluation",
#     )

#     classifier_args = vars(parser.parse_args())
#     input_path = classifier_args.pop("input")
#     output_path = classifier_args.pop("output")

#     clone_classifier = CloneClassifier(**classifier_args)

#     df = pd.read_csv(input_path).iloc[:1]
#     if(len(df)==1):
#         df = pd.concat([df, df], ignore_index=True)
#         clone_classifier.predict_1(df, save_path=output_path)
#     else:
#         clone_classifier.predict(df, save_path=output_path)

#     return


# if __name__ == "__main__":
#     main()




# # import argparse
# # import pandas as pd
# # from pathlib import Path
# # from clone_classifier import CloneClassifier


# # ROOT = Path(__file__).parent


# # def main():
# #     """Run the clone classifier on specified input file.

# #     By default, the input file is the C4 dataset located in `examples/c4.csv`,
# #     and its predictions output is saved as `results/res.csv`.

# #     Args:
# #         max_token_size: the maximum token size for the input data (default: 512)
# #         fp16: a flag to enable half-precision (default: False)
# #         input: the path to the input CSV file (default: 'examples/c4.csv')
# #         output: the path to save the output CSV file (default: 'results/res.csv')
# #         per_device_eval_batch_size: the batch size for evaluation per device (default: 32)
# #     """
# #     parser = argparse.ArgumentParser(
# #         formatter_class=argparse.ArgumentDefaultsHelpFormatter
# #     )
# #     parser.add_argument(
# #         "--max_token_size", type=int, metavar="", default=512, help="Max token size"
# #     )
# #     parser.add_argument("--fp16", action="store_true", help="Enable half-precision")
# #     parser.add_argument(
# #         "-i",
# #         "--input",
# #         type=str,
# #         metavar="",
# #         default=f"{ROOT / 'examples/c4.csv'}",
# #         help="Input file path",
# #     )
# #     parser.add_argument(
# #         "-o",
# #         "--output",
# #         type=str,
# #         metavar="",
# #         default=f"{ROOT / 'results/res.csv'}",
# #         help="Output file path",
# #     )
# #     parser.add_argument(
# #         "--per_device_eval_batch_size",
# #         type=int,
# #         metavar="",
# #         default=32,
# #         help="Batch size per device for evaluation",
# #     )

# #     classifier_args = vars(parser.parse_args())
# #     input_path = classifier_args.pop("input")
# #     output_path = classifier_args.pop("output")

# #     clone_classifier = CloneClassifier(**classifier_args)

# #     df = pd.read_csv(input_path,on_bad_lines='skip')
# #     clone_classifier.predict(df, save_path=output_path)

# #     return


# # if __name__ == "__main__":
# #     main()

import argparse
import os
import pandas as pd
from pathlib import Path
from clone_classifier import CloneClassifier

ROOT = Path(__file__).parent


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--max_token_size", type=int, metavar="", default=512, help="Max token size"
    )
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision")
    parser.add_argument(
        "-i",
        "--input_code_file",
        type=str,
        metavar="",
        required=True,
        help="Input code file path (code1.txt)",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        metavar="",
        required=True,
        help="Folder path containing the code files to compare against",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        metavar="",
        default=f"{ROOT / 'results/top_k_similar_codes.csv'}",
        help="Output file path for top k similar code results",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        metavar="",
        default=32,
        help="Batch size per device for evaluation",
    )
    parser.add_argument(
        "--k",
        type=int,
        metavar="",
        default=5,
        help="Number of top similar codes to return",
    )

    classifier_args = vars(parser.parse_args())
    input_code_file = classifier_args.pop("input_code_file")
    folder_path = classifier_args.pop("folder")
    output_path = classifier_args.pop("output")
    k = classifier_args.pop("k")

    clone_classifier = CloneClassifier(**classifier_args)

    # Step 1: Read the code from the input file (code1.txt)
    with open(input_code_file, 'r') as file:
        target_code = file.read()

    # Step 2: Read all code files from the folder
    code_repository = []
    for file_name in os.listdir(folder_path):
        file_path = Path(folder_path) / file_name
        if file_path.is_file() and file_path.suffix == ".txt":  # assuming .txt files
            with open(file_path, 'r') as f:
                content = f.read()
                if not content.strip():
                    content=""
                code_repository.append(content)

    # Step 3: Get the top k similar codes using the CloneClassifier
    top_k_results = clone_classifier.get_top_k_similar(target_code, code_repository, k=k)

    # Step 4: Save the results to a CSV file
    if output_path:
        top_k_results.to_csv(output_path, index=False)

    # Print the results
    print("Top K Similar Codes:")
    print(top_k_results)


if __name__ == "__main__":
    main()
