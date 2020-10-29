from note_labeler.initialize import create_datasets, get_model
import argparse
import ast
import logging

def init_app(data_path, schema, label_encoder, artifacts_path, encoding, sample_size):
    """Short summary.

    :param type data_path: Description of parameter `data_path`.
    :param type schema: Description of parameter `schema`.
    :param type label_encoder: Description of parameter `label_encoder`.
    :param type artifacts_path: Description of parameter `artifacts_path`.
    :param type encoding: Description of parameter `encoding`.
    :param type sample_size: Description of parameter `sample_size`.
    :return: Description of returned object.
    :rtype: type

    """
    if type(label_encoder)==str:
        label_encoder=ast.literal_eval(label_encoder)
    if type(schema)==str:
        schema=ast.literal_eval(schema)
    ds=create_datasets.RawDataset(data_path, schema, label_encoder, artifacts_path, encoding=encoding, sample_size=sample_size)
    ds.gen_artifacts()
    get_model.init_model(artifacts_path, encoding)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Setup files to run app")

    parser.add_argument(
        "-d",
        "--data_path",
        required=True
    )

    parser.add_argument(
        "-s",
        "--schema",
        help='Dict that maps label to "label", note text to "review"',
        required=True
    )

    parser.add_argument(
        "-l",
        "--label_encoder",
        help='dict that maps the string or int labels as appears in the data to 1 and 0, Ex: {"pos":1, "neg":0}',
        required=True
    )

    parser.add_argument(
        "-a",
        "--artifacts_path",
        help="path for the artifacts to be saved",
        required=True
    )
    parser.add_argument(
        "-e",
        "--encoding",
        help="csv encoding",
        required=False,
        default='utf-8'
    )
    parser.add_argument(
        "-f",
        "--sample_size",
        help="sample",
        required=False,
        default=None
    )

    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    init_app(**vars(args))
