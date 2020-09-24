import sys
import argparse
from os.path import basename, splitext
from tqdm import tqdm

from keras.callbacks import *

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_transformer.bin"

from ..utils import helper
from ..preprocessing import dataloader as dd
from e2emetrics import measure_scores
from ..models.transformer import transformer, transformer_inference, Transformer
from ..utils.eval import _beam_search, _decode_sequence
from ..utils.config import read_config_file

def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Evaluation script which generates sentences and then evaluates using metrics: BLEU, NIST, ROUGE,_L, CIDEr, METEOR.')

    parser.add_argument('snapshot', help ='Load a snapshot.')
    parser.add_argument('val_annotations', help='Path to validation source and target sequences file (both separated by a tab).')
    parser.add_argument('vocab', help='Load an already existing vocabulary file.')
    parser.add_argument('--val-golden-set', help='Path to the human annotated golden set for validation.')
    parser.add_argument('--evaluate-metrics',
                        help='If set to true, will evaluate on e2e-metrics and saves the output results.', action='store_true',
                        default=False)
    parser.add_argument('--generate',
                        help='Generates and saves the output sentence to the snapshot directory.', action='store_true',
                        default=False)
    parser.add_argument('--beam-search',
                        help='If set to true,returns the best beam search output.', action='store_true',
                        default=False)
    parser.add_argument('--beam-width',
                        help='Size of the beam width if beam search is used.',
                        default=5)
    parser.add_argument('--verbose',
                        help='Setting the verbosity flag prints the output.', action='store_true',
                        default=False)
    parser.add_argument('--config',
                        help='The configuration file.', default='config.ini')
    parser.add_argument('--log-path',
                        help='The logging directory including.',
                        default='../../logs')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    result_path = helper.make_dir(args.log_path)
    mfile = args.snapshot
    baseline_file = result_path + 'prediction.txt'

    # load configs
    configs = read_config_file(args.config)

    i_tokens, o_tokens = dd.make_s2s_dict(None, dict_file=args.vocab)

    s2s = Transformer(i_tokens, o_tokens,**configs['init'])
    model = transformer(inputs=None, transformer_structure=s2s)
    # model = transformer_inference(model)
    try:
        model.load_weights(mfile)
        model.compile('adam', 'mse')


    except:
        print('\n\nModel not found or incompatible with network! Exiting now')
        exit(-1)

    # if args.create_golden:
    #     prepare_evaluation.create_golden_sentences(args.valid_file, golden_file)

    with open(args.val_annotations, 'r') as fval:
        lines = fval.readlines()

    if args.generate:
        outputs = []
        lines = lines[1:] # skip the first line
        for line_raw in tqdm(lines, mininterval=2, desc='  - (Test)', leave=False):
            line_raw = line_raw.split('\t')
            padded_line = helper.parenthesis_split(line_raw[0], delimiter=' ', lparen="[", rparen="]")
            if args.beam_search:
                rets = _beam_search(
                    model=model,
                    input_seq=padded_line,
                    i_tokens=i_tokens,
                    o_tokens=o_tokens,
                    len_limit=int(configs['init']['len_limit']),
                    topk=args.beam_width,
                    delimiter=' ')
                for x, y in rets:
                    if args.verbose:
                        print(x)
                    outputs.append(x)
                    break
            else:
                ret = _decode_sequence(model=model,
                                       input_seq=padded_line,
                                       i_tokens=i_tokens,
                                       o_tokens=o_tokens,
                                       len_limit=int(configs['init']['len_limit']), delimiter=' ')
                if args.verbose:
                    print(ret)
                outputs.append(ret)

        with open(baseline_file, 'w') as fbase:
            for output in outputs:
                fbase.write("%s\n" % output)
        del outputs

    if args.evaluate_metrics:
        golden_file = args.val_golden_set
        data_src, data_ref, data_sys = measure_scores.load_data(golden_file, baseline_file, None)
        measure_names, scores = measure_scores.evaluate(data_src, data_ref, data_sys)
        print(scores)
        helper.store_settings(scores.__repr__(), result_path + 'metric_results.txt')


if __name__ == '__main__':
    main()