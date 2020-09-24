import sys
import argparse

import keras.backend as K
from keras.callbacks import *

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_transformer.bin"

from ..utils import helper, visualize
from ..preprocessing import dataloader as dd
from ..models.transformer import transformer, transformer_inference, Transformer
from ..utils.eval import _beam_search, _decode_sequence, _make_src_seq_matrix
from ..utils.config import read_config_file

def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(description='Simple generation script for a Transformer network.')

    parser.add_argument('snapshot', help='Resume training from a snapshot.')
    parser.add_argument('vocab', help='Load an already existing vocabulary file.')
    parser.add_argument('--config', help='The configuration file.', default='config.ini')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)



    mfile = args.snapshot
    print(mfile)
    # load configs
    configs = read_config_file(args.config)

    i_tokens, o_tokens = dd.make_s2s_dict(None, dict_file=args.vocab)


    s2s = Transformer(i_tokens, o_tokens,**configs['init'])
    model, enc_attn, dec_self_attn, dec_attn = transformer(inputs=None, transformer_structure=s2s, return_att=True)
    # model = transformer_inference(model)

    try:
        model.load_weights(mfile)
        model.compile('adam', 'mse')
    except:
        print('\n\nModel not found or incompatible with network! Exiting now')
        exit(-1)

    start = time.clock()
    padded_line = helper.parenthesis_split('Move the red cube on top of the blue cube',
                             delimiter=" ", lparen="[", rparen="]")

    ret = _decode_sequence(model=model,
                           input_seq=padded_line,
                           i_tokens=i_tokens,
                           o_tokens=o_tokens,
                           len_limit=int(configs['init']['len_limit']))
    end = time.clock()
    print("Time per sequence: {} ".format((end - start)))
    print(ret)
    while True:
        quest = input('> ')
        padded_line = helper.parenthesis_split(quest,
                                               delimiter=" ", lparen="[", rparen="]")
        rets = _beam_search(
            model=model,
            input_seq=padded_line,
            i_tokens=i_tokens,
            o_tokens=o_tokens,
            len_limit=int(configs['init']['len_limit']),
            topk=1,
            delimiter=' ')
        for x, y in rets:
            print(x, y)

        # The part below is concerned with the data visualization
        src_seq = _make_src_seq_matrix(padded_line, i_tokens)
        tgt_padded_line = helper.parenthesis_split(x, delimiter=" ", lparen="[", rparen="]")
        tgt_seq = _make_src_seq_matrix(tgt_padded_line, o_tokens)
        # the last, second last and the first layer are the most interpretable. attention[2]-> dec_attn contains the input-output attention
        get_attention = K.function(model.input, [enc_attn[3], dec_self_attn[3], dec_attn[3]])
        attention = get_attention([src_seq, tgt_seq])
        # print('encoder attention', attention[0], 'decoder self attention', attention[1], 'decoder attention', attention[2])

        padded_line.insert(0, '<BOS>')
        padded_line.extend(['<EOS>', '<PAD>'])
        tgt_padded_line.insert(0, '<BOS>')
        tgt_padded_line.extend(['<EOS>'])
        visualize.plot_attention_map(tgt_padded_line, padded_line, np.transpose(attention[2]))
        # visualize.plot_attention_map(padded_line, tgt_padded_line, attention[2]) #np.transpose(np.mean(attention[2], axis=0))
        # end visualization


if __name__ == '__main__':
    main()