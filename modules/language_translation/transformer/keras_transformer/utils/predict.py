import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

from keras.callbacks import *

# TODO (fabawi): Might need to integrate fast decoders, although the difference is not that huge (keep it in mind)
# def make_fast_decode_model(self):
#     src_seq_input = Input(shape=(None,), dtype='int32')
#     tgt_seq_input = Input(shape=(None,), dtype='int32')
#     src_seq = src_seq_input
#     tgt_seq = tgt_seq_input
#
#     src_pos = Lambda(self.get_pos_seq)(src_seq)
#     tgt_pos = Lambda(self.get_pos_seq)(tgt_seq)
#     if not self.src_loc_info: src_pos = None
#     enc_output = self.encoder(src_seq, src_pos)
#     self.encode_model = Model(src_seq_input, enc_output)
#
#     enc_ret_input = Input(shape=(None, self.d_model))
#     dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_ret_input)
#     final_output = self.target_layer(dec_output)
#     self.decode_model = Model([src_seq_input, enc_ret_input, tgt_seq_input], final_output)
#
#     self.encode_model.compile('adam', 'mse')
#     self.decode_model.compile('adam', 'mse')
#
# def decode_sequence_fast(self, input_seq, delimiter=''):
#     if self.decode_model is None: self.make_fast_decode_model()
#     src_seq = self.make_src_seq_matrix(input_seq)
#     enc_ret = self.encode_model.predict_on_batch(src_seq)
#
#     decoded_tokens = []
#     target_seq = np.zeros((1, self.len_limit), dtype='int32')
#     target_seq[0, 0] = self.o_tokens.startid()
#     for i in range(self.len_limit - 1):
#         output = self.decode_model.predict_on_batch([src_seq, enc_ret, target_seq])
#         sampled_index = np.argmax(output[0, i, :])
#         sampled_token = self.o_tokens.token(sampled_index)
#         decoded_tokens.append(sampled_token)
#         if sampled_index == self.o_tokens.endid(): break
#         target_seq[0, i + 1] = sampled_index
#     return delimiter.join(decoded_tokens[:-1])
#


def _make_src_seq_matrix(input_seq, i_tokens, len_limit=None):
    if len_limit is None or len_limit <  len(input_seq)+3:
        src_seq = np.zeros((1, len(input_seq)+3), dtype='int32')
    else:
        src_seq = np.zeros((1, len_limit), dtype='int32')

    src_seq[0, 0] = i_tokens.startid()
    for i, z in enumerate(input_seq): src_seq[0, 1 + i] = i_tokens.id(z)
    src_seq[0, len(input_seq) + 1] = i_tokens.endid()
    return src_seq


def _decode_sequence(model, input_seq, i_tokens, o_tokens, len_limit, delimiter=''):
    src_seq = _make_src_seq_matrix(input_seq, i_tokens, len_limit)
    decoded_tokens = []
    target_seq = np.zeros((1, len_limit), dtype='int32')
    target_seq[0, 0] = o_tokens.startid()
    for i in range(len_limit - 1):
        output = model.predict_on_batch([src_seq, target_seq])
        sampled_index = np.argmax(output[0, i, :])
        sampled_token = o_tokens.token(sampled_index)
        decoded_tokens.append(sampled_token)
        if sampled_index == o_tokens.endid(): break
        target_seq[0, i + 1] = sampled_index
    processed_data = [src_seq[:, :len_limit], target_seq[:, :len_limit]]
    return delimiter.join(decoded_tokens[:-1]), processed_data


def _beam_search(model, input_seq, i_tokens, o_tokens, len_limit, topk=5, delimiter=''):
    src_seq = _make_src_seq_matrix(input_seq, i_tokens, len_limit)
    src_seq = src_seq.repeat(topk, 0)

    final_results = []
    decoded_tokens = [[] for _ in range(topk)]
    decoded_logps = [0] * topk
    lastk = 1
    target_seq = np.zeros((topk, len_limit), dtype='int32')
    target_seq[:, 0] = o_tokens.startid()
    for i in range(len_limit - 1):
        if lastk == 0 or len(final_results) > topk * 3: break
        output = model.predict_on_batch([src_seq, target_seq])
        output = np.exp(output[:, i, :])
        output = np.log(output / np.sum(output, -1, keepdims=True) + 1e-8)
        cands = []
        for k, wprobs in zip(range(lastk), output):
            if target_seq[k, i] == o_tokens.endid(): continue
            wsorted = sorted(list(enumerate(wprobs)), key=lambda x: x[-1], reverse=True)
            for wid, wp in wsorted[:topk]:
                cands.append((k, wid, decoded_logps[k] + wp))
        cands.sort(key=lambda x: x[-1], reverse=True)
        cands = cands[:topk]
        backup_seq = target_seq.copy()
        for kk, zz in enumerate(cands):
            k, wid, wprob = zz
            target_seq[kk,] = backup_seq[k]
            target_seq[kk, i + 1] = wid
            decoded_logps[kk] = wprob
            decoded_tokens.append(decoded_tokens[k] + [o_tokens.token(wid)])
            if wid == o_tokens.endid(): final_results.append((decoded_tokens[k], wprob))
        decoded_tokens = decoded_tokens[topk:]
        lastk = len(cands)
    processed_data = [src_seq[:, :len_limit], target_seq[:, :len_limit]]
    final_results = [(x, y / (len(x) + 1)) for x, y in final_results]
    final_results.sort(key=lambda x: x[-1], reverse=True)
    final_results = [(delimiter.join(x), y) for x, y in final_results]
    return final_results, processed_data


def predict(generator, model, command, beam_search=False, beam_width=5, save_path=None):
    output_sequences = []

    padded_line = generator.get_source_sequence(command=command)
    if beam_search:
        rets, processed_data = _beam_search(
            model=model,
            input_seq=padded_line,
            i_tokens= generator.i_tokens,
            o_tokens=generator.o_tokens,
            len_limit=generator.sequence_max_length,
            topk=beam_width,
            delimiter=' ')
        for x, y in rets:
            # print(x)
            output_sequences.append(x)
    else:
        rets, processed_data = _decode_sequence(
            model=model,
            input_seq=padded_line,
            i_tokens=generator.i_tokens,
            o_tokens=generator.o_tokens,
            len_limit=generator.sequence_max_length,
            delimiter=' ')
        # print(rets)
        output_sequences.append(rets)

    baseline_file = os.path.join(save_path, 'transformer_predictions.txt')
    with open(baseline_file, 'w') as fbase:
        for output_sequence in output_sequences:
            fbase.write("%s\n" % output_sequence)

    predictions = {"output_sequences": output_sequences}
    return predictions, processed_data

