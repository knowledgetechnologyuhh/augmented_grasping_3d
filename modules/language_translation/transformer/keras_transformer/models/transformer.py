from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf

from keras_transformer.utils.helper import input_2_bool

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head)
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn

class DilatedConvBlock():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1, dilation_rate=2, dilation_mode='linear', dilation_layers=3):
        # dilation_mode :
        # linear: linear dilation rate expansion;
        # non-linear: multiply the dilation rate by layer index + 1
        self.w_1 = []
        self.w_2 = []

        for dilation_layer_index, dilation_layer in enumerate(range(0, dilation_layers)):
            if dilation_mode == 'linear':
                dilation_factor = 1
            elif dilation_mode == 'non-linear':
                dilation_factor = dilation_layer_index + 1

            if dilation_layer_index == 0:
                self.w_1.append(Conv1D(d_inner_hid, 1, activation='relu'))
                self.w_2.append(Conv1D(d_hid, 1))
            else:
                self.w_1.append(Conv1D(d_inner_hid, 1, activation='relu', dilation_rate=dilation_rate*dilation_factor))
                self.w_2.append(Conv1D(d_hid, 1, dilation_rate=dilation_rate*dilation_factor))

        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        for w_1_index, w_1 in enumerate(self.w_1):
            if w_1_index == 0:
                output = w_1(x)
            else:
                output = w_1(output)

        for w_2 in self.w_2:
            output = w_2(output)

        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)

class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, dilation=False, **dilation_properties):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        if dilation:
            self.pos_ffn_layer = DilatedConvBlock(d_model, d_inner_hid, dropout=dropout, **dilation_properties)
        else:
            self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


class DecoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, dilation=False, **dilation_properties):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        if dilation:
            self.pos_ffn_layer = DilatedConvBlock(d_model, d_inner_hid, dropout=dropout, **dilation_properties)
        else:
            self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None):
        output, slf_attn = self.self_att_layer(dec_input, dec_input, dec_input, mask=self_mask)
        output, enc_attn = self.enc_att_layer(output, enc_output, enc_output, mask=enc_mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn, enc_attn


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def get_pad_mask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def get_sub_mask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, dilation=False, **dilation_properties):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.emb_dropout = Dropout(dropout)
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout, dilation, **dilation_properties) for _ in range(layers)]

    def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
        x = self.emb_layer(src_seq)
        if src_pos is not None:
            pos = self.pos_layer(src_pos)
            x = Add()([x, pos])
        x = self.emb_dropout(x)
        if return_att: atts = []
        mask = Lambda(lambda x: get_pad_mask(x, x))(src_seq)
        for enc_layer in self.layers[:active_layers]:
            # x_before = x
            # x_after, att = enc_layer(x_before, mask)
            # x = Add()([x_before, x_after])
            x, att = enc_layer(x, mask)
            if return_att: atts.append(att)
        return (x, atts) if return_att else x


class Decoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v,
                 layers=6, dropout=0.1, word_emb=None, pos_emb=None, dilation=False, **dilation_properties):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout, dilation, **dilation_properties) for _ in range(layers)]

    def __call__(self, tgt_seq, tgt_pos, src_seq, enc_output, return_att=False, active_layers=999):
        dec = self.emb_layer(tgt_seq)
        pos = self.pos_layer(tgt_pos)
        x = Add()([dec, pos])

        self_pad_mask = Lambda(lambda x: get_pad_mask(x, x))(tgt_seq)
        self_sub_mask = Lambda(get_sub_mask)(tgt_seq)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])

        enc_mask = Lambda(lambda x: get_pad_mask(x[0], x[1]))([tgt_seq, src_seq])

        if return_att: self_atts, enc_atts = [], []
        for dec_layer in self.layers[:active_layers]:
            # x_before = x
            # x_after, self_att, enc_att = dec_layer(x_before, enc_output, self_mask, enc_mask)
            # x = Add()([x_before, x_after])
            x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)
        return (x, self_atts, enc_atts) if return_att else x


class Transformer:
    def __init__(self, i_tokens, o_tokens, len_limit=100, d_model=256,
                 d_inner_hid=512, n_head=4, d_k=64, d_v=64, layers=2, dropout=0.1, i_embedding_matrix=None, o_embedding_matrix=None,
                 share_word_emb=False, dilation=False, **dilation_properties):
        len_limit = int(len_limit)
        d_model = int(d_model)
        d_inner_hid = int(d_inner_hid)
        n_head = int(n_head)
        d_k = int(d_k)
        d_v = int(d_v)
        layers = int(layers)
        dropout = float(dropout)
        share_word_emb = input_2_bool(share_word_emb)
        dilation = input_2_bool(dilation)
        if 'dilation_rate' in dilation_properties:
            dilation_properties['dilation_rate'] = int(dilation_properties['dilation_rate'])
        if 'dilation_layers' in dilation_properties:
            dilation_properties['dilation_layers'] = int(dilation_properties['dilation_layers'])

        self.i_tokens = i_tokens
        self.o_tokens = o_tokens
        self.len_limit = len_limit
        self.src_loc_info = True
        self.d_model = d_model
        self.decode_model = None
        d_emb = d_model

        pos_emb = Embedding(len_limit, d_emb, trainable=False,
                            weights=[get_pos_encoding_matrix(len_limit, d_emb)])

        if i_embedding_matrix is None:
            i_word_emb = Embedding(i_tokens.num(), d_emb)
        else:
            i_word_emb = Embedding(i_tokens.num(), d_emb, weights=[i_embedding_matrix])
        if share_word_emb:
            assert i_tokens.num() == o_tokens.num()
            o_word_emb = i_word_emb
        else:
            if o_embedding_matrix is None:
                o_word_emb = Embedding(o_tokens.num(), d_emb)
            else:
                o_word_emb = Embedding(o_tokens.num(), d_emb, weights=[o_embedding_matrix])

        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout,
                               word_emb=i_word_emb, pos_emb=pos_emb, dilation=dilation, **dilation_properties)
        self.decoder = Decoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout,
                               word_emb=o_word_emb, pos_emb=pos_emb, dilation=dilation, **dilation_properties)

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def make_src_seq_matrix(self, input_seq):
        src_seq = np.zeros((1, len(input_seq) + 3), dtype='int32')
        src_seq[0, 0] = self.i_tokens.startid()
        for i, z in enumerate(input_seq): src_seq[0, 1 + i] = self.i_tokens.id(z)
        src_seq[0, len(input_seq) + 1] = self.i_tokens.endid()
        return src_seq

def default_classification_layer(o_tokens_num, name='transformer_classification'):
    return TimeDistributed(Dense(o_tokens_num, use_bias=False), name=name)


def transformer_inference(model):
    src_seq = Input(shape=(None,), dtype='int32')
    tgt_seq = Input(shape=(None,), dtype='int32')
    model_built = model([src_seq, tgt_seq])

    classification = model_built[0]

    return Model([src_seq, tgt_seq], classification)


def transformer(transformer_structure, inputs=None, active_layers=999, sublayers=None, return_att=False, encoder_only=False):
    if inputs is None:
        src_seq_input = Input(shape=(None,), dtype='int32')
        tgt_seq_input = Input(shape=(None,), dtype='int32')
    elif isinstance(inputs[0], int) and isinstance(inputs[1], int):
        src_seq_input = Input(shape=(inputs[0],), dtype='int32')
        tgt_seq_input = Input(shape=(inputs[1],), dtype='int32')
    else:
        src_seq_input = inputs[0]
        tgt_seq_input = inputs[1]

    src_seq = src_seq_input
    tgt_seq = Lambda(lambda x: x[:, :-1])(tgt_seq_input)
    # tgt_true = Lambda(lambda x: x[:, 1:])(tgt_seq_input)
    src_pos = Lambda(transformer_structure.get_pos_seq)(src_seq)
    tgt_pos = Lambda(transformer_structure.get_pos_seq)(tgt_seq)
    if not transformer_structure.src_loc_info: src_pos = None

    if return_att:
        enc_output, enc_attn = transformer_structure.encoder(src_seq, src_pos, active_layers=active_layers, return_att=True)
        if encoder_only:
            dec_output = enc_output
        else:
            dec_output, dec_self_attn, dec_attn = transformer_structure.decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers, return_att=True)
    else:
        enc_output = transformer_structure.encoder(src_seq, src_pos, active_layers=active_layers)
        if encoder_only:
            dec_output = enc_output
        else:
            dec_output = transformer_structure.decoder(tgt_seq, tgt_pos, src_seq, enc_output, active_layers=active_layers)
    if sublayers is not None:
        final_output = []
        for sublayer in sublayers:
            final_output.append(sublayer[1](dec_output))

    else:
        final_output = default_classification_layer(transformer_structure.o_tokens.num())(dec_output)
    if return_att:
        return Model([src_seq_input, tgt_seq_input], final_output), enc_attn, dec_self_attn, dec_attn
    else:
        return Model([src_seq_input, tgt_seq_input], final_output)
