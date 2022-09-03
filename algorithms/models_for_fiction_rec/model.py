import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from algorithms.models_for_fiction_rec.bilinear_attention import *


def gru_forward(gru, input, lengths, state=None, batch_first=True):
    gru.flatten_parameters()
    input_lengths, perm = torch.sort(lengths, descending=True)

    input = input[perm]
    if state is not None:
        state = state[perm].transpose(0, 1).contiguous()

    total_length=input.size(1)
    if not batch_first:
        input = input.transpose(0, 1)  # B x L x N -> L x B x N
    packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first)

    outputs, state = gru(packed, state)
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first, total_length=total_length)  # unpack (back to padded)

    _, perm = torch.sort(perm, descending=False)
    if not batch_first:
        outputs = outputs.transpose(0, 1)
    outputs=outputs[perm]
    state = state.transpose(0, 1)[perm]

    return outputs, state


def build_map(b_map, max=None):
    batch_size, b_len = b_map.size()
    if max is None:
        max=b_map.max() + 1
    if torch.cuda.is_available():
        b_map_ = torch.cuda.FloatTensor(batch_size, b_len, max).fill_(0)
    else:
        b_map_ = torch.zeros(batch_size, b_len, max)
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad=False
    return b_map_


class CNNText(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class CNNEncoder(nn.Module):

    def __init__(self, attribute_embedding_dim):
        super().__init__()
        Ci = 1
        Ks = [1, 2, 3, 4, 5, 6]
        Cos = [int(attribute_embedding_dim / len(Ks))] * (len(Ks) - 1)
        Cos.append(attribute_embedding_dim - sum(Cos))
        convs = []
        for i, Co in enumerate(Cos):
            K = Ks[i]
            convs.append(nn.Conv2d(Ci, Co, (K, attribute_embedding_dim)))
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        x = input.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        return x


class GRUEncoder(nn.Module):

    def __init__(self, attribute_embedding_dim):
        super().__init__()
        self.enc = nn.GRU(attribute_embedding_dim, int(attribute_embedding_dim / 2), num_layers=1, bidirectional=True,
                          batch_first=True)

    def forward(self, x):
        _, last_hidden = self.enc(x)
        representation = torch.cat([last_hidden[0], last_hidden[1]], dim=-1)
        return representation


class AttributesEncoder(nn.Module):
    def __init__(self, encoder_name: str = 'gru', attribute_embedding_dim=32):
        """

        @param encoder_name:
        """
        super().__init__()
        self.encoder_name = encoder_name
        if encoder_name == 'gru':
            self.enc = GRUEncoder(attribute_embedding_dim)
        elif encoder_name == 'cnn':
            self.enc = CNNEncoder(attribute_embedding_dim)
        else:
            raise NotImplementedError(encoder_name)

    def forward(self, input: torch.Tensor):
        """

        @param input:
        @return:
        """
        input_shape = input.shape
        input_flatten = input.reshape((-1, input_shape[-2], input_shape[-1]))
        representation = self.enc(input_flatten)
        result = representation.reshape((input_shape[0], input_shape[1], input_shape[3]))

        # samples = []
        # for i in range(input.shape[0]):
        #     representation = self.enc(input[i])
        #     samples.append(representation)
        # result = torch.cat(samples, dim=0)
        return result


class NovelNet(nn.Module):
    def __init__(self, embedding_size, hidden_size, item_vocab_size,
                 max_feature_index, features, attribute_embedding_size, mil, mode_loss=False, remove_repeat_mode=False,
                 dot_product=False, encoder_name=None, my_method=None, remove_mode=False, model_version='v1',
                 other_args: dict = None):
        """
        todo 负采样；用所有其它样本作为负样本的问题是，并不是所有其它样本都是真正的负样本
        todo 统计之前的方法推荐结果中，重复推荐的比例
        :param embedding_size:
        :param hidden_size:
        :param item_vocab_size:
        :param max_feature_index:
        :param features:
        :param attribute_embedding_size:
        :param mil:
        :param mode_loss:
        :param remove_repeat_mode:
        """
        super(NovelNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.item_vocab_size = item_vocab_size
        self.max_feature_index = max_feature_index
        self.features = features
        self.attribute_embedding_size = attribute_embedding_size
        self.mil = mil
        self.mode_loss = mode_loss
        self.remove_repeat_mode = remove_repeat_mode
        self.dot_product = dot_product
        if self.dot_product:
            self.hidden_size = embedding_size
            hidden_size = self.hidden_size
        self.encoder_name = encoder_name
        self.my_method = my_method
        self.remove_mode = remove_mode
        self.model_version = model_version
        self.other_args = other_args

        self.item_emb = nn.Embedding(item_vocab_size, embedding_size, padding_idx=0)

        # attribute embeddings
        if len(self.features) != 0:
            self.attribute_embs = torch.nn.ModuleDict()
            for feature in features:
                feature_emb = nn.Embedding(max_feature_index[feature] + 1, attribute_embedding_size, padding_idx=0)
                self.attribute_embs[feature] = feature_emb

        if self.encoder_name is not None:
            input_size_of_gru = embedding_size + attribute_embedding_size
            self.attribute_encoder = AttributesEncoder(attribute_embedding_dim=attribute_embedding_size,
                                                       encoder_name=self.encoder_name)
        else:
            input_size_of_gru = embedding_size + len(features) * attribute_embedding_size
        if 'bidirectional' in self.other_args and not self.other_args['bidirectional']:
            self.enc = nn.GRU(input_size_of_gru, hidden_size, num_layers=1, bidirectional=False,
                              batch_first=True)
        else:
            self.enc = nn.GRU(input_size_of_gru, int(hidden_size / 2), num_layers=1, bidirectional=True, batch_first=True)

        self.mode_attn = BilinearAttention(hidden_size, hidden_size, hidden_size)
        self.mode = nn.Linear(hidden_size, 2)

        self.repeat_attn = BilinearAttention(hidden_size, hidden_size, hidden_size)
        self.explore_attn = BilinearAttention(hidden_size, hidden_size, hidden_size)
        if not self.dot_product:
            self.explore = nn.Linear(hidden_size, item_vocab_size)
        if self.mil or self.my_method is not None or self.model_version in ('v2', 'v3', 'v4'):
            self.repeat_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1)
            )

    def model(self, data, train=True):
        if self.model_version == 'v1':
            batch_size=data['item_seq'].size(0)
            mask = data['item_seq'].ne(0)
            lengths = mask.float().sum(dim=-1).long()

            if len(self.features) != 0:
                embeddings = [self.item_emb(data['item_seq'])]
                if self.encoder_name is not None:
                    # todo 更有效的建模这些特征，形成层次模型 cnn/attention/gru/transformer，这些特征一起表达一个完整语义
                    attribute_embeddings = []
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        attribute_embeddings.append(feature_embeddings)
                    attribute_embeddings = torch.cat([e.unsqueeze(dim=-2) for e in attribute_embeddings], dim=-2)
                    attribute_representation = self.attribute_encoder(attribute_embeddings)
                    embeddings.append(attribute_representation)
                    embeddings = torch.cat(embeddings, dim=-1)
                else:
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        embeddings.append(feature_embeddings)
                    embeddings = torch.cat(embeddings, dim=-1)
            else:
                embeddings = self.item_emb(data['item_seq'])

            item_seq_embs = F.dropout(embeddings, p=0.5, training=self.training)

            output, state = gru_forward(self.enc, item_seq_embs, lengths, batch_first=True)
            state = F.dropout(state, p=0.5, training=self.training)
            output = F.dropout(output, p=0.5, training=self.training)

            explore_feature, attn, norm_attn = self.explore_attn(state.reshape(batch_size, -1).unsqueeze(1), output, output,
                                                                 mask=mask.unsqueeze(1))

            if self.dot_product:
                p_explore_temp = torch.matmul(explore_feature.squeeze(1), torch.transpose(self.item_emb.weight, 0, 1))
            else:
                p_explore_temp = self.explore(explore_feature.squeeze(1))
            # 没有移除重复推荐模块时，explore模块不需要推荐历史消费item
            if not self.remove_repeat_mode:
                explore_mask = torch.bmm((data['item_seq'] > 0).float().unsqueeze(1), data['source_map']).squeeze(1)
                p_explore_temp = p_explore_temp.masked_fill(explore_mask.bool(), float('-inf')) # not sure we need to mask this out, depends on experiment results
            if self.my_method is not None:
                p_explore = p_explore_temp
            else:
                p_explore = F.softmax(p_explore_temp, dim=-1)

            if not self.remove_repeat_mode:
                if self.mil:
                    # todo 单独一个分类器用于预测一个item接下来是否会被重复消费？
                    # softmax 类似与listwise优化，可能对排序更友好？
                    # todo 由于消费历史较少，在消费历史上进行softmax会不会使得历史item的分数自然偏高？
                    p_repeat_logit, p_repeat_temp = self.repeat_attn.score(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                              mask=mask.unsqueeze(1))
                    p_repeat_logit_normalized = F.sigmoid(p_repeat_logit).squeeze(dim=1) * data['item_mask']

                    # p_repeat_temp_logit = self.repeat_classifier(output).squeeze(dim=-1)
                    # p_repeat_temp = torch.sigmoid(p_repeat_temp_logit)
                    #
                    # p_repeat_temp_logit = p_repeat_temp_logit * data['item_mask']
                    # p_repeat_temp = (p_repeat_temp * data['item_mask']).unsqueeze(1)

                    p_repeat = torch.bmm(p_repeat_temp, data['source_map']).squeeze(1)

                    max_repeat_item_index = torch.argmax(p_repeat_logit_normalized, dim=-1)
                    repeat_mode = torch.gather(p_repeat_logit_normalized.squeeze(dim=1), 1,
                                               max_repeat_item_index.unsqueeze(dim=1))
                    explore_mode = 1 - repeat_mode
                    p_mode = torch.cat([explore_mode, repeat_mode], dim=-1)
                elif self.my_method is not None:
                    p_repeat_temp_logit = self.repeat_classifier(output).squeeze(dim=-1)

                    p_repeat_temp = (p_repeat_temp_logit * data['item_mask']).unsqueeze(1)

                    p_repeat = torch.bmm(p_repeat_temp, data['source_map']).squeeze(1)
                else:
                    _, p_repeat_temp = self.repeat_attn.score(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                              mask=mask.unsqueeze(1))
                    p_repeat = torch.bmm(p_repeat_temp, data['source_map']).squeeze(1)

                    mode_feature, attn, norm_attn = self.mode_attn(state.reshape(batch_size, -1).unsqueeze(1), output, output,
                                                                   mask=mask.unsqueeze(1))
                    # todo 所有交互过的item共享相同的模式分不合理，导致并不该再被推荐的item继续占据推荐列别的前面位置
                    # todo 可以先对所有item进行预测，模式基于历史交互过的item的情况可以放大或者缩小历史交互过的item的推荐概率
                    p_mode = F.softmax(self.mode(mode_feature.squeeze(1)), dim=-1)

                if self.remove_mode:
                    p = p_explore + p_repeat
                    if self.my_method is not None:
                        p = F.softmax(p, dim=-1)
                    return p, None
                else:
                    p = p_mode[:, 0].unsqueeze(-1) * p_explore + p_mode[:, 1].unsqueeze(-1) * p_repeat
                    return p, p_mode[:, 1]
            else:
                return p_explore, None
        elif self.model_version == 'v2':
            batch_size = data['item_seq'].size(0)
            mask = data['item_seq'].ne(0)
            lengths = mask.float().sum(dim=-1).long()

            id_embeddings = self.item_emb(data['item_seq'])
            if len(self.features) != 0:
                embeddings = [id_embeddings]
                for feature in self.features:
                    feature_embeddings = self.attribute_embs[feature](data[feature])
                    embeddings.append(feature_embeddings)
                embeddings = torch.cat(embeddings, dim=-1)
            else:
                embeddings = id_embeddings

            item_seq_embs = F.dropout(embeddings, p=0.5, training=self.training)

            output, state = gru_forward(self.enc, item_seq_embs, lengths, batch_first=True)
            state = F.dropout(state, p=0.5, training=self.training)
            output = F.dropout(output, p=0.5, training=self.training)

            explore_feature, attn, norm_attn = self.explore_attn(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                                 output,
                                                                 mask=mask.unsqueeze(1))

            if self.dot_product:
                p_explore_temp = torch.matmul(explore_feature.squeeze(1), torch.transpose(self.item_emb.weight, 0, 1))
            else:
                p_explore_temp = self.explore(explore_feature.squeeze(1))

            # 缩放历史交互过的item的概率，有的需要变大，有的要缩小；可加监督信号，后面还会出现的item变大，否则变小
            scale_weights = self.repeat_classifier(output).squeeze(dim=-1)
            scale_weights = torch.exp(scale_weights)

            # todo 这里是不是不用item_mask，而用一个损失表明哪个权重较大，其它都小更为合理？
            scale_weights = (scale_weights * data['item_mask']).unsqueeze(1)

            scale_weights = torch.bmm(scale_weights, data['source_map']).squeeze(1)

            # 保持没有交互过的item的值不变
            scale_mask = torch.bmm((data['item_seq'] > 0).float().unsqueeze(1), data['source_map']).squeeze(1)
            scale_weights = scale_weights.masked_fill(~scale_mask.bool(), 1)

            p = p_explore_temp * scale_weights
            p = F.softmax(p, dim=-1)
            return p, None
        elif self.model_version == 'v3':
            batch_size = data['item_seq'].size(0)
            mask = data['item_seq'].ne(0)
            lengths = mask.float().sum(dim=-1).long()

            item_id_embeddings = self.item_emb(data['item_seq'])
            if len(self.features) != 0:
                embeddings = [item_id_embeddings]
                if self.encoder_name is not None:
                    # todo 更有效的建模这些特征，形成层次模型 cnn/attention/gru/transformer，这些特征一起表达一个完整语义
                    attribute_embeddings = []
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        attribute_embeddings.append(feature_embeddings)
                    attribute_embeddings = torch.cat([e.unsqueeze(dim=-2) for e in attribute_embeddings], dim=-2)
                    attribute_representation = self.attribute_encoder(attribute_embeddings)
                    embeddings.append(attribute_representation)
                    embeddings = torch.cat(embeddings, dim=-1)
                else:
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        embeddings.append(feature_embeddings)
                    embeddings = torch.cat(embeddings, dim=-1)
            else:
                embeddings = item_id_embeddings

            item_seq_embs = F.dropout(embeddings, p=0.5, training=self.training)

            output, state = gru_forward(self.enc, item_seq_embs, lengths, batch_first=True)
            state = F.dropout(state, p=0.5, training=self.training)
            output = F.dropout(output, p=0.5, training=self.training)

            if 'explore_feature' in self.other_args and self.other_args['explore_feature'] == 'state':
                explore_feature = state
            elif 'explore_feature' in self.other_args and self.other_args['explore_feature'] == 'merge':
                attention_explore_feature, _, _ = self.explore_attn(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                                    output,
                                                                    mask=mask.unsqueeze(1))
                explore_feature = attention_explore_feature + state
            else:
                explore_feature, attn, norm_attn = self.explore_attn(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                                     output,
                                                                     mask=mask.unsqueeze(1))
            if self.dot_product:
                p_explore_temp = torch.matmul(explore_feature.squeeze(1), torch.transpose(self.item_emb.weight, 0, 1))
            else:
                p_explore_temp = self.explore(explore_feature.squeeze(1))

            # 没有移除重复推荐模块时，explore模块不需要推荐历史消费item
            explore_mask = torch.bmm((data['item_seq'] > 0).float().unsqueeze(1), data['source_map']).squeeze(1)
            p_explore_temp = p_explore_temp.masked_fill(explore_mask.bool(), float('-inf'))
            p_explore = F.softmax(p_explore_temp, dim=-1)

            # todo 叫loss，或者缩放因子，或者都加
            # 当非重复推荐时，所有值都为0
            if 'repeat_mask' in self.other_args and self.other_args['repeat_mask'] == 'item_mask':
                repeat_mask = data['item_mask'].ne(0)
            else:
                repeat_mask = mask
            p_repeat_raw, p_repeat_temp = self.repeat_attn.score(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                                 mask=repeat_mask.unsqueeze(1), activation='softmax')
            p_repeat_sigmoid = F.sigmoid(p_repeat_raw)
            # p_repeat_sigmoid = p_repeat_sigmoid.masked_fill(~repeat_mask.unsqueeze(1), 0)

            if 'scale' in self.other_args and self.other_args['scale']:
                # 缩放历史交互过的item的概率，有的需要变大，有的要缩小；可加监督信号，后面还会出现的item变大，否则变小
                scale_weights = self.repeat_classifier(output).squeeze(dim=-1)
                scale_weights = torch.sigmoid(scale_weights)
                p_repeat_temp = (p_repeat_temp.squeeze(dim=1) * scale_weights).unsqueeze(dim=1)

                p_repeat_sigmoid = scale_weights

            p_repeat = torch.bmm(p_repeat_temp, data['source_map']).squeeze(1)

            p = p_explore + p_repeat

            if 'repeat_label' in self.other_args and self.other_args['repeat_label'] == 'sigmoid':
                p_repeat_prediction = p_repeat_sigmoid
            else:
                p_repeat_prediction = p_repeat_temp

            if 'global_loss' in self.other_args and self.other_args['global_loss']:
                p_repeat_raw = torch.bmm(p_repeat_raw, data['source_map']).squeeze(1)
                p_global = p_explore_temp + p_repeat_raw
                p_global = F.softmax(p_global, dim=-1)
                return p, p_repeat_prediction, p_global
            else:
                return p, p_repeat_prediction
        elif self.model_version == 'v4':
            batch_size = data['item_seq'].size(0)
            mask = data['item_seq'].ne(0)
            lengths = mask.float().sum(dim=-1).long()

            item_id_embeddings = self.item_emb(data['item_seq'])
            if len(self.features) != 0:
                embeddings = [item_id_embeddings]
                if self.encoder_name is not None:
                    # todo 更有效的建模这些特征，形成层次模型 cnn/attention/gru/transformer，这些特征一起表达一个完整语义
                    attribute_embeddings = []
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        attribute_embeddings.append(feature_embeddings)
                    attribute_embeddings = torch.cat([e.unsqueeze(dim=-2) for e in attribute_embeddings], dim=-2)
                    attribute_representation = self.attribute_encoder(attribute_embeddings)
                    embeddings.append(attribute_representation)
                    embeddings = torch.cat(embeddings, dim=-1)
                else:
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        embeddings.append(feature_embeddings)
                    embeddings = torch.cat(embeddings, dim=-1)
            else:
                embeddings = item_id_embeddings

            item_seq_embs = F.dropout(embeddings, p=0.5, training=self.training)

            output, state = gru_forward(self.enc, item_seq_embs, lengths, batch_first=True)
            state = F.dropout(state, p=0.5, training=self.training)
            output = F.dropout(output, p=0.5, training=self.training)

            explore_feature, _, _ = self.explore_attn(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                      output,
                                                      mask=mask.unsqueeze(1))

            if self.dot_product:
                p_explore_temp_not_masked = torch.matmul(explore_feature.squeeze(1), torch.transpose(self.item_emb.weight, 0, 1))
            else:
                p_explore_temp_not_masked = self.explore(explore_feature.squeeze(1))

            # 没有移除重复推荐模块时，explore模块不需要推荐历史消费item
            explore_mask = torch.bmm((data['item_seq'] > 0).float().unsqueeze(1), data['source_map']).squeeze(1)
            p_explore_temp = p_explore_temp_not_masked.masked_fill(explore_mask.bool(), float('-inf'))
            p_explore = F.softmax(p_explore_temp, dim=-1)

            # todo 叫loss，或者缩放因子，或者都加
            # 当非重复推荐时，所有值都为0
            if 'repeat_mask' in self.other_args and self.other_args['repeat_mask'] == 'item_mask':
                repeat_mask = data['item_mask'].ne(0)
            else:
                repeat_mask = mask
            p_repeat_raw, p_repeat_temp = self.repeat_attn.score(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                                 mask=repeat_mask.unsqueeze(1), activation='softmax')
            p_repeat_sigmoid = F.sigmoid(p_repeat_raw)
            # p_repeat_sigmoid = p_repeat_sigmoid.masked_fill(~repeat_mask.unsqueeze(1), 0)

            if not train:
                if 'decoder_activation_function' in self.other_args \
                        and self.other_args['decoder_activation_function'] == 'softmax':
                    max_indices = torch.sort(p_repeat_temp, dim=-1, descending=True)[1][:, :, 0]
                    p_repeat_temp = p_repeat_temp.masked_fill(~F.one_hot(max_indices, num_classes=49).ne(0), 0)
                elif 'decoder_activation_function' in self.other_args \
                        and self.other_args['decoder_activation_function'] == 'sigmoid':
                    max_indices = torch.sort(p_repeat_sigmoid, dim=-1, descending=True)[1][:, :, 0]
                    p_repeat_temp = p_repeat_sigmoid.masked_fill(~F.one_hot(max_indices, num_classes=49).ne(0), 0)
                elif 'decoder_activation_function' in self.other_args \
                        and self.other_args['decoder_activation_function'] == 'threshold':
                    threshold = self.other_args['threshold']
                    threshold_mask = p_repeat_sigmoid < threshold
                    p_repeat_temp = p_repeat_sigmoid.masked_fill(threshold_mask, 0)
                elif 'decoder_activation_function' in self.other_args \
                        and self.other_args['decoder_activation_function'] == 'repeat_k':
                    repeat_k = self.other_args['repeat_k']
                    if self.other_args['repeat_k_type'] == 'sigmoid':
                        repeat_input = p_repeat_sigmoid
                    else:
                        repeat_input = p_repeat_temp
                    target_indices = torch.sort(repeat_input, dim=-1, descending=True)[1]
                    final = None
                    for i in range(repeat_k):
                        max_indices = target_indices[:, :, i]
                        temp = repeat_input.masked_fill(~F.one_hot(max_indices, num_classes=49).ne(0), 0)
                        if final is None:
                            final = temp
                        else:
                            final += temp
                    p_repeat_temp = final
                elif 'decoder_activation_function' in self.other_args \
                        and self.other_args['decoder_activation_function'] == 'explore_only':
                    p_repeat_temp.fill_(0.0)
                elif 'decoder_activation_function' in self.other_args \
                        and self.other_args['decoder_activation_function'] == 'last_n':
                    if self.other_args['last_n_type'] == 'sigmoid':
                        repeat_input = p_repeat_sigmoid
                    else:
                        repeat_input = p_repeat_temp

                    if self.other_args['only_last_n']:
                        p_repeat_temp = data['candidate_mask'].unsqueeze(dim=1)
                    else:
                        p_repeat_temp = (repeat_input * data['candidate_mask'].unsqueeze(dim=1))
                else:
                    pass
            p_repeat = torch.bmm(p_repeat_temp, data['source_map']).squeeze(1)

            if train and 'repeat_loss' in self.other_args and self.other_args['repeat_loss'] == 'point_only':
                p = F.softmax(p_explore_temp_not_masked, dim=-1)
            else:
                p = p_explore + p_repeat

            return p, p_repeat_sigmoid
        elif self.model_version == 'v5':
            # 不单独建模重复消费行为
            batch_size = data['item_seq'].size(0)
            mask = data['item_seq'].ne(0)
            lengths = mask.float().sum(dim=-1).long()

            item_id_embeddings = self.item_emb(data['item_seq'])
            if len(self.features) != 0:
                embeddings = [item_id_embeddings]
                if self.encoder_name is not None:
                    # todo 更有效的建模这些特征，形成层次模型 cnn/attention/gru/transformer，这些特征一起表达一个完整语义
                    attribute_embeddings = []
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        attribute_embeddings.append(feature_embeddings)
                    attribute_embeddings = torch.cat([e.unsqueeze(dim=-2) for e in attribute_embeddings], dim=-2)
                    attribute_representation = self.attribute_encoder(attribute_embeddings)
                    embeddings.append(attribute_representation)
                    embeddings = torch.cat(embeddings, dim=-1)
                else:
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        embeddings.append(feature_embeddings)
                    embeddings = torch.cat(embeddings, dim=-1)
            else:
                embeddings = item_id_embeddings

            item_seq_embs = F.dropout(embeddings, p=0.5, training=self.training)

            output, state = gru_forward(self.enc, item_seq_embs, lengths, batch_first=True)
            state = F.dropout(state, p=0.5, training=self.training)
            output = F.dropout(output, p=0.5, training=self.training)

            explore_feature, _, _ = self.explore_attn(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                      output,
                                                      mask=mask.unsqueeze(1))

            if self.dot_product:
                p_explore_temp_not_masked = torch.matmul(explore_feature.squeeze(1), torch.transpose(self.item_emb.weight, 0, 1))
            else:
                p_explore_temp_not_masked = self.explore(explore_feature.squeeze(1))

            p_explore = F.softmax(p_explore_temp_not_masked, dim=-1)

            return p_explore, None
        elif self.model_version == 'v6':
            # 预测用户将会与历史中的哪个交互或者都不交互
            batch_size = data['item_seq'].size(0)
            mask = data['item_seq'].ne(0)
            lengths = mask.float().sum(dim=-1).long()

            item_id_embeddings = self.item_emb(data['item_seq'])
            if len(self.features) != 0:
                embeddings = [item_id_embeddings]
                if self.encoder_name is not None:
                    # todo 更有效的建模这些特征，形成层次模型 cnn/attention/gru/transformer，这些特征一起表达一个完整语义
                    attribute_embeddings = []
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        attribute_embeddings.append(feature_embeddings)
                    attribute_embeddings = torch.cat([e.unsqueeze(dim=-2) for e in attribute_embeddings], dim=-2)
                    attribute_representation = self.attribute_encoder(attribute_embeddings)
                    embeddings.append(attribute_representation)
                    embeddings = torch.cat(embeddings, dim=-1)
                else:
                    for feature in self.features:
                        feature_embeddings = self.attribute_embs[feature](data[feature])
                        embeddings.append(feature_embeddings)
                    embeddings = torch.cat(embeddings, dim=-1)
            else:
                embeddings = item_id_embeddings

            item_seq_embs = F.dropout(embeddings, p=0.5, training=self.training)

            output, state = gru_forward(self.enc, item_seq_embs, lengths, batch_first=True)
            state = F.dropout(state, p=0.5, training=self.training)
            output = F.dropout(output, p=0.5, training=self.training)

            # 没有移除重复推荐模块时，explore模块不需要推荐历史消费item
            explore_mask = torch.bmm((data['item_seq'] > 0).float().unsqueeze(1), data['source_map']).squeeze(1)

            # todo 叫loss，或者缩放因子，或者都加
            # 当非重复推荐时，所有值都为0
            if 'repeat_mask' in self.other_args and self.other_args['repeat_mask'] == 'item_mask':
                repeat_mask = data['item_mask'].ne(0)
            else:
                repeat_mask = mask
            p_repeat_raw, p = self.repeat_attn.score(state.reshape(batch_size, -1).unsqueeze(1), output,
                                                                 mask=repeat_mask.unsqueeze(1), activation='softmax')

            p = torch.bmm(p, data['source_map']).squeeze(1)
            p = p.masked_fill(~explore_mask.bool(), float('-1'))

            return p, None
        else:
            raise NotImplementedError(self.model_version)

    def do_train(self, data):
        scores, repeat_scores = self.model(data)
        loss = F.nll_loss((scores + 1e-8).log(), data['item_tgt'].reshape(-1), ignore_index=0)  # 0 is used as padding
        if self.model_version in ('v3', 'v4'):
            weight_loss_weight = 1.0
            if 'weight_loss_weight' in self.other_args:
                weight_loss_weight = self.other_args['weight_loss_weight']
            repeat_scores = repeat_scores.reshape((-1, 1))
            if 'repeat_mask' in self.other_args and self.other_args['repeat_mask'] == 'item_mask':
                item_mask = data['item_mask']
            else:
                item_mask = data['item_seq'].ne(0).float()
            if 'weight_loss' in self.other_args and self.other_args['weight_loss'] == 'weight_label':
                valid_label_flag = (data['weight_label'] != -100).reshape((-1, 1))
                weight_label = (data['weight_label'] * item_mask).reshape((-1, 1))
                valid_repeat_scores = repeat_scores[valid_label_flag]
                valid_weight_label = weight_label[valid_label_flag]
                weight_loss = nn.BCELoss()(valid_repeat_scores, valid_weight_label)
                loss += (weight_loss * weight_loss_weight)
            elif 'weight_loss' in self.other_args and self.other_args['weight_loss'] == 'weight_label2':
                valid_label_flag = (data['weight_label2'] != -100).reshape((-1, 1))
                weight_label2 = (data['weight_label2'] * item_mask).reshape((-1, 1))
                valid_repeat_scores = repeat_scores[valid_label_flag]
                valid_weight_label = weight_label2[valid_label_flag]
                weight_loss = nn.BCELoss()(valid_repeat_scores, valid_weight_label)
                loss += (weight_loss * weight_loss_weight)
            else:
                pass
        else:
            if self.mode_loss:
                # 存在的问题：
                # 1. 当target为0时，并不一定历史交互的item都不是想点的
                mode_loss = nn.BCELoss()(repeat_scores, data['mode_label'].reshape(-1))
                loss += mode_loss
        return loss

    def do_infer(self, data):
        scores = self.model(data, train=False)[0]
        scores, index = torch.sort(scores, dim=-1, descending=True)
        return scores, index

    def forward(self, data, method='mle_train'):
        data['source_map'] = build_map(data['item_seq'], max=self.item_vocab_size)
        if method == 'train':
            return self.do_train(data)
        elif method == 'infer':
            return self.do_infer(data)