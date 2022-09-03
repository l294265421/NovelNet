from collections import defaultdict
import json
import copy

from torch.utils.data import Dataset
import csv
import codecs
import torch
import numpy as np

category_id_mapping = {"category": {"\u4ed9\u4fa0": 1, "\u60ac\u7591": 2, "\u5c0f\u8bf4": 3, "\u9752\u6625": 4, "\u53e4\u8a00": 5, "\u5386\u53f2": 6, "\u73b0\u8a00": 7, "\u5e7b\u60c5": 8, "\u7384\u5e7b": 9, "\u8f7b\u5c0f\u8bf4": 10, "\u519b\u4e8b": 11, "\u90fd\u5e02": 12, "\u79d1\u5e7b": 13, "\u6b66\u4fa0": 14, "null": 15, "\u6e38\u620f": 16, "\u5947\u5e7b": 17, "\u4f53\u80b2": 18, "\u53e4\u7c4d": 19, "\u73b0\u5b9e": 20, "\u77ed\u7bc7": 21, "\u7eaf\u7231": 22, "\u52b1\u5fd7": 23, "": 24, "\u7ae5\u4e66": 25, "\u5fc3\u7406": 26, "\u79d1\u666e": 27, "\u6559\u6750\u6559\u8f85": 28, "\u6587\u5316": 29, "\u6295\u8d44": 30, "\u793e\u79d1": 31, "\u4e24\u6027": 32, "\u4e8c\u6b21\u5143": 33, "\u516c\u7248": 34, "\u5b97\u6559": 35, "\u54f2\u5b66": 36, "\u4f11\u95f2": 37, "\u6cd5\u5f8b": 38, "\u653f\u6cbb": 39, "\u7ba1\u7406": 40, "\u6587\u5b66": 41, "\u4f20\u8bb0": 42, "\u7ecf\u6d4e": 43, "\u517b\u751f": 44}, "subcategory": {"\u53e4\u5178\u4ed9\u4fa0": 1, "\u5947\u5999\u4e16\u754c": 2, "\u4fee\u771f\u6587\u660e": 3, "\u60c5\u611f": 4, "\u7231\u60c5/\u60c5\u611f": 5, "\u5bab\u95f1\u5b85\u6597": 6, "\u67b6\u7a7a\u5386\u53f2": 7, "\u6b66\u4fa0": 8, "\u8c6a\u95e8\u4e16\u5bb6": 9, "\u9752\u6625\u7eaf\u7231": 10, "\u5f02\u65cf\u604b\u60c5": 11, "\u7a7f\u8d8a\u5947\u60c5": 12, "\u4e1c\u65b9\u7384\u5e7b": 13, "\u5e7b\u60f3\u4fee\u4ed9": 14, "\u795e\u8bdd\u4fee\u771f": 15, "\u884d\u751f\u540c\u4eba": 16, "\u519b\u65c5\u751f\u6daf": 17, "\u90fd\u5e02\u751f\u6d3b": 18, "\u79e6\u6c49\u4e09\u56fd": 19, "\u8be1\u79d8\u60ac\u7591": 20, "\u5f02\u672f\u8d85\u80fd": 21, "\u65f6\u7a7a\u7a7f\u68ad": 22, "\u4f20\u7edf\u6b66\u4fa0": 23, "\u738b\u671d\u4e89\u9738": 24, "\u4fa6\u63a2\u63a8\u7406": 25, "\u4e0a\u53e4\u5148\u79e6": 26, "\u5f02\u4e16\u5927\u9646": 27, "\u4e24\u664b\u968b\u5510": 28, "null": 29, "\u53e4\u5178\u67b6\u7a7a": 30, "\u7535\u5b50\u7ade\u6280": 31, "\u53f2\u8bd7\u5947\u5e7b": 32, "\u6e38\u620f\u5f02\u754c": 33, "\u5a31\u4e50\u660e\u661f": 34, "\u7bee\u7403\u8fd0\u52a8": 35, "\u53e4\u4eca\u4f20\u5947": 36, "\u53e6\u7c7b\u5e7b\u60f3": 37, "\u53e4\u4ee3\u60c5\u7f18": 38, "\u4e24\u5b8b\u5143\u660e": 39, "\u865a\u62df\u7f51\u6e38": 40, "\u5b50\u90e8": 41, "\u5a5a\u604b\u60c5\u7f18": 42, "\u6218\u4e89\u5e7b\u60f3": 43, "\u53e4\u4ee3\u8a00\u60c5": 44, "\u9752\u6625\u6821\u56ed": 45, "\u604b\u7231\u65e5\u5e38": 46, "\u73b0\u5b9e\u767e\u6001": 47, "\u6e38\u620f\u7cfb\u7edf": 48, "\u90fd\u5e02\u5f02\u80fd": 49, "\u6297\u6218\u70fd\u706b": 50, "\u539f\u751f\u5e7b\u60f3": 51, "\u672b\u4e16\u5371\u673a": 52, "\u8fdb\u5316\u53d8\u5f02": 53, "\u77ed\u7bc7\u5c0f\u8bf4": 54, "\u5e7d\u51a5\u60c5\u7f18": 55, "\u73b0\u4ee3\u4fee\u771f": 56, "\u8db3\u7403\u8fd0\u52a8": 57, "\u7ecf\u5546\u79cd\u7530": 58, "\u5973\u5c0a\u738b\u671d": 59, "\u4ed9\u4fa3\u5947\u7f18": 60, "\u6050\u6016\u60ca\u609a": 61, "\u6b66\u4fa0\u5e7b\u60f3": 62, "\u5386\u53f2\u4f20\u8bb0": 63, "\u4fa6\u63a2/\u60ac\u7591/\u63a8\u7406": 64, "\u53e4\u7eaf": 65, "\u667a\u5546/\u667a\u8c0b": 66, "\u4e0a\u53e4\u86ee\u8352": 67, "\u73b0\u4ee3\u9b54\u6cd5": 68, "\u6821\u56ed": 69, "\u641e\u7b11\u5410\u69fd": 70, "\u63a2\u9669\u751f\u5b58": 71, "\u8c0d\u6218\u7279\u5de5": 72, "\u5546\u6218\u804c\u573a": 73, "\u9ad8\u6b66\u4e16\u754c": 74, "\u795e\u79d8\u6587\u5316": 75, "\u672a\u6765\u4e16\u754c": 76, "\u793e\u4f1a\u4e61\u571f": 77, "\u5f02\u80fd\u8d85\u672f": 78, "\u6b66\u4fa0\u540c\u4eba": 79, "\u661f\u9645\u604b\u6b4c": 80, "\u53db\u9006\u6210\u957f": 81, "\u540c\u4eba\u884d\u751f": 82, "\u63a8\u7406\u4fa6\u63a2": 83, "\u53e4\u6b66\u673a\u7532": 84, "\u8d85\u7ea7\u79d1\u6280": 85, "\u661f\u9645\u6587\u660e": 86, "\u5251\u4e0e\u9b54\u6cd5": 87, "\u4f53\u80b2\u8d5b\u4e8b": 88, "\u6e38\u620f\u4e3b\u64ad": 89, "": 90, "\u5e7b\u60f3\u7eaf\u7231": 91, "\u5916\u56fd\u513f\u7ae5\u6587\u5b66": 92, "\u519b\u4e8b\u6218\u4e89": 93, "\u6210\u529f\u52b1\u5fd7": 94, "\u7384\u5e7b/\u65b0\u6b66\u4fa0/\u9b54\u5e7b/\u79d1\u5e7b": 95, "\u9ed1\u6697\u5e7b\u60f3": 96, "\u6e05\u7a7f\u6c11\u56fd": 97, "\u5e94\u7528\u5fc3\u7406\u5b66": 98, "\u4e2d\u56fd\u53f2": 99, "\u767e\u79d1\u77e5\u8bc6": 100, "\u9b54\u5e7b": 101, "\u793e\u4f1a": 102, "\u56db\u5927\u540d\u8457": 103, "\u5916\u56fd\u5386\u53f2": 104, "\u73b0\u7eaf": 105, "\u9b54\u6cd5\u5e7b\u60c5": 106, "\u5386\u53f2\u795e\u8bdd": 107, "\u60ca\u609a/\u6050\u6016": 108, "\u6e05\u53f2\u6c11\u56fd": 109, "\u552f\u7f8e\u5e7b\u60f3": 110, "\u7f51\u6e38\u60c5\u7f18": 111, "\u4e2d\u5c0f\u5b66\u6559\u8f85": 112, "\u6c11\u65cf\u6587\u5316": 113, "\u5f71\u89c6\u5c0f\u8bf4": 114, "\u5904\u4e16\u5b66": 115, "\u60ac\u7591\u63a2\u9669": 116, "\u56fd\u672f\u65e0\u53cc": 117, "\u8bc1\u5238/\u80a1\u7968": 118, "\u9752\u5c11\u5e74\u52b1\u5fd7": 119, "\u79d1\u666e/\u767e\u79d1": 120, "\u751f\u7269\u4e16\u754c": 121, "\u793e\u4f1a\u5b66": 122, "\u8fdc\u53e4\u795e\u8bdd": 123, "\u6027\u77e5\u8bc6": 124, "\u5e7b\u7231": 125, "\u6c11\u56fd\u60c5\u7f18": 126, "\u60ac\u7591/\u60ca\u609a": 127, "\u7ecf\u53f2\u5b50\u96c6": 128, "\u5916\u56fd\u5c0f\u8bf4": 129, "\u56fd\u5b66\u666e\u53ca\u8bfb\u7269": 130, "\u5f71\u89c6\u884d\u751f": 131, "\u53e3\u624d/\u6f14\u8bb2/\u8fa9\u8bba": 132, "\u4f5b\u6559": 133, "\u5fc3\u7075\u4e0e\u4fee\u517b": 134, "\u9038\u95fb\u91ce\u53f2": 135, "\u9752\u6625\u75bc\u75db": 136, "\u4e2d\u56fd\u54f2\u5b66": 137, "\u4e16\u754c\u540d\u8457": 138, "\u54f2\u5b66\u77e5\u8bc6\u8bfb\u7269": 139, "\u90fd\u5e02": 140, "\u8d22\u5546/\u8d22\u5bcc\u667a\u6167": 141, "\u4f5c\u54c1\u96c6": 142, "\u4eba\u9645\u4ea4\u5f80": 143, "\u4eba\u751f\u54f2\u5b66": 144, "\u52b1\u5fd7\u7ecf\u5178\u8457\u4f5c": 145, "\u6210\u529f/\u6fc0\u52b1": 146, "\u79d1\u5e7b": 147, "\u5e7d\u9ed8\u7b11\u8bdd": 148, "\u6c11\u6cd5": 149, "\u7231\u60c5\u5a5a\u59fb": 150, "\u53e4\u6b66\u672a\u6765": 151, "\u52a8\u6f2b\u884d\u751f": 152, "\u653f\u6cbb\u7406\u8bba": 153, "\u6027\u683c\u4e0e\u4e60\u60ef": 154, "\u8fdc\u53e4\u6d2a\u8352": 155, "\u5e02\u573a/\u8425\u9500": 156, "\u7206\u7b11/\u65e0\u5398\u5934": 157, "\u4e2d\u56fd\u53e4\u5178\u5c0f\u8bf4": 158, "\u7406\u8d22\u6280\u5de7": 159, "\u7eaa\u5b9e\u6587\u5b66": 160, "\u4e2d\u56fd\u73b0\u5f53\u4ee3\u968f\u7b14": 161, "\u4e2d\u56fd\u513f\u7ae5\u6587\u5b66": 162, "\u5386\u4ee3\u5e1d\u738b": 163, "\u5973\u6027\u52b1\u5fd7": 164, "\u8ba4\u77e5\u5fc3\u7406\u5b66": 165, "\u793e\u4f1a\u5fc3\u7406\u5b66": 166, "\u70ed\u8840\u6c5f\u6e56": 167, "\u5f71\u89c6\u5267\u672c": 168, "\u4e16\u754c\u53f2": 169, "\u4e94\u4ee3\u5341\u56fd": 170, "\u5a5a\u59fb": 171, "\u7075\u6c14\u590d\u82cf": 172, "\u6587\u5b66\u7406\u8bba": 173, "\u53f2\u6599\u5178\u7c4d": 174, "\u5fc3\u7406\u5b66\u7ecf\u5178\u8457\u4f5c": 175, "\u5386\u53f2\u666e\u53ca\u8bfb\u7269": 176, "\u4e61\u571f": 177, "\u6b66\u4fa0\u60c5\u7f18": 178, "\u540d\u5bb6\u4f5c\u54c1": 179, "\u60c5\u5546/\u60c5\u7eea\u7ba1\u7406": 180, "\u7ecf\u6d4e\u901a\u4fd7\u8bfb\u7269": 181, "\u6587\u96c6": 182, "\u5065\u5eb7\u767e\u79d1": 183, "\u4e2d\u56fd\u53e4\u8bd7\u8bcd": 184, "\u5386\u53f2": 185, "\u5fc3\u7406\u767e\u79d1": 186, "\u519b\u4e8b": 187, "\u5916\u56fd\u968f\u7b14": 188, "\u5176\u4ed6\u884d\u751f": 189, "\u9a6c\u514b\u601d\u4e3b\u4e49\u7406\u8bba": 190, "\u4eba\u7269\u4f20\u8bb0": 191, "\u8d22\u7ecf\u4eba\u7269": 192, "\u5b98\u573a": 193, "\u897f\u65b9\u5947\u5e7b": 194, "\u519b\u4e8b\u53f2": 195, "\u4e2d\u56fd\u6587\u5316": 196, "\u5386\u53f2\u968f\u7b14": 197, "\u519b\u4e8b\u4eba\u7269": 198, "\u79d1\u5b66\u4e16\u754c": 199}}


class NovelNetDataset(Dataset):
    def __init__(self, sample_file, max_session_len, data_type, debug=False, config={}):
        """
        data_type
        {
            'feature': 1,
            'training': 2,
            'dev': 3,
            'test': 4,
            'None': 5
        }
        @param sample_file:
        @param max_session_len:
        @param data_type:
        @param debug:
        """
        super(NovelNetDataset, self).__init__()

        self.sample_file = sample_file
        self.max_session_len = max_session_len
        self.data_type = data_type
        self.debug = debug
        self.config = config

        self.item_atts = dict()
        self.samples = []
        self.max_feature_index = defaultdict(int)
        self.columns = None
        self.load_from_dataframe()

    def pad_item_seq(self, features, columns):
        """

        @param features:
        @param columns:
        @return:
        """
        max_session_len = self.max_session_len
        if 'new_as_one' in self.config and self.config['new_as_one']:
            max_session_len += 1
        result = []
        for i, feature in enumerate(features):
            if columns[i] not in ['session_id', 'target_item_id', 'mode_label']:
                padding_index = 0
                if 'label' in columns[i]:
                    padding_index = -100
                padded_feature = feature + [padding_index] * (max_session_len - len(feature))
                result.append(padded_feature)
            else:
                result.append(feature)
        return result

    def update_max_feature_index(self, features, feature_name):
        """

        @param features:
        @param feature_name:
        @return:
        """
        max_feature_index_candidate = max(features)
        if max_feature_index_candidate > self.max_feature_index[feature_name]:
            self.max_feature_index[feature_name] = max_feature_index_candidate

    def extract_features(self, item_seq, target_item, leftover_items):
        """
        1. item mask 历史交互item，同一个item，只有最后一个需要参与是否重复推荐
        @param item_seq:
        @param target_item:
        @return:
        """
        target_item = copy.deepcopy(target_item)
        if 'new_as_one' in self.config and self.config['new_as_one']:
            # max(self.config['item_popularity'])
            new_novel = copy.deepcopy(item_seq[-1])
            new_novel_id = 50833
            new_novel.ItemId = new_novel_id

            new_novel.Attributes = json.loads(new_novel.Attributes)

            new_novel.Attributes['book_id'] = ''
            new_novel.Attributes['bookname'] = ''
            new_novel.Attributes['category'] = ''
            new_novel.Attributes['subtype'] = ''
            new_novel.Attributes['author'] = ''
            new_novel.Attributes['expose'] = -1
            new_novel.Attributes['click'] = -1
            new_novel.Attributes['intro'] = -1
            new_novel.Attributes['read'] = -1
            new_novel.Attributes['real_read'] = -1
            new_novel.Attributes['collect'] = -1
            new_novel.Attributes['read_duration'] = -1
            new_novel.Attributes = json.dumps(new_novel.Attributes)

            item_seq.append(new_novel)

            if target_item['ItemId'] not in set([item['ItemId'] for item in item_seq]):
                target_item.ItemId = new_novel_id

        session_id = [item_seq[0]['SessionId']]
        item_ids = [item['ItemId'] for item in item_seq]
        target_item_id = [target_item['ItemId']]
        leftover_item_ids = [e['ItemId'] for e in leftover_items]

        # item type, 1 old novels, 2 new novels
        item_type = [1 if item_id != 50833 else 2 for item_id in item_ids]

        # 2015-aaai-Will you “reconsume” the near past? Fast prediction on short-term reconsumption behaviors.pdf
        item_popularity = [self.config['item_popularity'][item_id] if item_id in self.config['item_popularity'] else 0
                           for item_id in item_ids]
        item_count_dict = defaultdict(int)
        item_count = []  # Window Repeat Ratio
        for item_id in item_ids:
            item_count_dict[item_id] += 1
            item_count.append(item_count_dict[item_id])

        # gaps 2016-WWW-Modeling User Consumption Sequences
        item_times = [item['Time'] for item in item_seq]
        temporal_gaps = []
        item_num_gaps = []
        last_positions = {}
        for i, item_id in enumerate(item_ids):
            item_time = item_times[i]
            if item_id not in last_positions:
                temporal_gaps.append(0)
                item_num_gaps.append(0)
                last_positions[item_id] = [i, item_time]
            else:
                start, last_item_time = last_positions[item_id]
                distinct_items = set(item_ids[start + 1: i])
                temporal_gap = min(int((item_time - last_item_time) / 3600) + 1, 8 * 24 + 1)
                item_num_gap = len(distinct_items)
                temporal_gaps.append(temporal_gap)
                item_num_gaps.append(item_num_gap)
                last_positions[item_id] = [i, item_time]


        item_attributes = [json.loads(item['Attributes']) for item in item_seq]
        categories = [attributes['category'] for attributes in item_attributes]
        subcategories = [attributes['subtype'] for attributes in item_attributes]
        categories_id = [category_id_mapping['category'][category] if category in category_id_mapping['category'] else 0 for category in categories]
        subcategories_id = [category_id_mapping['subcategory'][category] if category in category_id_mapping['subcategory'] else 0 for category in subcategories]

        if 'new_as_one' in self.config and self.config['new_as_one']:
            mode_label = [1.0] if target_item_id[0] in item_ids[: -1] else [0.0]
        else:
            mode_label = [1.0] if target_item_id[0] in item_ids else [0.0]
        weight_label = [1.0 if item_ids[i] == target_item_id[0] else 0.0 for i in range(len(item_ids))]

        weight_label2 = [1.0 if item_ids[i] in (target_item_id + leftover_item_ids) else 0.0
                         for i in range(len(item_ids))]

        # 每个item在历史中最后一次出现的位置的mask为1，其它位置为0
        item_mask = []
        for i in range(len(item_ids) - 1, -1, -1):
            item_id = item_ids[i]
            if item_id not in item_ids[i + 1:]:
                item_mask.insert(0, 1.0)
            else:
                item_mask.insert(0, 0.0)

        candidate_mask = [0] * len(item_mask)
        scale_last_n = 2
        candidates = []
        if 'scale_last_n' in self.config:
            scale_last_n = self.config['scale_last_n']
        for i in range(len(item_mask) - 1, -1, -1):
            if item_mask[i] == 1.0:
                candidate_mask[i] = 1.0
                candidates.append(item_ids[i])
            if len(candidates) >= scale_last_n:
                break
        candidate_mask2 = [1.0 if item_ids[i] in candidates else 0.0 for i in range(len(item_mask))]

        position_ids = [i for i in range(len(item_ids), 0, -1)]

        predict_time = target_item['Time']
        # 0 padding
        item_time_diff = [min(int((predict_time - item_time) / 3600) + 1, 8 * 24 + 1) for item_time in item_times]

        item_attributes = [json.loads(item['Attributes']) for item in item_seq]
        # 0 padding 1 no 2 yes
        item_clicks = [item['click'] + 1 for item in item_attributes]
        item_intro = [item['intro'] + 1 for item in item_attributes]
        item_read = [item['read'] + 1 for item in item_attributes]
        item_real_read = [item['real_read'] + 1 for item in item_attributes]
        item_collect = [item['collect'] + 1 for item in item_attributes]
        # 按分钟算，大于60分钟算为60分钟
        item_read_duration = [int((3600 if item['read_duration'] > 3600 else item['read_duration']) / 60) + 1
                              if item['read_duration'] != -1 else 0  # new_as_one
                              for item in item_attributes]

        result = [session_id, item_ids, target_item_id, item_time_diff,
                  item_clicks, item_intro, item_read, item_real_read,
                  item_collect, item_read_duration, item_mask, mode_label, weight_label, weight_label2,
                  candidate_mask, candidate_mask2, position_ids, categories_id, subcategories_id,
                  item_popularity, item_count, item_type, temporal_gaps, item_num_gaps]
        columns = ['session_id', 'item_ids', 'target_item_id', 'item_time_diff',
                   'item_clicks', 'item_intro', 'item_read', 'item_real_read',
                   'item_collect', 'item_read_duration', 'item_mask', 'mode_label', 'weight_label', 'weight_label2',
                   'candidate_mask', 'candidate_mask2', 'position_ids', 'categories_id', 'subcategories_id',
                   'item_popularity', 'item_count', 'item_type', 'temporal_gaps', 'item_num_gaps']
        if self.columns is None:
            self.columns = columns
        for i in range(len(result)):
            if columns[i] in ['session_id', 'target_item_id', 'mode_label']:
                continue
            self.update_max_feature_index(result[i], columns[i])
        return result, columns

    def generate_sample(self, features):
        """

        @param features:
        @return:
        """
        result = [torch.tensor(feature) for feature in features]
        return result

    def history_coverage(self, history_items: list, predicted_items: list):
        """

        :param history_items:
        :param predicted_items:
        :return:
        """
        valid_history_items = set(history_items)
        history_items_in_predicted = [1 if e in predicted_items else 0 for e in valid_history_items]
        result = sum(history_items_in_predicted) / len(valid_history_items)
        return result

    def load_from_dataframe(self):
        sessions = defaultdict(list)
        target_row = 0
        unique_items = set()
        for row in self.sample_file.iterrows():
            sessions[row[1]['SessionId']].append(row[1])
            if row[1]['DataType'] == self.data_type:
                target_row += 1
            if row[1]['DataType'] == self.data_type or (self.data_type == 2 and row[1]['DataType'] == 1):
                unique_items.add(row[1]['ItemId'])
            if self.debug and target_row >= 2048:
                break
        sessions = [sorted(e, key=lambda x: x['Time']) for e in sessions.values()]

        target_sessions = []
        for session in sessions:
            keep = False
            for e in session:
                if e['DataType'] == self.data_type:
                    keep = True
                    break
            if keep:
                target_sessions.append(session)
            else:
                print('filtered')

        size_of_sessions = [len(e) if len(e) < self.max_session_len else self.max_session_len for e in target_sessions]
        print('user num: %d' % len(sessions))
        print('avg size of sessions: %f' % np.mean(size_of_sessions))
        print('median size of sessions: %f' % np.median(size_of_sessions))
        print('item num: %d' % len(unique_items))
        print('interaction num: %d' % target_row)

        # 统计不在出现的item的比例
        total_interaction = 0
        total_coverage = 0
        for session in sessions:
            session = session[: self.max_session_len]
            item_ids = [e['ItemId'] for e in session]
            for index in range(1, len(session)):
                total_interaction += 1
                coverage = self.history_coverage(item_ids[: index], item_ids[index:])
                total_coverage += coverage
        print('ground_truth_coverage: %f' % (total_coverage / total_interaction))

        for session in sessions:
            session = session[: self.max_session_len]
            for feature_end_index in range(1, len(session)):
                item_seq = session[: feature_end_index]
                target_item = session[feature_end_index]
                if target_item['DataType'] != self.data_type:
                    continue
                features, columns = self.extract_features(item_seq, target_item,
                                                          leftover_items=session[feature_end_index + 1:])

                if 'evaluation_mode' in self.config:
                    model_label_index = columns.index('mode_label')
                    model_label = features[model_label_index][0]
                    if self.config['evaluation_mode'] == 'repeat':
                        if model_label == 0:
                            continue
                    elif self.config['evaluation_mode'] == 'new':
                        if model_label == 1:
                            continue
                    else:
                        raise NotImplementedError(self.config['evaluation_mode'])

                features = self.pad_item_seq(features, columns)
                sample = self.generate_sample(features)
                self.samples.append(sample)
        self.len = len(self.samples)
        print('data size: ', self.len)

    def load(self):
        clean = lambda l: [int(x) for x in l.strip('[]').split(',')]

        id=0
        with codecs.open(self.sample_file, encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='|')
            for row in csv_reader:
                id+=1
                self.samples.append([torch.tensor([id]), torch.tensor(clean(row[0])), torch.tensor(clean(row[1]))])

        self.len=len(self.samples)
        print('data size: ', self.len)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    id, item_seq, item_tgt, item_time_diff, item_clicks, item_intro, item_read, item_real_read, item_collect, \
    item_read_duration, item_mask, mode_label, weight_label, weight_label2, candidate_mask, candidate_mask2,\
    position_ids, categories_id, subcategories_id, item_popularity, item_count, item_type, temporal_gaps,\
        item_num_gaps = zip(*data)

    return {
        'id': torch.cat(id),
        'item_seq': torch.stack(item_seq),
        'item_tgt': torch.stack(item_tgt),
        'item_time_diff': torch.stack(item_time_diff),
        'item_clicks': torch.stack(item_clicks),
        'item_intro': torch.stack(item_intro),
        'item_read': torch.stack(item_read),
        'item_real_read': torch.stack(item_real_read),
        'item_collect': torch.stack(item_collect),
        'item_read_duration': torch.stack(item_read_duration),
        'item_mask': torch.stack(item_mask),
        'mode_label': torch.stack(mode_label),
        'weight_label': torch.stack(weight_label),
        'weight_label2': torch.stack(weight_label2),
        'candidate_mask': torch.stack(candidate_mask),
        'candidate_mask2': torch.stack(candidate_mask2),
        'position_ids': torch.stack(position_ids),
        'categories_id': torch.stack(categories_id),
        'subcategories_id': torch.stack(subcategories_id),
        'item_popularity': torch.stack(item_popularity),
        'item_count': torch.stack(item_count),
        'item_type': torch.stack(item_type),
        'temporal_gaps': torch.stack(temporal_gaps),
        'item_num_gaps': torch.stack(item_num_gaps)
    }