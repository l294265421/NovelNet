import os
import sys

from common import common_path
from utils import file_utils


def get_target_parget(line, substr, offset, length=sys.maxsize):
    """

    :param line:
    :param substr:
    :param offset:
    :param length:
    :return:
    """
    result = line[line.index(substr) + offset:][: length]
    return result


def parse_line(line: str):
    """

    :param line: epoch: 4 valid (MRR 0.4991, HR 0.4991), test (MRR 0.4657, HR 0.4657)
    :return:
    """
    target_part = get_target_parget(line, 'test', 0)
    mrr = get_target_parget(target_part, 'MRR', 4, 6)
    hr = get_target_parget(target_part, 'HR', 3, 6)
    return hr, mrr


if __name__ == '__main__':
    log_path = os.path.join(common_path.repeat_aware_rec_base_dir, 'worker_0-stdout')
    lines = file_utils.read_all_lines(log_path)
    evaluation_mode_and_result_lines = {}
    evaluation_mode = None
    result_lines = []
    result_line_flag = False
    for line in lines:
        if line.startswith('evaluation_mode'):
            evaluation_mode = line[line.index(':') + 2:]
        elif line.startswith('@'):
            result_lines.append(line)
            result_line_flag = True
        else:
            if result_line_flag:
                result_lines.append(line)
                result_line_flag = False

                if len(result_lines) == 10:
                    evaluation_mode_and_result_lines[evaluation_mode] = result_lines
                    result_lines = []
    for mode, target_lines in evaluation_mode_and_result_lines.items():
        hrs = []
        mrrs = []
        top = -1
        for line in target_lines:
            if line.startswith('@'):
                top = line
                continue
            if mode == 'all' and top == '@1':
                print(line)
            hr, mrr = parse_line(line)
            hrs.append(hr)
            mrrs.append(mrr)
        output_line_elements = [mode] + hrs + mrrs
        print('\t'.join(output_line_elements))
