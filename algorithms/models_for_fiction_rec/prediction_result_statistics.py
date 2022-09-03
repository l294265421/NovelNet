from common import common_path

if __name__ == '__main__':
    model_serial_number = '94'
    best_epoch = '8'
    filepath = common_path.project_dir + '/rar_base_dir/%ss/%s.test' % (model_serial_number, best_epoch)
    instances = []
    with open(filepath) as input_file:
        for line in input_file:
            # 预测结果|ground_truth|model_label|history
            parts = [eval(e) for e in line.split('|')]
            instances.append(parts)
    instance_num = len(instances)
    repeat_prediction_num = 0
    for instance in instances:
        top_one = instance[0][0]
        if top_one in instance[-1]:
            repeat_prediction_num += 1
    print('model_serial_number\tbest_epoch\tinstance_num\trepeat_prediction_num\trepeat_ratio')
    print('%s\t%s\t%d\t%d\t%f' % (model_serial_number, best_epoch, instance_num, repeat_prediction_num,
                                 repeat_prediction_num / instance_num))
