# our_2

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        # 对于每个标签及其对应的原型列表进行迭代：
        if len(proto_list) > 1:
            # 如果原型列表中有多个原型：
            proto = 0 * proto_list[0].data
            # 初始化一个与原型相同形状的零张量
            for i in proto_list:
                # 对原型列表中的每个原型进行迭代：
                proto += i.data
                # 将每个原型的张量数据相加
            protos[label] = proto / len(proto_list)
            # 计算平均原型，即将总和除以原型数量
        else:
            # 如果原型列表中只有一个原型：
            protos[label] = proto_list[0]
            # 则直接将该原型赋值给对应的标签

    return protos
    # 返回更新后的全局原型字典


def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label



