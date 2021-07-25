import json

RESULTS = {
    200: {'code': 200, 'msg': 'success'},
    400: {'code': 400, 'msg': 'require paremeter'}
}
"""
    describe:指定请求数据的必须个视
    args:无
    return：指定数据形式（dict类型）
"""


def get_require_data():
    request_type = {
        'type': False,
        'data': False
    }
    return request_type


"""
    describe:指定请求数据的必须个视
    args:无
    return：指定数据形式（dict类型）
"""


def get_result_model():
    return{
        'code': '',
        'msg': '',
        'data': {},
        'cost_time': 0
    }


def is_valid_json(require_data, data):
    try:
        data = json.loads(data)
        # print(type(data))
        tag = True
        for key in require_data:
            if key in data.keys():
                if require_data[key] != False:
                    tag = is_valid_json(require_data[key], data[key])
                else:
                    return False
                if not tag:
                    return False
        return data
    except:
        return False


def result_data(data, cost_time=0):
    result = get_result_model()
    result['msg'] = RESULTS[200]['msg']
    result['code'] = RESULTS[200]['code']
    result['data'] = data
    result['cost_time'] = cost_time
    return result


def wrong_request(code):
    return RESULTS[code]
