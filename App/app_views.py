import json
from flask import Blueprint
from flask import request
from utils.result_views_util import *
from utils.common_util import base64_2_array    
from detect_apis.project01_main import shuiguofenlei
import numpy as np

main = Blueprint('main', __name__)


@main.route('/detect', methods=['GET', 'POST'])
def main_dect():
    request_data = request.get_data()

    # print("data",request_data)

    print(type(request_data))
    data = json.loads(request_data)

    # data=is_valid_json(get_require_data(),request_data)
    # print(data)

    if data:
        data = base64_2_array(data['data'])
        # print(data)
        print("*****************************************************")
        print(type(data))
        result = shuiguofenlei(data)
        response = result_data({'result_type': int(result)})
        return json.dumps(response)
    else:
        return json.dumps(wrong_request(400))
