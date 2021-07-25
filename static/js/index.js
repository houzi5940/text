const main_config = {
    //请求主机,''默认为本机
    request_host: '',
    api_url: {
        //api配置
        garbage_detect: { //需要更改此处，且后面也要跟着替换
            method: 'post',
            url: '/app/detect',
            result_transfer: {
                '0': '香蕉',
                '1': '杨桃',
                '2': '樱桃',
                '3': '榴莲',
                '4': '山竹',
                '5': '柠檬',
                '6': '龙眼',
                '7': '菠萝',
                '8': '红毛丹',
                '9': '草莓',
            }
        }
    },
}

function package_data(img_data, file) {
    var base64_data = img_data.substring(img_data.indexOf(',') + 1)
    var file_type = file.type.substring(file.type.indexOf('/') + 1)
    return {
        'type': file_type,
        'data': base64_data
    }
}

function createXHR() {
    var req = null;
    if (window.XMLHttpRequest) {
        req = new XMLHttpRequest();
    } else {
        req = new ActiveXObject("Microsoft.XMLHTTP");
    }
    return req;
}

function detect_img(img, file, requ_name) {
    document.getElementById(requ_name).innerText = 0
        // document.getElementById('result').innerHTML = json_view({});
    var data = package_data(img.src, file)
    var xhr = createXHR()
    xhr.open(main_config.api_url.garbage_detect.method, main_config.request_host + main_config.api_url.garbage_detect.url, true)

    xhr.send(JSON.stringify(data))
    xhr.onreadystatechange = function(e) {
        var resp = JSON.parse(xhr.response)
        console.log(resp)
        document.getElementById(requ_name).innerText = main_config.api_url.garbage_detect.result_transfer[resp.data.result_type]
    }
    // document.getElementById('img-scanning').setAttribute('style', 'display:none;')
}





document.getElementById("listInput").addEventListener("change", function() {
    var files = this.files;
    var oUL = document.getElementById("divUL");
    oUL.innerHTML = "";

    for (let i = 0; i < files.length; i++) {
        var get_file = files[i]
            // alert(get_file.height)
        oUL.innerHTML += '<div  class = "img_text"><div class="img_div"><img style=" " id="img' + i + '" class="img_hw" /></div><div><p id="detect_result' + i + '">123</p></div></div>'
        const reader = new FileReader();
        reader.readAsDataURL(get_file)
        reader.onload = function() {
            let name = 'img' + i
            let requ_name = "detect_result" + i
            document.getElementById(name).src = reader.result;
            const img = new Image()
            img.src = this.result
            // document.getElementById(name).width = img.width / 4;
            detect_img(img, get_file, requ_name)
                // alert(img.width)
                // var imgList = document.getElementById("img" + i);
                // imgList.style.height = this.height / 4 + 'px';
                // imgList
        }

    }
});

// style = "float:left height: 200px width:100px border:5px solid red"