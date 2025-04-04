import requests

# 定义主机和端口
host = "http://localhost"  # 请将此替换为实际的主机地址
port = 8010  # 请将此替换为实际的端口号

# 定义请求的URL路径
url = f"{host}:{port}/generate"

# 定义请求参数
params = {
    'prompt': '''Wikipedia Title: Tufts University School of Medicine
The Tufts University School of Medicine is one of the eight schools that constitute Tufts University. The "Times Higher Education (THE)" and the "Academic Ranking of World Universities (ARWU)" consistently rank Tufts among the world's best medical research institutions for clinical medicine. Located on the university's health sciences campus in downtown Boston, Massachusetts, the medical school has clinical affiliations with thousands of doctors and researchers in the United States and around the world, as well as at its affiliated hospitals in both Massachusetts (including Tufts Medical Center, St. Elizabeth's Medical Center, Lahey Hospital and Medical Center and Baystate Medical Center), and Maine (Maine Medical Center). According to Thomson Reuters' "Science Watch", Tufts University School of Medicine's research impact rates sixth among U.S medical schools for its overall medical research and within the top 5 for specialized research areas such as chronic obstructive pulmonary disorder, urology, cholera, public health & health care science, and pediatrics. In addition, Tufts University School of Medicine is ranked 44th in research and 38th in primary care according to "U.S. News & World Report".

Wikipedia Title: Lake Wales Medical Center
Lake Wales Medical Center is a hospital in Lake Wales, Florida. It is owned by health care provider Community Health Systems. Lake Wales Medical Center's main building is a general use hospital that includes an emergency department, an intensive care unit and various outpatient services. Nearby, at 1120 Carlton Avenue, Suite 1300, is the hospital Neurodiagnostic and Sleep Center.

Wikipedia Title: Lake Wales, Florida
Lake Wales is a city in Polk County, Florida, United States. The population was 14,225 at the 2010 census. As of 2014, the population estimated by the U.S. Census Bureau is 15,140. It is part of the Lakeland–Winter Haven Metropolitan Statistical Area. Lake Wales is located in central Florida, west of Lake Kissimmee and east of Tampa.

Q: Answer the following question.
What was the 2014 population of the city where Lake Wales Medical Center is located?
A: 15,140

... (省略中间部分以简化示例) ...

Wikipedia Title: Beaverdam Run
Beaverdam Run\' is a short creek draining the east slopes of the Mahoning Hills, and a right bank tributary of the Lehigh River. The creek\'s banks are one of the two most likely valleys that pack animals traversed to reach boats on the river so the Anthracite from the earliest coal mining activity in Carbon County, Pennsylvania was transshipped onto boats on the river.

Q: Answer the following question.
What is the length of the river into which Pack Creek runs after it goes through the Spanish Valley?
A:'''
}

# 其他请求参数
request_params = {
    'max_input': None,
    'max_length': 200,
    'min_length': 1,
    'do_sample': False,
    'temperature': 1.0,
    'top_k': 50,
    'top_p': 1.0,
    'num_return_sequences': 1,
    'repetition_penalty': None,
    'length_penalty': None,
    'keep_prompt': False
}

try:
    # 发送GET请求，合并params和request_params作为查询参数
    response = requests.get(url, params={**params, **request_params})

    # 检查响应状态码
    if response.status_code == 200:
        # 打印响应内容
        print("响应内容如下：")
        print(response.text)
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print(f"响应内容：{response.text}")

except requests.exceptions.RequestException as e:
    # 处理请求异常
    print(f"请求过程中发生错误：{e}")
