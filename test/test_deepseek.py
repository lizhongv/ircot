# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import json
import os
import dashscope  # pip install dashscope
os.environ['DASHSCOPE_API_KEY'] = "sk-83cd50057e5c4c82aae71b32168a1ea1"


context1 = """
Transformer 模型是自然语言处理领域的一项重大突破，它引入了一种全新的架构，摒弃了传统的循环神经网络（RNN）和卷积神经网络（CNN）结构，转而采用自注意力机制（Self-Attention Mechanism）来捕捉序列中的依赖关系。

Transformer 模型由编码器（Encoder）和解码器（Decoder）两部分组成。编码器负责将输入序列转换为一系列的特征表示，解码器则根据这些特征表示生成输出序列。自注意力机制允许模型在处理每个位置的输入时，能够关注到序列中的其他位置，从而更好地捕捉全局信息。

在编码器中，输入序列首先被嵌入到一个低维向量空间中，然后经过多个编码器层的处理。每个编码器层包含两个子层：多头自注意力层（Multi-Head Self-Attention Layer）和前馈神经网络层（Feed-Forward Neural Network Layer）。多头自注意力层通过多个不同的注意力头来并行地计算注意力分数，从而捕捉不同方面的依赖关系。前馈神经网络层则对多头自注意力层的输出进行非线性变换。

解码器的结构与编码器类似，但在每个解码器层中还包含一个额外的多头注意力层，用于关注编码器的输出。解码器的输出经过一个线性层和 softmax 函数，得到每个位置的概率分布，从而生成最终的输出序列。

Transformer 模型的优点包括：
1. 并行计算能力强：由于自注意力机制的使用，Transformer 模型可以并行处理输入序列，大大提高了训练和推理的速度。
2. 能够捕捉长距离依赖关系：自注意力机制允许模型在处理每个位置的输入时，能够关注到序列中的其他位置，从而更好地捕捉长距离依赖关系。
3. 可扩展性好：Transformer 模型的结构简单，易于扩展，可以通过增加编码器和解码器的层数来提高模型的性能。

Transformer 模型的应用非常广泛，包括机器翻译、文本生成、问答系统、语音识别等领域。在机器翻译中，Transformer 模型已经成为了主流的模型架构，取得了非常好的效果。

请详细解释 Transformer 模型在语音识别中的应用原理。
"""


context2 = """
Transformer 模型是自然语言处理领域的一项重大突破，它引入了一种全新的架构，摒弃了传统的循环神经网络（RNN）和卷积神经网络（CNN）结构，转而采用自注意力机制（Self-Attention Mechanism）来捕捉序列中的依赖关系。

Transformer 模型由编码器（Encoder）和解码器（Decoder）两部分组成。编码器负责将输入序列转换为一系列的特征表示，解码器则根据这些特征表示生成输出序列。自注意力机制允许模型在处理每个位置的输入时，能够关注到序列中的其他位置，从而更好地捕捉全局信息。

在编码器中，输入序列首先被嵌入到一个低维向量空间中，然后经过多个编码器层的处理。每个编码器层包含两个子层：多头自注意力层（Multi-Head Self-Attention Layer）和前馈神经网络层（Feed-Forward Neural Network Layer）。多头自注意力层通过多个不同的注意力头来并行地计算注意力分数，从而捕捉不同方面的依赖关系。前馈神经网络层则对多头自注意力层的输出进行非线性变换。

解码器的结构与编码器类似，但在每个解码器层中还包含一个额外的多头注意力层，用于关注编码器的输出。解码器的输出经过一个线性层和 softmax 函数，得到每个位置的概率分布，从而生成最终的输出序列。

Transformer 模型的优点包括：
1. 并行计算能力强：由于自注意力机制的使用，Transformer 模型可以并行处理输入序列，大大提高了训练和推理的速度。
2. 能够捕捉长距离依赖关系：自注意力机制允许模型在处理每个位置的输入时，能够关注到序列中的其他位置，从而更好地捕捉长距离依赖关系。
3. 可扩展性好：Transformer 模型的结构简单，易于扩展，可以通过增加编码器和解码器的层数来提高模型的性能。

Transformer 模型的应用非常广泛，包括机器翻译、文本生成、问答系统、语音识别等领域。在机器翻译中，Transformer 模型已经成为了主流的模型架构，取得了非常好的效果。

请对比 Transformer 模型与 RNN 在文本生成任务中的性能差异。
"""


def guanwang_deepseek_v3():
    client = OpenAI(api_key="sk-db5fb24a73174298add61ada4f273ef2", base_url="https://api.deepseek.com")

    response1 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": context1}
        ],
        stream=False
    )
    print("Response 1:")
    print(response1.choices[0].message.content)
    print(response1.usage)

    response2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": context2}
        ],
        stream=False
    )
    print("Response 2:")
    print(response2.choices[0].message.content)
    print(response2.usage)


def ali_deepseek_r1_dash():

    messages = [
        {'role': 'user', 'content': '大学生应该恋爱吗'}
    ]

    response = dashscope.Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="deepseek-r1",
        messages=messages,
        result_format='message'
    )

    print("=" * 20 + "第一轮对话" + "=" * 20)
    print("=" * 20 + "思考过程" + "=" * 20)
    print(response.output.choices[0].message.reasoning_content)
    print("=" * 20 + "最终答案" + "=" * 20)
    print(response.output.choices[0].message.content)

    messages.append({'role': 'assistant', 'content': response.output.choices[0].message.content})
    messages.append({'role': 'user', 'content': '如何找到一个合适的恋人'})

    print("=" * 20 + "第二轮对话" + "=" * 20)
    response = dashscope.Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="deepseek-r1",
        messages=messages,
        result_format='message'
    )

    print("=" * 20 + "思考过程" + "=" * 20)
    print(response.output.choices[0].message.reasoning_content)
    print("=" * 20 + "最终答案" + "=" * 20)
    print(response.output.choices[0].message.content)


def ali_deepseek_v3_dash():
    response1 = dashscope.Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-turbo",  # "qwen-plus"
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': context1}
            # {'role': 'user', 'content': "你是谁？"}
        ],
        result_format='message'
    )
    print(response1)
    print(response1.output.choices[0].message.content)
    print(response1.usage)

    response2 = dashscope.Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-turbo",  # "qwen-plus"
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': context2}
        ],
        result_format='message'
    )
    print(response2)
    print(response2.output.choices[0].message.content)
    print(response2.usage)


def ali_deepseek_v3_openai():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    response1 = client.chat.completions.create(
        model="qwen-plus",  # "qwen-turbo",  # "deepseek-v3"  "qwen-plus"
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': context1}],
    )
    print(response1.model_dump_json())
    print(response1.choices[0].message.content)
    print(response1.usage)

    response2 = client.chat.completions.create(
        model="qwen-plus",  # "qwen-turbo",  # "deepseek-v3"  "qwen-plus"
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': context2}],
    )
    print(response2.model_dump_json())
    print(response2.choices[0].message.content)
    print(response2.usage)


if __name__ == "__main__":
    # guanwang_deepseek_v3()
    # https://help.aliyun.com/zh/model-studio/user-guide/context-cache?scm=20140722.S_help%40%40%E6%96%87%E6%A1%A3%40%402862577.S_RQW%40ag0%2BBB2%40ag0%2BBB1%40ag0%2Bhot%2Bos0.ID_2862577-RL_%E7%BC%93%E5%AD%98-LOC_doc%7EUND%7Eab-OR_ser-PAR1_212a5d4017423390938815173d9b80-V_4-P0_0-P1_0&spm=a2c4g.11186623.help-search.i67
    # https://help.aliyun.com/zh/model-studio/getting-started/models?spm=0.0.0.i6#9f8890ce29g5u
    # ali_deepseek_r1_dash()
    # ali_deepseek_v3_dash()
    ali_deepseek_v3_openai()
