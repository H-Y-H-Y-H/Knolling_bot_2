import torch
from torch import nn


class CustomRNN(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, lengths):
        """
        以batch_first=True为例，
        input shape(batch_size, seq_length, input_size)
        lengths shape(batch_size)，每个句子的长度，无序

        1. 使用pack_padded_sequence进行打包
          输入：
            lengths只支持放在cpu上运行，不支持cuda
            enforce_sorted默认为True，表示传入的lengths已按从大到小排序
            由于我们传入的初始数据input没有按长度排序，所以设enforce_sorted=False
          输出：
            包

        2. 将包喂入RNN网络
          输入：
            包
          输出：
            result(包)和hn；
            result依然是个打包状态的包
            hn是最后一个单词喂入RNN网络是的输出
            当使用nn.RNN或nn.GRU时，hn是Tensor；使用nn.LSTM时，hn=(h_n, c_n)是一个元组
            注意：无论batch_first为何值，hn形状总是(num_directions * num_layers, batch_size, hidden_size)
            注意：输出hn的batch维度已经恢复为原始输入input的句子顺序

        3.使用pad_packed_sequence对上一步的result解包（即填充0）
          输入：
            total_length参数一般不需要，因为lengths列表中一般含最大值。但分布式训练时是将一个batch切分了，故一定要有！
          输出：
            output和lens
            lens和输入的lengths相等。
            注意：output的形状为(batch_size, seq_length, num_directions * hidden_size)
            注意：output的batch维度已经恢复为元时输入input的句子顺序
        """
        package = nn.utils.rnn.pack_padded_sequence(input, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False)
        result, hn = super().forward(package)
        output, lens = nn.utils.rnn.pad_packed_sequence(result, batch_first=self.batch_first, total_length=input.shape[self.batch_first])
        return output, hn


# 测试示例
if __name__ == '__main__':
    print(torch.cuda.is_available())
    # 词向量矩阵：10个单词，每个词向量5维
    E = nn.Embedding(10, 5, _weight=torch.arange(50).float().view(10, 5)).cuda()
    # 定义句子
    seqs = torch.LongTensor([
        [8, 9, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7],
        [5, 6, 7, 8, 0, 0, 0],
    ]).cuda()
    lens = torch.Tensor([2, 7, 4])  # 每个句子的真实长度
    # 定义网络
    lstm = CustomRNN(input_size=5, hidden_size=8, batch_first=True, num_layers=3, bidirectional=True).cuda()  # 改进的LSTM模型

    x = E(seqs)  # shape(3,7,5)
    out, hn = lstm(x, lens)

    print('RNN output shape:', out.shape)  # out shape(3,7,16)  # 双向LSTM
    if isinstance(hn, tuple):
        print('hn shape', hn[0].shape)
        print('cn shape', hn[1].shape)
    else:
        print('hn shape', hn.shape)
