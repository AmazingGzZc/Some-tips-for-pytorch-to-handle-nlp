# Some-tips-for-pytorch-to-handle-nlp
# word2vec训练出的词向量与pytorch中的Embedding层的关系：
  word2vec和Embedding层都会将词语转变为稠密的词向量，word2vec中有skip-gram和CBow两种模式，word2vec本质上还是单层的神经网络，通过监督训练来训练神经网络权重，而神经网络的权重就是我们所得的词向量。<br>
  pytorch中的Embedding层我认为也是一层全连接层，但是全连接层的权重是随机初始化的，Embedding层的权重也就是我们的词向量。pytorch中的Embedding层的输入是词语在词典中的位置，所以需要我们自己构建词典。<br>
  我们也可以将word2vec预训练好的词向量赋予给Embedding层，同时可以通过是否固定Embedding层权重来决定是否对预训练好的词向量进行微调。
# pytorch如何处理可变长度文本：
  pytorch可以通过使用RNN等网络结构来进行可变长度文本的训练。需要调用pack_padded_sequence和pad_packed_sequence两个方法，前者是将数据打包，需要我们将文本数据按长度由长到短进行排序，同时获取每一条文本的长度，将二者输入给pack_padded_sequence。然后将pack_padded_sequence的结果输入到RNN中，然后将RNN的输出输入到pad_packed_sequence中。
# pytorch构建自己的数据集：
  我们可以通过构建Dataset的子类来构建自己的数据集，需要重写_getitem_和_len_方法，在init中定义自己的数据路径，具体可以参考construct_data.py。