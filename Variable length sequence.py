import torch
def packedd(x,y):   #x,y分别为经过construct_data处理之后的训练数据矩阵和对应的标签
    batch_size=x.shape[0]
    lengths=[]
    for i in range(batch_size):
        non_zero=torch.nonzero(x[i])
        lengths.append(non_zero.shape[0])
    length=sorted(lengths,reverse=True)
    data=torch.zeros(x.shape[0],x.shape[1])
    if y is None:
        label=None
        for i in range(x.shape[0]):
            elment=lengths.index(max(lengths))
            data[i]=x[elment]
            lengths[elment]=0
    else:
        label=torch.zeros(y.shape[0],y.shape[1])
        for i in range(x.shape[0]):
            elment=lengths.index(max(lengths))
            data[i]=x[elment]
            label[i]=y[elment]
            lengths[elment]=0
packed=nn.utils.rnn.pack_padded_sequence(x,input_lengths,batch_first=True)#此处的x代表已经要输入lstm的数据，如果不是词向量矩阵，还需要自己构建或者通过Embedding层。是否用batch_first根据自身情况而定，True代表数据维度为 batch_size*sequence_length*word_embedding_size,如果是False，数据维度为sequence_length*batch_size*word_embedding_size
outputs,(h_t,c_t)=torch.nn.LSTM(packed,hidden)
outputs,_=nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
