

#official
import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer

from EarlyStop import EarlyStopping

#my libaray
from Data.load_data import *
from model import *



def train(model, train_data,train_batchsize, val_data,val_batchsize, learning_rate, epochs,early_stop_patience,model_path):

    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=train_batchsize, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=val_batchsize)
    # 判断是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    # 定义损失函数和优化器
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print_shape_flag=True
    #设置早停
    early_stopping=EarlyStopping(patience=early_stop_patience,verbose=True,path=model_path)
    # 开始进入训练循环
    for epoch_num in range(epochs):
        #####################
        # validate the model#
        #####################
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        model.train()
        for train_input, train_label in tqdm(train_dataloader):

            if print_shape_flag:
                print("Example attention_mask's shape ",train_input['attention_mask'].shape)
                print("Example input_ids's shape ", train_input['input_ids'].shape)
                print("Example train_label's shape ", train_label.shape)
                print("Example input_ids", train_input['input_ids'])
                print_shape_flag=False

            train_label = train_label.long().to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            input = {
                'attention_mask': mask,
                'input_ids': input_id
            }

            # 通过模型得到输出
            output = model(input)
            # 计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()


        #####################
        # validate the model#
        #####################
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in tqdm(val_dataloader):
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.long().to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                input = {
                    'attention_mask': mask,
                    'input_ids': input_id
                }
                output = model(input)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')

        early_stopping(total_loss_val / len(val_data),model)

        if early_stopping.early_stop:
            print("Early stopping")
            break



def test(model,tokenizer,sentence_index:int,sentence:str,device:int)->None:

    bert_input=tokenizer(sentence,
              padding='max_length',
              max_length=512,
              truncation=True,
              return_tensors="pt")
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        mask = bert_input['attention_mask'].to(device)
        input_id = bert_input['input_ids'].squeeze(1).to(device)
        input = {
            'attention_mask': mask,
            'input_ids': input_id
        }
        output = model(input)
        index=output.argmax(dim=1)
        print(str(sentence_index)+" : "+sentence+" 's emotion  :   "+EmotionLabels[index.item()])

def main():
    # 准备一些参数
    MODEL_PATH = './Config/'
    TOKENIZER_PATH='./Config/'
    EPOCHS = 0
    LR = 1e-6
    early_stop_patience = 5
    train_batch_size = 30
    val_batch_size = 10
    test_batch_size = 8
    choice =2
    model_save_path = "./pertrain_bert_10.pt"
    if choice==0:
        print("load initial pretrain model.")
        model = BertClassifier(MODEL_PATH)
        torch.save(model.state_dict(), model_save_path)
        pass
    elif choice==1:
        #准备数据
        print("Begin to load data.")
        df = load_data("Data/data.jsonl")
        df_train, df_val, df_test = split_data(df)
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
        train_data=EmotionDataset(tokenizer,df_train)
        val_data = EmotionDataset(tokenizer,df_val)
        #test_data = EmotionDataset(tokenizer, df_test)


        # 准备模型
        print("Begin to build model.")
        model = BertClassifier(MODEL_PATH)

        print("Begin to train bert.")
        train(model=model,
              train_data=train_data,
              train_batchsize=train_batch_size,
              val_data=val_data,
              val_batchsize=val_batch_size,
              learning_rate=LR,
              epochs=EPOCHS,
              early_stop_patience=early_stop_patience,
              model_path=model_save_path)
        print("Train bert ends up.")
    else:
        print("Begin to test bert on sentences classification.")
        print("The model is being test is "+model_save_path)
        model = BertClassifier(MODEL_PATH)
        model.load_state_dict(torch.load(model_save_path))
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

        test(model=model,tokenizer=tokenizer,sentence_index=0,sentence='you are a jerk',device=0)
        test(model=model,tokenizer=tokenizer,sentence_index=1,sentence='accompany you to go through a long time',device=0)
        test(model=model,tokenizer=tokenizer,sentence_index=2,sentence='everyone has days when they feel dejected or down',device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=3, sentence='son of your bitch', device=0)
        test(model=model, tokenizer=tokenizer, sentence_index=4,sentence='the water of the fountain does not die, and the fire of love does not die', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=5,sentence='love is never a matter of innumerable twists and turns Never been forsaken never hurt how to know love', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=6,sentence='i adore you i want to be your companion the rest of my life',device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=7, sentence='no one can understand me my poor heart aches',device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=8, sentence='who said that i would go home tonight it s a lie stop fooling me around', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=9, sentence='i never thought i could have passed the calculus exam lucky me', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=10, sentence='what are you guy doing here this is a private room', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=11, sentence='my roommate is gay', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=12, sentence='i am now nearly finished the week detox and i feel amazing', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=13, sentence='he or she is terminally ill', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=14, sentence='he is a big-boned and strong man', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=15, sentence='i can not believe my eyes. it is surprise', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=16, sentence='feel blue today', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=17, sentence='new year，new life', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=18, sentence='who can have thought of it', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=19, sentence='it is hard to believe', device=0)
        test(model=model, tokenizer=tokenizer,sentence_index=20, sentence='i was dumbstruck when i heard about what happened to sam', device=0)

        print("Test bert ends up.")






'''
1、测试内容：huggingface上bert-base-uncased模型在单个GPU上的训练。https://huggingface.co/google-bert/bert-base-uncased
2、数据集：huggingface上的一个Emotion文本分类任务。 https://huggingface.co/datasets/dair-ai/emotion 使用unsplit，一共416809个样例，自己划分。
3、感谢 知乎https://zhuanlan.zhihu.com/p/524487313 的讲解对我本次测试的代码和框架给与了很多帮助，再此感谢。
'''

if __name__ == '__main__':
    main()

