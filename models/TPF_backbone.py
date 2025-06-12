import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType, get_peft_model
from transformers import GPT2Tokenizer
import torch.nn.functional as F
from models.GPT2_arch import AccustumGPT2Model
from layers.Freq_Block import Freq_Block


class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        self.count = 0
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1)
        self.word_embedding = word_embedding.T

    def forward(self, x, prompt, time_fusion):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)
        x = self.linear(x)
        time_fusion = self.linear(time_fusion)
        # time_fusion,_ = self.cross_attention(time_fusion.transpose(0, 1),self.word_embedding.transpose(0, 1),self.word_embedding.transpose(0, 1))
        x_time = x
        time_pub = torch.cat((prompt, x_time), dim=1)

        time_private_prompt = time_pub
        # time_private_prompt = time_private_prompt.transpose(0,1)
        # time_private_prompt,_=self.self_attention(time_private_prompt,time_private_prompt,time_private_prompt)
        time_pub, _ = self.self_attention(time_pub, time_pub, time_pub)
        q = time_fusion.transpose(0, 1)
        k = v = time_pub.transpose(0, 1)
        time_pub, heat = self.cross_attention(q, k, v)
        # if flag:
        #     self.count = self.count + 1
        print("score:", heat[0].shape)
        #     plot_heatmap(heat[0].detach().cpu().numpy(),save_path = self.count)
        time_pub = time_pub.transpose(0, 1)
        time_private_fusion = time_fusion

        time_private_prompt = time_private_prompt.transpose(1, 2)
        time_private_prompt = nn.Linear(time_private_prompt.shape[-1], time_private_fusion.shape[1]).to(
            self.linear.weight.device)(time_private_prompt)
        time_private_prompt = time_private_prompt.transpose(1, 2)

        return time_pub, time_private_fusion, time_private_prompt


def prompt_build(x, description, factor, seq_len, pred_len):
    B, T, N = x.shape
    prompt = []
    # 构建提示词
    for b in range(B):  # 遍历每个样本
        prompt_ = (
            f"<|start_prompt|>Dataset description: {description} "
            f"<|start_prompt|>External factors: {factor} "
            f"Task description: forecast the next {str(pred_len)} steps given the previous {str(seq_len)} steps information. "
        )
        prompt.append(prompt_)

    return prompt


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.cross_attention = None
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.content = configs.content
        self.device = device
        self.factor = configs.factor
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn", "c_proj"]
        )

        self.task_name = configs.task_name

        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.gpt2_fussion = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                              output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.gpt2_prompt = AccustumGPT2Model.from_pretrained("gpt2", output_attentions=True,
                                                             output_hidden_states=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.gpt2_fussion.h = self.gpt2_fussion.h[:configs.gpt_layers]
        self.gpt2_prompt.h = self.gpt2_prompt.h[:configs.gpt_layers]

        # self.gpt2_prompt=get_peft_model(self.gpt2_prompt,peft_config)
        self.gpt2 = get_peft_model(self.gpt2, peft_config)
        # self.gpt2_fussion = get_peft_model(self.gpt2_fussion, peft_config)

        word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for i, (name, param) in enumerate(self.gpt2_fussion.named_parameters()):
            if 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)])

        self.text_proj = nn.ModuleList(
            [nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers + 1)])

        self.in_layer = Encoder_PCA(configs.seq_len, word_embedding, hidden_dim=configs.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        for layer in (self.gpt2_fussion, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

        self.freq_block = Freq_Block(configs, device, word_embedding).to(device)

    def forecast(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        # create prompt
        prompt = prompt_build(x, self.content, self.factor, self.seq_len, self.pred_len)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids

        prompt_embeddings = self.gpt2.get_input_embeddings()(prompt.to(x.device))  # (batch, prompt_token, dim)

        # times net 1D->2D->1D
        time_fusion = self.freq_block.forward(x)

        x = rearrange(x, 'b l m -> b m l')
        time_fusion = rearrange(time_fusion, 'b l m -> b m l')

        time_pub, time_private_fusion, time_private_prompt = self.in_layer(x, prompt=prompt_embeddings,
                                                                           time_fusion=time_fusion)

        outputs_prompt, intermidiate_feat_prompt = self.gpt2_prompt(inputs_embeds=time_private_prompt)
        outputs_text, intermidiate_feat_text = self.gpt2(inputs_embeds=time_pub)
        outputs_time, intermidiate_feat_time = self.gpt2_fussion(inputs_embeds=time_private_fusion)

        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8).to(x.device)
        outputs_prompt += time_private_prompt
        outputs_text += time_pub
        outputs_time += time_private_fusion

        intermidiate_feat_prompt = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_prompt))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])

        outputs_prompt = self.out_layer(outputs_prompt[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])
        outputs_time = self.out_layer(outputs_time[:, -M:, :])

        outputs_prompt = rearrange(outputs_prompt, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')
        outputs_time = rearrange(outputs_time, 'b m l -> b l m')

        outputs_prompt = outputs_prompt * stdev + means
        outputs_time = outputs_time * stdev + means
        outputs_text = outputs_text * stdev + means

        self.w = nn.Parameter(torch.ones(3)).to(self.device)
        self.w = self.w / self.w.sum()
        ans = self.w[0] * outputs_time + self.w[1] * outputs_text + self.w[2] * outputs_prompt

        return {
            'outputs_text': ans,
            'outputs_time': ans,
            'outputs_prompt': ans,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
            'intermidiate_prompt': intermidiate_feat_prompt,
        }

    # Modify the application according to the above !!!
    def classification(self, x):
        B, L, M = x.shape

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_text1)
        outputs_text, intermidiate_feat_text = self.gpt2_fussion(inputs_embeds=outputs_text1)

        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = outputs_time.reshape(B, -1)
        outputs_text = outputs_text.reshape(B, -1)

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
        }

    def imputation(self, x, mask):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)

        stdev = torch.sqrt(torch.sum(x ** 2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_fussion(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
        }

    def anomaly_detection(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_fussion(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1

        intermidiate_feat_time = tuple(
            [self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple(
            [self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time': outputs_time,
            'intermidiate_time': intermidiate_feat_time,
            'intermidiate_text': intermidiate_feat_text,
        }

    def forward(self, x, mask=None, ):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output
