import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.image_size=config['image_size']
        self.patch_size=config['patch_size']
        self.num_channels=config['num_channels']
        self.hidden_size=config['hidden_size']
        
        self.num_patches=(self.image_size//self.patch_size)**2
        
        self.projection=nn.Conv2d(self.num_channels,self.hidden_size,kernel_size=self.patch_size,stride=self.patch_size)
        self.cls_token=nn.Parameter(torch.zeros(1,1,self.hidden_size))
        self.position_embeddings=nn.Parameter(torch.zeros(1,self.num_patches+1,self.hidden_size))
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self,x):
        B,_, H, W = x.shape
        assert H == W == self.image_size, f"Input image size ({H}*{W}) doesn't match model ({self.image_size}*{self.image_size})."
        
        x=self.projection(x) #(B,hidden size,H/patch size,W/patch size)
        x=x.flatten(2).transpose(1,2) #(B,num patches,hidden size)
        
        cls_token=self.cls_token.expand(B,-1,-1)
        x=torch.cat((cls_token,x),dim=1)
        
        x=x+self.position_embeddings
        x=self.dropout(x)
        return x
    
class AttentionHead(nn.Module):
    def __init__(self,dropout,hidden_size,attention_head_size,bias):
        super().__init__()
        self.hidden_size=hidden_size
        self.attention_head_size=attention_head_size
        self.bias=bias
        
        self.query=nn.Linear(self.hidden_size,self.attention_head_size,bias=self.bias)
        self.key=nn.Linear(self.hidden_size,self.attention_head_size,bias=self.bias)
        self.value=nn.Linear(self.hidden_size,self.attention_head_size,bias=self.bias)
        
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        #(B,seq_len,hidden_size)-->(B,seq_len,attention_head_size)
        query=self.query(x) 
        key=self.key(x)
        value=self.value(x)
        
        attention_scores=torch.matmul(query,key.transpose(-1, -2))
        attention_scores=attention_scores/math.sqrt(self.attention_head_size)
        attention_probs=nn.functional.softmax(attention_scores,dim=-1)
        attention_probs=self.dropout(attention_probs)
        attention_output=torch.matmul(attention_probs,value)
        
        return (attention_output, attention_probs)
    
class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.hidden_size=config['hidden_size']
        self.num_attention_heads=config['num_attention_heads']
        self.attention_head_size=self.hidden_size//self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.bias=config['bias']
        self.heads=nn.ModuleList([])
        
        for _ in range(self.num_attention_heads):
            head=AttentionHead(config['attention_dropout'],self.hidden_size,self.attention_head_size,self.bias)
            self.heads.append(head)
        
        self.output_projection=nn.Linear(self.all_head_size,self.hidden_size)
        self.dropout=nn.Dropout(config['dropout'])
        
        
    def forward(self,x,output_attentions=False):
        attention_outputs=[head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)
        
        if not output_attentions:
            return (attention_output,None)
        
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)
        
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dense_1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self,x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config['hidden_size'])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config['hidden_size'])
        
    def forward(self,x,output_attentions=False):
        attention_output,attention_probs=self.attention(self.layernorm_1(x),output_attentions=output_attentions)
        x=x+attention_output
        mlp_output=self.mlp(self.layernorm_2(x))
        x=x+mlp_output
        
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)
        
class Encoder(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.encoder_blocks=nn.ModuleList([])
        for _ in range(config['num_of_encoder_blocks']):
            encoder_block=EncoderBlock(config)
            self.encoder_blocks.append(encoder_block)
        
    def forward(self,x,output_attentions=False):
        attentions=[]
        
        for encoder_block in self.encoder_blocks:
            x,attention_probs=encoder_block(x,output_attentions=output_attentions)
            if output_attentions:
                attentions.append(attention_probs)
        
        if not output_attentions:
            return (x,None)
        else:
            return (x,attentions)
    
class ViTForClassfication(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config['image_size']
        self.hidden_size = config['hidden_size']
        self.num_classes = config['num_classes']

        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        logits = self.classifier(encoder_output[:, 0, :])
        
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, attentions)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config['initializer_range'])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config['initializer_range'],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config['initializer_range'],
            ).to(module.cls_token.dtype)
    
if __name__ == '__main__':
    
    config = {
        'image_size': 224,
        'patch_size': 16,
        'num_channels': 3,
        'hidden_size': 768,
        'dropout': 0.5,
        'attention_dropout':0.5,
        'num_of_encoder_blocks': 6,
        'num_attention_heads': 8,
        'intermediate_size': 4 * 768,
        'initializer_range': 0.02,
        'num_classes': 10,
        'bias': True
    }

    model = ViTForClassfication(config)

    batch_size = 4 
    dummy_input = torch.randn(batch_size, config['num_channels'], config['image_size'], config['image_size'])

    logits, attentions = model(dummy_input, output_attentions=True)

    print("Logits:", logits)
    print("Attention Maps:", attentions)
