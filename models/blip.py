import warnings
warnings.filterwarnings("ignore")
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import AutoTokenizer, AutoModel
from models.transformer import Transformer
from models.hopfield_layers import HopfieldLayer
from models.vit_pretrained import *

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]


CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
    'Aorta',
    'Bone/Spine',
    'Hemidiaphragm',
    'Lung Volume'
]

STATE_PROMPTS = {
    0: None,  # 跳过
    1: "is present in the image.",
    2: "is absent in the image.",
    3: "is uncertain in the image."
}

conditions_dict = {i: condition for i, condition in enumerate(CONDITIONS)}

custom_tokens = ['[BLA]', '[POS]', '[NEG]', '[UNC]']

def remove_custom_tokens(text):
    for token in custom_tokens:
        text = text.replace(token, '')
    return text

import torch
import torch.nn as nn
import torch.nn.functional as F

class BioClinicalBERTExtractor:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Loading Bio_ClinicalBERT from: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def extract_features(self, texts, 
                        max_length: int = 512,
                        pooling: str = "mean") -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded)
            last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]

        if pooling == "cls":
            features = last_hidden_state[:, 0, :]
        elif pooling == "mean":
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1)
            features = sum_embeddings / sum_mask
        elif pooling == "max":
            features, _ = torch.max(last_hidden_state, dim=1)
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        return features.cuda()
    
    def extract_cls_features(self, texts, 
                           max_length: int = 100) -> torch.Tensor:
        return self.extract_features(texts, max_length, pooling="cls")

class ResidualSemanticAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256, dropout=0.1):
        super().__init__()

        self.projection1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.projection2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        if input_dim != output_dim:
            self.dim_adapter = nn.Linear(input_dim, output_dim)
        else:
            self.dim_adapter = nn.Identity()
        
    def forward(self, bert_features):
        residual = self.dim_adapter(bert_features)
        x = self.projection1(bert_features)
        x = self.projection2(x)
        output = x + residual
        return output



class SwishGLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * F.silu(gate)

class SequenceFusionWithSwishGLU(nn.Module):
    def __init__(self, input_dims, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.input_dims = input_dims
        self.total_input_dim = sum(input_dims)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.init_method = 'normal'
        self.proj = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim * 2),
            SwishGLU(dim=-1),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif self.init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif self.init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif self.init_method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def _align_sequence_length(self, features):
        max_len = max([f.size(1) for f in features])
        aligned_features = []
        for f in features:
            curr_len = f.size(1)
            
            if curr_len < max_len:
                # 填充
                pad_len = max_len - curr_len
                padding = torch.zeros(f.size(0), pad_len, f.size(2), 
                                    device=f.device, dtype=f.dtype)
                f = torch.cat([f, padding], dim=1)
            elif curr_len > max_len:
                f = f[:, :max_len, :]
            aligned_features.append(f)
        
        return aligned_features
    
    def forward(self, *features):
        aligned_features = self._align_sequence_length(features)
        concatenated = torch.cat(aligned_features, dim=-1)  # [N, L, sum(D_i)]
        projected = self.proj(concatenated)  # [N, L, hidden_dim]
        output = self.output(projected)  # [N, L, output_dim]
        return output

class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 args,
                 tokenizer=None,
                 prompt = '',
                 ):
        super().__init__()
        self.args = args
        vision_width = 768
        self.visual_encoder = VitEncoder()
        state_dict=torch.load('/path/to/medvit_pretrained.pth',map_location='cpu')
        new_dict = {}
        for k, v in state_dict.items():
           if k.startswith('img_encoder.model.'):
              obj = {k.replace('img_encoder.model.', ''): v}
              new_dict.update(obj)
        msg=self.visual_encoder.load_state_dict(state_dict=new_dict,strict=False)
        print(msg)
        
        self.cls_head = nn.Linear(vision_width+512, 18*4)
        nn.init.normal_(self.cls_head.weight, std=0.02)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

        self.textual_proj = nn.Linear(vision_width,768)
        self.textual_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.textual_proj.bias.data.zero_()
        
        self.visual_proj=nn.Linear(vision_width,768)
        self.visual_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.visual_proj.bias.data.zero_()
        
        self.vision_proj = nn.Linear(vision_width, 512)
        self.vision_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.vision_proj.bias.data.zero_()

        self.semantic_adapter= ResidualSemanticAdapter(input_dim=768, hidden_dim=512, output_dim=768, dropout=0.1)
        self.prompt_encoder=BioClinicalBERTExtractor('/path/to/BioClinBERT')

        visual_memory=np.load('hires_cam.npy')
        visual_pattern = torch.from_numpy(visual_memory).unsqueeze(0)
        self.visual_pattern=visual_pattern
        
        self.V_HopfieldLayers = HopfieldLayer(
            input_size=768,
            hidden_size=1024,
            output_size= vision_width, # 768
            pattern_size=768,
            quantity=self.visual_pattern.size(1),
            scaling=None,   
            num_heads=6,
            batch_first=True,         
            normalize_stored_pattern=True,
            normalize_state_pattern=True,
            dropout=0.1
        )
        
        self.V_HopfieldLayers.lookup_weights = nn.Parameter(self.visual_pattern, requires_grad=False)

        textual_memory = np.load('textual_pattern.npy')
        textual_pattern = torch.from_numpy(textual_memory).unsqueeze(0)

        self.R_HopfieldLayers = HopfieldLayer(
            input_size=768,
            hidden_size=1024,
            output_size= vision_width, # 768
            pattern_size=768,
            quantity=textual_pattern.size(1),
            scaling=None,  
            num_heads=6,              
            batch_first=True,         
            normalize_stored_pattern=True,
            normalize_state_pattern=True,
            dropout=0.1
        )
        self.R_HopfieldLayers.lookup_weights = nn.Parameter(textual_pattern, requires_grad=False)
        
        self.fusion_module=SequenceFusionWithSwishGLU((768,768),768,768)
          
        self.tokenizer = tokenizer   
        
        decoder_config = BertConfig.from_json_file('./config/bert_config.json')
        decoder_config.encoder_width = vision_width
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        self.text_decoder = BertLMHeadModel.from_pretrained('./bert-base-uncased',config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        self.memory = Transformer(d_model=512,
                                  num_encoder_layers=1,
                                  num_decoder_layers=1,
                                  num_queries=1)
        
    def forward(self, image, final_prompt ,caption, cls_labels, clip_memory, criterion_cls, base_probs):
            
        image_embeds, avg_embeds = self.visual_encoder(image)

        clip_memory = torch.permute(clip_memory, (1, 0, 2))
        query_embed = self.vision_proj(avg_embeds)
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1)


        image_embeds_rag=self.visual_proj(image_embeds)
        image_embeds_rag=self.V_HopfieldLayers(image_embeds_rag)
        image_embeds=image_embeds+image_embeds_rag


        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 18)
        cls_preds[:, 1, :] += torch.log(torch.from_numpy(base_probs)).view(1, -1).to(image.device)
        loss_cls = criterion_cls(cls_preds, cls_labels)
    
        
        prompt_features=[]
        for prompt in final_prompt:
            prompt_feature=self.prompt_encoder.extract_cls_features(prompt)
            prompt_features.append(prompt_feature)
        prompt_features= torch.cat(prompt_features, dim=0)
        prompt_features=self.semantic_adapter(prompt_features)
        avg_embeds_rag=self.textual_proj(prompt_features)
        avg_embeds_rag=avg_embeds_rag.unsqueeze(dim=1)
        rag_output_r=self.R_HopfieldLayers(avg_embeds_rag)
        rag_output_r=rag_output_r.squeeze(dim=1)
        prompt_features=prompt_features+ rag_output_r

        
        prompt_expanded = prompt_features.unsqueeze(1).repeat(1,image_embeds.size(1),1)
        image_embeds_residual = image_embeds
        image_embeds=self.fusion_module(image_embeds,prompt_expanded)+image_embeds_residual
        
        
        text = self.tokenizer(caption, padding='longest', truncation=True, return_tensors="pt").to(image.device)
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100) 
        decoder_targets[:,:self.prompt_length] = -100
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )

        loss_lm = decoder_output.loss    
        return loss_lm, loss_cls
        
    def generate(self,image, clip_memory, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds, avg_embeds = self.visual_encoder(image) 
       
        clip_memory = torch.permute(clip_memory, (1, 0, 2))
        query_embed = self.vision_proj(avg_embeds)
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1)


        image_embeds_rag=self.visual_proj(image_embeds)
        image_embeds_rag=self.V_HopfieldLayers(image_embeds_rag)
        image_embeds=image_embeds+image_embeds_rag


        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4,18)
        cls_preds = F.softmax(cls_preds, dim=1)
        cls_preds_logits = cls_preds[:, 1, :14]
        cls_preds = torch.argmax(cls_preds, dim=1).cpu().numpy().tolist()
        prefix_prompts = []
        for j in range(len(cls_preds)):
            prefix_prompt = ' '.join([SCORES[c] for c in cls_preds[j]])+' '
            prefix_prompts.append(prefix_prompt)
        final_prompts = []
        for j in range(len(cls_preds)):
            prompts=[]
            for index,c in  enumerate(cls_preds[j]):
                if c == 0:
                   continue
                disease_name = conditions_dict[index]
                prompt_template = STATE_PROMPTS.get(c, "status unknown in the image.")
                prompt = f"{disease_name} {prompt_template}"
                prompts.append(prompt)
            final_prompt=' '.join(prompts)+' '
            final_prompts.append(final_prompt)

        prompt_features=[]
        for prompt_2 in final_prompts:
            prompt_feature=self.prompt_encoder.extract_cls_features(prompt_2)
            prompt_features.append(prompt_feature)
        prompt_features= torch.cat(prompt_features, dim=0)
        prompt_features=self.semantic_adapter(prompt_features)
        avg_embeds_rag=self.textual_proj(prompt_features)
        avg_embeds_rag=avg_embeds_rag.unsqueeze(dim=1)
        rag_output_r=self.R_HopfieldLayers(avg_embeds_rag)
        rag_output_r=rag_output_r.squeeze(dim=1)
        prompt_features=prompt_features+ rag_output_r
        prompt_expanded = prompt_features.unsqueeze(1).repeat(1,image_embeds.size(1),1)
        image_embeds_residual = image_embeds
        image_embeds=self.fusion_module(image_embeds,prompt_expanded)+image_embeds_residual


        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        
        text = self.tokenizer(prefix_prompts, return_tensors="pt")
        input_ids = text.input_ids.to(image.device)
        attn_masks = text.attention_mask.to(image.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 
        attn_masks = attn_masks[:, :-1] 
        outputs = self.text_decoder.generate(input_ids=input_ids,
                                             min_length=min_length, # 4.25 Transformers
                                             max_new_tokens=max_length,
                                             num_beams=num_beams,
                                             eos_token_id=self.tokenizer.sep_token_id,
                                             pad_token_id=self.tokenizer.pad_token_id, 
                                             repetition_penalty=repetition_penalty,
                                             attention_mask = attn_masks,
                                             **model_kwargs) 

        captions = []    
        for i, output in enumerate(outputs):
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(prefix_prompts[0]):])

        return captions, cls_preds, cls_preds_logits

def blip_decoder(args, tokenizer, **kwargs):
    model = BLIP_Decoder(args, tokenizer, **kwargs)
    return model
