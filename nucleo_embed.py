import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List
import warnings
warnings.filterwarnings('ignore')


class NucleotideTransformerEmbedder:
    
    def __init__(self, 
                 model_name: str = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                 max_length: int = 1000,
                 embedding_dim: int = 768,
                 device: str = None):
        
        self.model_name = model_name
        self.max_length = max_length
        self.embedding_dim = min(embedding_dim, 2048)
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Nucleotide Transformer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)
        
        self.model.eval()
        
        # 모델의 기본 임베딩 차원
        # Nucleotide Transformer는 esm (Evolutionary Scale Modeling) 구조 사용
        self.base_dim = self.model.config.hidden_size
        print(f"✅ 기본 임베딩 차원: {self.base_dim}")
        
        # 차원 조정이 필요한 경우
        if self.embedding_dim != self.base_dim:
            self.projection = torch.nn.Linear(self.base_dim, self.embedding_dim).to(self.device)
            print(f"📐 임베딩 차원 조정: {self.base_dim} -> {self.embedding_dim}")
        else:
            self.projection = None
        
        print("✅ 모델 로딩 완료!\n")
    
    def prepare_sequence(self, sequence: str) -> str:
        # 대문자로 변환
        sequence = sequence.upper()
        
        # 유효한 염기만 필터링
        valid_bases = set('ACGT')
        filtered_seq = ''.join([base for base in sequence if base in valid_bases])
        
        # 공백으로 구분 (각 염기를 개별 토큰으로)
        spaced_seq = ' '.join(filtered_seq)
        
        return spaced_seq
    
    def encode_sequence(self, sequence: str, pooling: str = 'mean') -> torch.Tensor:
    
        # 서열 전처리
        prepared_seq = self.prepare_sequence(sequence)
        
        # 토큰화
        inputs = self.tokenizer(
            prepared_seq,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 임베딩 추출
        with torch.no_grad():
            # Nucleotide Transformer의 출력 구조
            outputs = self.model.esm(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # 마지막 레이어
            
            # Pooling 전략 선택
            if pooling == 'cls':
                # [CLS] 토큰 (첫 번째 토큰)
                embedding = hidden_states[:, 0, :]
            elif pooling == 'mean':
                # 평균 풀링 (attention mask 고려)
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embedding = sum_embeddings / sum_mask
            elif pooling == 'max':
                # 최대 풀링
                embedding = torch.max(hidden_states, dim=1)[0]
            else:
                # 기본: mean pooling
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embedding = sum_embeddings / sum_mask
        
        # 차원 조정이 필요한 경우
        if self.projection is not None:
            embedding = self.projection(embedding)
        
        return embedding.squeeze(0)
    
    def encode_batch(self, sequences: List[str], batch_size: int = 4, 
                     pooling: str = 'mean') -> np.ndarray:
        
        embeddings = []
        total = len(sequences)
        
        print(f"🧬 {total}개 서열 임베딩 중 (pooling: {pooling})...")
        
        for i in range(0, total, batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_embeddings = []
            
            for seq in batch_sequences:
                emb = self.encode_sequence(seq, pooling=pooling)
                batch_embeddings.append(emb.cpu().detach().numpy())
            
            embeddings.extend(batch_embeddings)
            
            # 진행 상황 출력
            if (i + batch_size) % 50 == 0 or (i + batch_size) >= total:
                progress = min(i + batch_size, total)
                print(f"  ⏳ 진행: {progress}/{total} ({progress/total*100:.1f}%)")
        
        embeddings_array = np.array(embeddings)
        print(f"✅ 임베딩 완료! shape: {embeddings_array.shape}\n")
        
        return embeddings_array
    
    def save_embeddings(self, embeddings: np.ndarray, ids: List[str], output_path: str):
        
        # 컬럼 이름 생성 (emb_0000, emb_0001, ..., emb_0767 형식)
        n_dims = embeddings.shape[1]
        column_names = [f'emb_{i:04d}' for i in range(n_dims)]
        
        # DataFrame 생성
        df = pd.DataFrame(embeddings, columns=column_names)
        df.insert(0, 'ID', ids)
        
        # CSV로 저장
        df.to_csv(output_path, index=False)
        print(f"💾 임베딩 저장 완료: {output_path}")
        print(f"   - 파일 형태: {df.shape}")
        print(f"   - 컬럼: ID, emb_0000 ~ emb_{n_dims-1:04d}")


def main():
    
    # 파일 경로 설정
    input_csv = "./data/test.csv"
    output_csv = "./data/output/nucleotide_embeddings.csv"
    
    df = pd.read_csv(input_csv)
    print(f"✅ 로드된 데이터: {len(df)}개 서열\n")
    
    # 2. Nucleotide Transformer 초기화
    embedder = NucleotideTransformerEmbedder(
        model_name="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        max_length=1000,  # Nucleotide Transformer는 더 긴 서열 처리 가능
        embedding_dim=768,  # 경진대회 형식 (768차원)
        device='cpu'  
    )
    
    # 3. 임베딩 생성
    embeddings = embedder.encode_batch(
        sequences=df['seq'].tolist(),
        batch_size=2,  # 큰 모델이므로 배치 크기 작게
        pooling='mean'  # mean pooling이 변이 감지에 더 좋음
    )
    
    # 4. 결과 저장
    embedder.save_embeddings(
        embeddings=embeddings,
        ids=df['ID'].tolist(),
        output_path=output_csv
    )
    
    print(f"   - 입력 서열: {len(df)}개")
    print(f"   - 임베딩 차원: {embeddings.shape[1]}")
    print(f"   - 출력 파일: {output_csv}")
    
    return output_csv


if __name__ == "__main__":
    output_file = main()