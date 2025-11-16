import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel  
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DNABertEmbedder:

    def __init__(self, model_name: str = "zhihan1996/DNA_bert_6", 
                 max_length: int = 512,
                 embedding_dim: int = 768,
                 device: str = None):
        
        self.model_name = model_name
        self.max_length = max_length
        self.embedding_dim = min(embedding_dim, 2048)  # 경진대회 제한
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"디바이스 사용: {self.device}")
        print(f"모델 로딩 중: {model_name}...")
        
        # BertTokenizer와 BertModel 사용
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        
        self.model.eval()  
        
        # DNA-BERT의 기본 임베딩 차원 확인
        self.base_dim = self.model.config.hidden_size
        print(f"기본 임베딩 차원: {self.base_dim}")
        
        # 차원 조정이 필요한 경우 Linear layer 추가
        if self.embedding_dim != self.base_dim:
            self.projection = torch.nn.Linear(self.base_dim, self.embedding_dim).to(self.device)
            print(f"임베딩 차원을 {self.base_dim} -> {self.embedding_dim}으로 조정")
        else:
            self.projection = None
        
        print("✅ 모델 로딩 완료!\n")
    
    def kmer_tokenize(self, sequence: str, k: int = 6) -> str:
        # 대문자로 변환
        sequence = sequence.upper()
        
        # k-mer 생성
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            # 유효한 염기(A, C, G, T)만 포함된 k-mer만 추가
            if all(base in 'ACGT' for base in kmer):
                kmers.append(kmer)
        
        return ' '.join(kmers)
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """단일 DNA 서열을 임베딩 벡터로 변환"""
        # k-mer 토큰화
        kmer_seq = self.kmer_tokenize(sequence)
        
        # 토큰화
        inputs = self.tokenizer(
            kmer_seq,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 임베딩 추출
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS] 토큰의 임베딩 사용 (문장 전체를 대표)
            embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
        
        # 차원 조정이 필요한 경우
        if self.projection is not None:
            embedding = self.projection(embedding)
        
        return embedding.squeeze(0)  # shape: (embedding_dim,)
    
    def encode_batch(self, sequences: List[str], batch_size: int = 8) -> np.ndarray:
        
        embeddings = []
        
        print(f"총 {len(sequences)}개 서열 임베딩 중...")
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # 배치 내 각 서열 처리
            batch_embeddings = []
            for seq in batch_sequences:
                emb = self.encode_sequence(seq)
                batch_embeddings.append(emb.cpu().numpy())
            
            embeddings.extend(batch_embeddings)
            
            # 진행 상황 출력
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(sequences):
                print(f"  진행: {min(i + batch_size, len(sequences))}/{len(sequences)}")
        
        embeddings_array = np.array(embeddings)
        print(f"임베딩 완료! 최종 shape: {embeddings_array.shape}\n")
        
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
        print(f"임베딩 저장 완료: {output_path}")
        print(f"파일 형태: {df.shape}")
        print(f"컬럼: ID, emb_0000 ~ emb_{n_dims-1:04d}")


def main():

    # 파일 경로 설정
    input_csv = "./data/test.csv"
    output_csv = "./data/output/dnabert_embeddings.csv"
    
    print("="*60)
    print("DNA-BERT 임베딩 추출 시작")
    print("="*60 + "\n")
    
    # 1. 데이터 로드
    print(f"입력 파일: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"📁 로드된 데이터: {len(df)}개 서열\n")
    
    # 2. DNA-BERT 임베더 초기화
    embedder = DNABertEmbedder(
        model_name="zhihan1996/DNA_bert_6",
        max_length=512,
        embedding_dim=768,  # 기본 768차원 사용 (필요시 2048까지 가능)
        device='cpu'  # GPU 사용 가능 시 'cuda'로 변경
    )
    
    # 3. 임베딩 생성
    embeddings = embedder.encode_batch(
        sequences=df['seq'].tolist(),
        batch_size=4
    )
    
    # 4. 결과 저장
    embedder.save_embeddings(
        embeddings=embeddings,
        ids=df['ID'].tolist(),
        output_path=output_csv
    )
    
    print("\n" + "="*60)
    print("임베딩 추출 완료!")
    print("="*60)
    
    return output_csv


if __name__ == "__main__":
    output_file = main()