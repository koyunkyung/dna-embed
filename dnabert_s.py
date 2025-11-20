import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel # 변경됨
from typing import List
import warnings
warnings.filterwarnings('ignore')

class DNABertSEmbedder:

    def __init__(self, model_name: str = "zhihan1996/DNABERT-S", 
                 max_length: int = 512,
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
        
        print(f"디바이스 사용: {self.device}")
        print(f"모델 로딩 중: {model_name}...")
        gpu_count = torch.cuda.device_count()
        print(f"사용 가능한 GPU 개수: {gpu_count}개")
        
        # [변경] DNABERT-S는 AutoClass와 trust_remote_code=True가 필수입니다.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        
        # 기본 임베딩 차원 확인 (일반적으로 768)
        # DNABERT-S 모델 config 구조에 따라 hidden_size 속성 위치가 다를 수 있어 안전하게 처리
        if hasattr(self.model.config, 'hidden_size'):
            self.base_dim = self.model.config.hidden_size
        else:
            self.base_dim = 768 # Default for BERT base
            
        print(f"기본 임베딩 차원: {self.base_dim}")

        base_model.to(self.device)

        if gpu_count > 1:
            print(f"🔥 {gpu_count}개의 GPU(A40)를 병렬로 사용합니다!")
            self.model = torch.nn.DataParallel(base_model)
        else:
            self.model = base_model

        self.model.eval()
        
        # 차원 조정
        self.embedding_dim = min(embedding_dim, 2048)
        if self.embedding_dim != self.base_dim:
            self.projection = torch.nn.Linear(self.base_dim, self.embedding_dim).to(self.device)
        else:
            self.projection = None
            
        print("✅ 모델 로딩 완료!\n")
    
    # [삭제] DNABERT-S는 k-mer 토큰화 함수가 필요 없습니다.
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """단일 DNA 서열을 DNABERT-S 임베딩 벡터로 변환"""
        
        # DNA 서열 그대로 사용
        dna_seq = sequence.upper() 
        
        # 토큰화 (Raw String 입력)
        inputs = self.tokenizer(
            dna_seq,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 임베딩 추출
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])
            hidden_states = outputs[0] # [1, sequence_length, 768]
            
            # [변경] Mean Pooling 사용 (DNABERT-S 권장 방식)
            # Padding 부분은 제외하고 평균을 구하는 것이 정석이지만, 
            # 간단한 구현을 위해 전체 평균을 사용하거나(논문 구현체 방식), attention mask를 고려할 수 있습니다.
            # 여기서는 DNABERT-S 공식 예제 코드인 torch.mean(hidden_states[0], dim=0) 방식을 따릅니다.
            embedding = torch.mean(hidden_states[0], dim=0) # shape: (hidden_size,)
        
        # 차원 조정이 필요한 경우
        if self.projection is not None:
            embedding = self.projection(embedding)
        
        return embedding # shape: (embedding_dim,)
    
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
        
        # ID 컬럼이 있으면 추가, 없으면 인덱스로 대체 가능하지만 여기선 입력받은 ID 사용
        if ids is not None and len(ids) == len(df):
            df.insert(0, 'ID', ids)
        
        # CSV로 저장
        df.to_csv(output_path, index=False)
        print(f"임베딩 저장 완료: {output_path}")
        print(f"파일 형태: {df.shape}")
        if ids is not None:
            print(f"컬럼: ID, emb_0000 ~ emb_{n_dims-1:04d}")
        else:
            print(f"컬럼: emb_0000 ~ emb_{n_dims-1:04d}")


def main():

    # [설정] 파일 경로
    input_csv = "./data/test.csv"
    output_csv = "./data/output/dnabert_embeddings.csv"
    
    print("="*60)
    print("DNABERT-S 임베딩 추출 시작")
    print("="*60 + "\n")
    
    # 1. 데이터 로드
    # CSV 파일에 'ID'와 'seq' 컬럼이 있다고 가정
    try:
        print(f"입력 파일: {input_csv}")
        df = pd.read_csv(input_csv)
        print(f"📁 로드된 데이터: {len(df)}개 서열\n")
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. ({input_csv})")
        return

    # 2. DNABERT-S 임베더 초기화
    embedder = DNABertSEmbedder(
        model_name="zhihan1996/DNABERT-S", # 모델명 변경
        max_length=512,     # 필요에 따라 조절 (너무 길면 OOM 발생 가능)
        embedding_dim=768,  # DNABERT-S 기본 출력
        device='cuda'        # GPU 사용 시 'cuda'
    )
    
    # 3. 임베딩 생성
    # 데이터 프레임 컬럼명이 다를 경우 수정 필요 (예: df['sequence'])
    embeddings = embedder.encode_batch(
        sequences=df['seq'].tolist(),
        batch_size=256 # GPU 메모리에 따라 조절
    )
    
    # 4. 결과 저장
    embedder.save_embeddings(
        embeddings=embeddings,
        ids=df['ID'].tolist() if 'ID' in df.columns else None,
        output_path=output_csv
    )
    
    print("\n" + "="*60)
    print("임베딩 추출 완료!")
    print("="*60)
    
    return output_csv


if __name__ == "__main__":
    main()