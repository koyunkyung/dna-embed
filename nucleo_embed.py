import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')


class NucleotideTransformerEmbedder:
    
    def __init__(self, 
                 model_name: str = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                 max_length: int = 1000,
                 embedding_dim: int = None,
                 device: str = 'cuda:0',
                 use_fp16: bool = True):
        
        self.model_name = model_name
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device = torch.device(device)
        
        print(f"🚀 디바이스: {self.device}")
        if self.device.type == 'cuda':
            gpu_id = self.device.index if self.device.index is not None else 0
            print(f"   GPU: {torch.cuda.get_device_name(gpu_id)} "
                  f"({torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB)")
        
        # Nucleotide Transformer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # FP16 사용 시
        if use_fp16 and self.device.type == 'cuda':
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)
            self.dtype = torch.float16
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                trust_remote_code=True
            ).to(self.device)
            self.dtype = torch.float32
        
        self.model.eval()
        
        # 모델의 기본 임베딩 차원
        self.base_dim = self.model.config.hidden_size
        
        # embedding_dim이 지정되지 않으면 기본 차원 사용
        if embedding_dim is None:
            self.embedding_dim = self.base_dim
        else:
            self.embedding_dim = min(embedding_dim, 2048)
            
        # 차원 조정이 필요한 경우에만 projection 생성
        if self.embedding_dim != self.base_dim:
            self.projection = torch.nn.Linear(self.base_dim, self.embedding_dim)
            self.projection = self.projection.to(dtype=self.dtype, device=self.device)
        else:
            self.projection = None
    
    def prepare_sequence(self, sequence: str) -> str:
        sequence = sequence.upper()
        valid_bases = set('ACGT')
        filtered_seq = ''.join([base for base in sequence if base in valid_bases])
        spaced_seq = ' '.join(filtered_seq)
        return spaced_seq
    
    def encode_batch(self, sequences: List[str], batch_size: int = 16, 
                     pooling: str = 'mean') -> np.ndarray:
        """
        단일 GPU에서 배치 처리
        """
        embeddings = []
        total = len(sequences)
        
        for i in range(0, total, batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # 배치 전처리
            prepared_seqs = [self.prepare_sequence(seq) for seq in batch_sequences]
            
            # 배치 토큰화
            inputs = self.tokenizer(
                prepared_seqs,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 배치 임베딩 추출
            with torch.no_grad():
                outputs = self.model.esm(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                # Mean pooling
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # 차원 조정
                if self.projection is not None:
                    batch_embeddings = self.projection(batch_embeddings)
                batch_embeddings = torch.nan_to_num(
                    batch_embeddings,
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )
            
            # CPU로 이동 및 numpy 변환
            embeddings.append(batch_embeddings.cpu().float().detach().numpy())
        
        return np.vstack(embeddings)


class MultiGPUEmbedder:
    
    def __init__(self, 
                 model_name: str = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                 max_length: int = 1000,
                 embedding_dim: int = None,
                 use_fp16: bool = True):
        
        self.n_gpus = torch.cuda.device_count()
        print(f"🔍 사용 가능한 GPU: {self.n_gpus}개")
        
        for i in range(self.n_gpus):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
        print()
        
        # 각 GPU에 별도 모델 인스턴스 생성
        print("📥 GPU별 모델 로딩 중...")
        self.embedders = []
        
        for gpu_id in range(min(self.n_gpus, 2)):  # 최대 2개 GPU 사용
            print(f"\n🔧 GPU {gpu_id} 초기화 중...")
            embedder = NucleotideTransformerEmbedder(
                model_name=model_name,
                max_length=max_length,
                embedding_dim=embedding_dim,
                device=f'cuda:{gpu_id}',
                use_fp16=use_fp16
            )
            self.embedders.append(embedder)
        
        self.n_workers = len(self.embedders)
        print(f"\n✅ {self.n_workers}개 GPU 준비 완료!\n")
    
    def encode_batch_multi_gpu(self, sequences: List[str], batch_size: int = 16,
                              pooling: str = 'mean') -> np.ndarray:
        """
        여러 GPU에서 병렬로 배치 처리
        """
        total = len(sequences)
        
        # 데이터를 GPU 개수만큼 분할
        chunk_size = (total + self.n_workers - 1) // self.n_workers
        chunks = [sequences[i:i+chunk_size] for i in range(0, total, chunk_size)]
        
        print(f"🧬 {total}개 서열 임베딩 중")
        print(f"   - GPU 개수: {self.n_workers}개")
        print(f"   - GPU당 처리: {chunk_size}개")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Pooling: {pooling}")
        print()
        
        start_time = time.time()
        
        def process_chunk(gpu_id, chunk):
            """각 GPU에서 chunk 처리"""
            embedder = self.embedders[gpu_id]
            chunk_embeddings = []
            
            for i in range(0, len(chunk), batch_size):
                batch_start = time.time()
                batch = chunk[i:i+batch_size]
                
                # 배치 처리
                batch_emb = embedder.encode_batch(batch, batch_size=len(batch), pooling=pooling)
                chunk_embeddings.append(batch_emb)
                
                # 진행 상황 출력
                progress = i + len(batch)
                if progress % 100 == 0 or progress >= len(chunk):
                    batch_time = time.time() - batch_start
                    mem_allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
                    
                    print(f"  GPU{gpu_id} ⏳ {progress}/{len(chunk)} ({progress/len(chunk)*100:.1f}%) | "
                          f"배치: {batch_time:.2f}s | "
                          f"메모리: {mem_allocated:.1f}/{mem_reserved:.1f}GB")
            
            return np.vstack(chunk_embeddings)
        
        # 멀티스레딩으로 각 GPU에서 동시 처리
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_chunk, gpu_id, chunk) 
                      for gpu_id, chunk in enumerate(chunks)]
            results = [future.result() for future in futures]
        
        # 결과 합치기
        embeddings_array = np.vstack(results)
        total_time = time.time() - start_time
        
        print(f"\n✅ 임베딩 완료! shape: {embeddings_array.shape}")
        print(f"⏱️  총 소요 시간: {total_time/60:.2f}분 ({total_time:.1f}초)")
        print(f"📊 평균 속도: {total/total_time:.1f} sequences/second")
        print(f"🚀 GPU당 처리량: {total/total_time/self.n_workers:.1f} sequences/second/GPU\n")
        
        return embeddings_array
    
    def save_embeddings(self, embeddings: np.ndarray, ids: List[str], output_path: str):
        # 컬럼 이름 생성
        n_dims = embeddings.shape[1]
        column_names = [f'emb_{i:04d}' for i in range(n_dims)]
        
        # DataFrame 생성
        df = pd.DataFrame(embeddings, columns=column_names)
        df.insert(0, 'ID', ids)
        
        # CSV로 저장
        print(f"💾 CSV 저장 중...")
        df.to_csv(output_path, index=False)
        print(f"✅ 임베딩 저장 완료: {output_path}")
        print(f"   - 파일 형태: {df.shape}")
        print(f"   - 컬럼: ID, emb_0000 ~ emb_{n_dims-1:04d}")


def main():
    
    # 파일 경로 설정
    input_csv = "./data/test.csv"
    output_csv = "./data/output/nucleotide_embeddings.csv"
    
    print("="*60)
    print("🧬 Nucleotide Transformer 임베딩 생성 (Multi-GPU)")
    print("="*60)
    print()
    
    # 1. 데이터 로드
    print("📂 데이터 로딩...")
    df = pd.read_csv(input_csv)
    print(f"✅ 로드된 데이터: {len(df):,}개 서열\n")
    
    # 2. Multi-GPU Embedder 초기화
    embedder = MultiGPUEmbedder(
        model_name="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        max_length=1000,
        embedding_dim=2048,  # 2048차원
        use_fp16=True
    )
    
    # 3. 임베딩 생성 (멀티 GPU)
    embeddings = embedder.encode_batch_multi_gpu(
        sequences=df['seq'].tolist(),
        batch_size=16,  # 각 GPU당 배치 크기
        pooling='mean'
    )
    
    # 4. 결과 저장
    embedder.save_embeddings(
        embeddings=embeddings,
        ids=df['ID'].tolist(),
        output_path=output_csv
    )
    
    return output_csv


if __name__ == "__main__":
    output_file = main()