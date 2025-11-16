import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from dnabert_embed import DNABertEmbedder

class VariantGenerator:
    
    def __init__(self, seed: int = 812):
        random.seed(seed)
        np.random.seed(seed)
        self.bases = ['A', 'C', 'G', 'T']
    
    def create_single_variant(self, sequence: str, mutation_type: str = 'SNV') -> Tuple[str, List[Dict]]:
        
        seq_list = list(sequence.upper())
        mutations = []
        
        # 유효한 위치 찾기 (ACGT만 있는 위치)
        valid_positions = [i for i, base in enumerate(seq_list) if base in self.bases]
        
        if not valid_positions:
            return sequence, []
        
        if mutation_type == 'SNV':
            # Single Nucleotide Variant (한 염기 치환)
            pos = random.choice(valid_positions)
            original_base = seq_list[pos]
            
            # 다른 염기로 변경
            possible_bases = [b for b in self.bases if b != original_base]
            new_base = random.choice(possible_bases)
            
            seq_list[pos] = new_base
            
            mutations.append({
                'type': 'SNV',
                'position': pos,
                'original': original_base,
                'mutated': new_base
            })
        
        elif mutation_type == 'deletion':
            # 염기 삭제
            pos = random.choice(valid_positions)
            original_base = seq_list[pos]
            seq_list[pos] = ''
            
            mutations.append({
                'type': 'deletion',
                'position': pos,
                'original': original_base,
                'mutated': '-'
            })
        
        elif mutation_type == 'insertion':
            # 염기 삽입
            pos = random.choice(valid_positions)
            new_base = random.choice(self.bases)
            seq_list.insert(pos, new_base)
            
            mutations.append({
                'type': 'insertion',
                'position': pos,
                'original': '-',
                'mutated': new_base
            })
        
        return ''.join(seq_list), mutations
    
    def create_multiple_variants(self, sequence: str, num_mutations: int = 1) -> Tuple[str, List[Dict]]:
        
        current_seq = sequence
        all_mutations = []
        
        for _ in range(num_mutations):
            current_seq, mutations = self.create_single_variant(current_seq, mutation_type='SNV')
            all_mutations.extend(mutations)
        
        return current_seq, all_mutations
    
    def generate_variant_dataset(self, sequences: List[str], ids: List[str], 
                                 num_variants_per_seq: int = 1) -> pd.DataFrame:
        
        data = []
        
        for idx, (seq_id, seq) in enumerate(zip(ids, sequences)):
            # Reference 추가
            data.append({
                'ID': seq_id,
                'seq': seq,
                'type': 'reference',
                'ref_id': seq_id,
                'mutations': []
            })
            
            # Variant 생성
            for var_num in range(num_variants_per_seq):
                var_seq, mutations = self.create_single_variant(seq, mutation_type='SNV')
                var_id = f"{seq_id}_var{var_num+1}"
                
                data.append({
                    'ID': var_id,
                    'seq': var_seq,
                    'type': 'variant',
                    'ref_id': seq_id,
                    'mutations': mutations
                })
        
        return pd.DataFrame(data)


class VariantEvaluator:
    
    def __init__(self):
        self.embeddings = None
        self.embedding_ids = None
        self.df_data = None
        self.variant_generator = VariantGenerator()
    
    def set_embeddings(self, embeddings: np.ndarray, ids: List[str], df_data: pd.DataFrame = None):
        self.embeddings = embeddings
        self.embedding_ids = ids
        self.df_data = df_data
    
    def create_real_variant_pairs(self) -> List[Tuple[int, int]]:
        
        if self.df_data is None:
            raise ValueError("df_data가 설정되지 않았습니다.")
        
        pairs = []
        
        # reference와 variant를 매칭
        for idx, row in self.df_data.iterrows():
            if row['type'] == 'variant':
                # 해당 variant의 reference 찾기
                ref_id = row['ref_id']
                var_id = row['ID']
                
                # 인덱스 찾기
                try:
                    ref_idx = list(self.embedding_ids).index(ref_id)
                    var_idx = list(self.embedding_ids).index(var_id)
                    pairs.append((ref_idx, var_idx))
                except ValueError:
                    continue
        
        print(f"\n✅ 실제 ref-variant 쌍 생성: {len(pairs)}개")
        
        return pairs
    
    def compute_cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:

        similarity = cosine_similarity([vec1], [vec2])[0, 0]
        distance = 1 - similarity
        return distance
    
    def evaluate_with_real_variants(self, ref_var_pairs: List[Tuple[int, int]]) -> Dict[str, float]:
        
        distances = []
        
        print(f"\n📏 {len(ref_var_pairs)}개 쌍의 코사인 거리 계산 중...")
        
        for i, (ref_idx, var_idx) in enumerate(ref_var_pairs):
            ref_embedding = self.embeddings[ref_idx]
            var_embedding = self.embeddings[var_idx]
            distance = self.compute_cosine_distance(ref_embedding, var_embedding)
            distances.append(distance)
            
            if (i + 1) % 20 == 0 or (i + 1) == len(ref_var_pairs):
                print(f"  ⏳ 진행: {i + 1}/{len(ref_var_pairs)}")
        
        results = {
            'mean_distance': float(np.mean(distances)),
            'median_distance': float(np.median(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'num_pairs': len(distances)
        }
        
        print("✅ 평가 완료!\n")
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, float]):
        """평가 결과 출력"""
        print(f"\n평가 쌍 수: {results['num_pairs']}개")
        print(f"\n코사인 거리 통계:")
        print(f"  • 평균 거리:   {results['mean_distance']:.6f}")
        print(f"  • 중앙값 거리: {results['median_distance']:.6f}")
        print(f"  • 표준편차:    {results['std_distance']:.6f}")
        print(f"  • 최소 거리:   {results['min_distance']:.6f}")
        print(f"  • 최대 거리:   {results['max_distance']:.6f}")

def main():

    # 1. 데이터 로드
    df_original = pd.read_csv("./data/test.csv")
    df_embeddings = pd.read_csv("./data/output/dnabert_embeddings.csv")
    
    print(f"📁 원본 서열: {len(df_original)}개")
    print(f"📁 임베딩: {len(df_embeddings)}개\n")
    
    # 2. 변이 생성
    generator = VariantGenerator(seed=812)
    df_variants = generator.generate_variant_dataset(
        sequences=df_original['seq'].tolist(),
        ids=df_original['ID'].tolist(),
        num_variants_per_seq=1
    )
    print(f"✅ 변이 생성: {len(df_variants)}개 (ref+var)\n")
    
    # 3. 변이 서열 임베딩
    embedder = DNABertEmbedder(device='cpu')
    variant_seqs = df_variants[df_variants['type']=='variant']['seq'].tolist()
    variant_embeddings = embedder.encode_batch(variant_seqs, batch_size=4)
    print()
    
    # 4. 임베딩 통합
    ref_embeddings = df_embeddings[[c for c in df_embeddings.columns if c.startswith('emb_')]].values
    all_embeddings = np.vstack([ref_embeddings, variant_embeddings])
    all_ids = df_embeddings['ID'].tolist() + df_variants[df_variants['type']=='variant']['ID'].tolist()
    
    print(f"통합 임베딩: {all_embeddings.shape}\n")
    
    # 5. 평가
    evaluator = VariantEvaluator()
    evaluator.set_embeddings(all_embeddings, all_ids, df_variants)
    pairs = evaluator.create_real_variant_pairs()
    results = evaluator.evaluate_with_real_variants(pairs)
    evaluator.print_evaluation_results(results)
    
    return results


if __name__ == "__main__":
    results = main()
        
