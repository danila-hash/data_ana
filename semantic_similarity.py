import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class SemanticSimilarity:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the SemanticSimilarity class with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.similarity_matrix = None
        
        # Create directory for saving results
        os.makedirs('semantic_similarity_res', exist_ok=True)

    def create_embeddings(self, texts):
        """
        Create embeddings for the input texts.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Document embeddings
        """
        print("Creating embeddings...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        return self.embeddings

    def calculate_similarity_matrix(self, texts):
        """
        Calculate similarity matrix between all pairs of documents.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        if self.embeddings is None:
            self.create_embeddings(texts)
        
        print("Calculating similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        return self.similarity_matrix

    def find_similar_articles(self, texts, dates=None, threshold=0.6):
        """
        Find pairs of similar articles based on similarity threshold.
        
        Args:
            texts (list): List of text documents
            dates (list, optional): List of dates corresponding to the documents
            threshold (float): Similarity threshold (0 to 1)
            
        Returns:
            pd.DataFrame: DataFrame containing similar article pairs
        """
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix(texts)
        
        # Create a list to store similar article pairs
        similar_pairs = []
        
        # Find pairs above threshold
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if self.similarity_matrix[i, j] >= threshold:
                    pair = {
                        'article1_idx': i,
                        'article2_idx': j,
                        'similarity': self.similarity_matrix[i, j],
                        'article1_text': texts[i],
                        'article2_text': texts[j]
                    }
                    if dates is not None:
                        pair['article1_date'] = dates[i]
                        pair['article2_date'] = dates[j]
                    similar_pairs.append(pair)
        
        # Create DataFrame
        similar_df = pd.DataFrame(similar_pairs)
        
        # Sort by similarity score
        similar_df = similar_df.sort_values('similarity', ascending=False)
        
        # Save results
        similar_df.to_csv('semantic_similarity_res/similar_articles.csv', index=False)
        
        # Print summary
        print(f"\nFound {len(similar_pairs)} pairs of similar articles (similarity >= {threshold})")
        print("\nTop 5 most similar pairs:")
        print(similar_df.head().to_string())
        
        return similar_df

    def analyze_similarity_distribution(self, texts):
        """
        Analyze the distribution of similarity scores.
        
        Args:
            texts (list): List of text documents
        """
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix(texts)
        
        # Get upper triangular part of similarity matrix (excluding diagonal)
        upper_tri = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
        
        # Calculate statistics
        stats = {
            'mean_similarity': np.mean(upper_tri),
            'median_similarity': np.median(upper_tri),
            'std_similarity': np.std(upper_tri),
            'min_similarity': np.min(upper_tri),
            'max_similarity': np.max(upper_tri)
        }
        
        # Save statistics
        pd.DataFrame([stats]).to_csv('semantic_similarity_res/similarity_stats.csv', index=False)
        
        print("\nSimilarity Score Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")
        
        return stats

    def run_analysis(self, texts, dates=None, threshold=0.6):
        """
        Run complete semantic similarity analysis pipeline.
        
        Args:
            texts (list): List of text documents
            dates (list, optional): List of dates corresponding to the documents
            threshold (float): Similarity threshold (0 to 1)
        """
        # Calculate similarity matrix
        self.calculate_similarity_matrix(texts)
        
        # Analyze similarity distribution
        self.analyze_similarity_distribution(texts)
        
        # Find similar articles
        similar_articles = self.find_similar_articles(texts, dates, threshold)
        
        return similar_articles
