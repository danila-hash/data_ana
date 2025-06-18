# first line: 33
    def _fit_model(self, texts, min_topic_size=10, n_neighbors=15, n_components=5):
        """
        Fit the BERTopic model to the texts (internal method).
        
        Args:
            texts (list): List of text documents
            min_topic_size (int): Minimum size of topics
            n_neighbors (int): Number of neighbors for UMAP
            n_components (int): Number of components for UMAP
        """
        print("Fitting BERTopic model...")
        
        # Initialize UMAP with adjusted parameters for better separation
        umap_model = UMAP(
            n_neighbors=10,  # Reduced from 15 to get more local structure
            n_components=10,  # Increased from 5 to allow more dimensions for separation
            min_dist=0.1,  # Increased from 0.0 to allow more spacing between clusters
            metric='cosine',
            random_state=42
        )
        
        # Initialize HDBSCAN with more aggressive clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=5,  # Reduced from 10 to allow smaller clusters
            min_samples=3,  # Reduced from 5 to allow more clusters
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            gen_min_span_tree=True  # Enable minimum spanning tree for better cluster separation
        )
        
        # Initialize CountVectorizer with stopwords
        vectorizer_model = CountVectorizer(
            min_df=2,
            token_pattern=r'(?u)\b\w+\b',
            stop_words='english'  # Add English stopwords
        )
        
        # Create and fit BERTopic model
        topic_model = BERTopic(
            embedding_model=self.model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=True,
            nr_topics=15,  # Explicitly set number of topics
            calculate_probabilities=True  # Ensure we get probability distributions
        )
        
        # Fit the model directly on texts
        topics, probs = topic_model.fit_transform(texts)
        return topic_model, topics, probs
