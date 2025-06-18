# first line: 35
    def _create_embeddings(self, texts):
        """
        Create embeddings for the input texts (internal method).
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Document embeddings
        """
        print("Creating embeddings...")
        return self.model.encode(texts, show_progress_bar=True)
