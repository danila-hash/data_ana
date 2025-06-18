import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import os
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn.functional as F

class ThemeClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the ThemeClassifier with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.theme_keywords = {
            'Legal and Regulatory Affairs': {
                'trial', 'ruling', 'court', 'judge', 'lawsuit', 'sec', 'regulation',
                'regulatory', 'legal', 'bankman', 'fried', 'sbf', 'ftx', 'settlement',
                'compliance', 'investigation', 'prosecutor', 'attorney', 'justice',
                'law', 'enforcement', 'verdict', 'testimony', 'witness', 'evidence',
                'charges', 'criminal', 'civil', 'case', 'appeal', 'defendant'
            },
            'Security and Fraud': {
                'hack', 'attack', 'exploit', 'vulnerability', 'security', 'breach',
                'fraud', 'scam', 'phishing', 'malware', 'ransomware', 'stolen',
                'theft', 'compromise', 'suspicious', 'laundering', 'investigation',
                'forensic', 'incident', 'recovery', 'patch', 'fix', 'audit',
                'protection', 'firewall', 'encryption'
            },
            'NFTs and Digital Collectibles': {
                'nft', 'collectible', 'art', 'auction', 'rare', 'unique', 'gallery',
                'artist', 'creator', 'collection', 'opensea', 'metadata', 'mint',
                'token', 'digital', 'asset', 'ownership', 'royalty', 'creative',
                'artwork', 'exhibition', 'auction', 'bid', 'sale', 'collector',
                'bored', 'ape', 'punk', 'metaverse'
            },
            'Technology and Innovation': {
                'ai', 'artificial', 'intelligence', 'machine', 'learning', 'blockchain',
                'protocol', 'development', 'upgrade', 'innovation', 'technology',
                'software', 'hardware', 'platform', 'infrastructure', 'architecture',
                'scalability', 'performance', 'optimization', 'gaming', 'game',
                'mobile', 'app', 'application', 'interface', 'sdk', 'api'
            },
            'Governance and Organizations': {
                'dao', 'governance', 'vote', 'proposal', 'community', 'decentralized',
                'autonomous', 'organization', 'member', 'stakeholder', 'decision',
                'policy', 'protocol', 'consensus', 'participation', 'election',
                'ballot', 'quorum', 'amendment', 'constitution', 'framework',
                'leadership', 'management', 'board', 'executive'
            },
            'Market and Financial Services': {
                'market', 'trade', 'price', 'volume', 'volatility', 'exchange',
                'banking', 'financial', 'institution', 'payment', 'transaction',
                'liquidity', 'asset', 'investment', 'portfolio', 'fund', 'capital',
                'trading', 'analysis', 'forecast', 'trend', 'indicator', 'chart',
                'technical', 'fundamental', 'strategy'
            }
        }
        
        # Theme descriptions for semantic similarity
        self.theme_descriptions = {
            'Legal and Regulatory Affairs': 'Legal proceedings, regulatory compliance, court cases, investigations, and enforcement actions in the cryptocurrency industry.',
            'Security and Fraud': 'Security incidents, hacks, exploits, fraud cases, cybersecurity measures, and protective actions in blockchain systems.',
            'NFTs and Digital Collectibles': 'Non-fungible tokens, digital art, collectibles, virtual assets, and the NFT marketplace ecosystem.',
            'Technology and Innovation': 'Technological developments, innovations, AI integration, blockchain protocols, and technical improvements.',
            'Governance and Organizations': 'Decentralized governance, organizational structures, voting systems, and community-driven decision making.',
            'Market and Financial Services': 'Market trends, trading activities, financial services, investment strategies, and economic aspects of cryptocurrency.'
        }
        
        # Create theme embeddings
        self.theme_embeddings = {
            theme: self.model.encode(desc, convert_to_tensor=True)
            for theme, desc in self.theme_descriptions.items()
        }
        
        # Priority order for tie-breaking
        self.priority_order = [
            'Legal and Regulatory Affairs',
            'Security and Fraud',
            'NFTs and Digital Collectibles',
            'Technology and Innovation',
            'Governance and Organizations',
            'Market and Financial Services'
        ]

    def _count_theme_keywords(self, text, keywords):
        """Count how many keywords from a theme appear in the text."""
        return sum(1 for keyword in keywords if keyword in text.lower())

    def _get_semantic_similarity(self, text):
        """Get semantic similarity scores between text and theme descriptions."""
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        # Ensure embeddings are on the same device
        text_embedding = text_embedding.to(next(iter(self.theme_embeddings.values())).device)
        
        similarities = {}
        for theme, theme_emb in self.theme_embeddings.items():
            # Reshape embeddings for cosine similarity calculation
            text_emb_reshaped = text_embedding.view(1, -1)
            theme_emb_reshaped = theme_emb.view(1, -1)
            # Calculate cosine similarity
            similarity = F.cosine_similarity(text_emb_reshaped, theme_emb_reshaped)
            similarities[theme] = float(similarity.item())
        
        return similarities

    def classify_text(self, text):
        """
        Classify a single text using keyword matching and semantic similarity.
        
        Args:
            text (str): Text to classify
            
        Returns:
            str: Theme label
        """
        # Try keyword matching first
        theme_counts = {
            theme: self._count_theme_keywords(text, keywords)
            for theme, keywords in self.theme_keywords.items()
        }
        
        max_count = max(theme_counts.values())
        if max_count > 0:
            # If there are keyword matches, use them
            best_themes = [
                theme for theme, count in theme_counts.items()
                if count == max_count
            ]
            
            # If multiple themes have the same keyword count, use priority order
            for theme in self.priority_order:
                if theme in best_themes:
                    return theme
        
        # If no keywords match or too many matches, use semantic similarity
        similarities = self._get_semantic_similarity(text)
        return max(similarities.items(), key=lambda x: x[1])[0]

    def classify_texts(self, texts):
        """
        Classify multiple texts.
        
        Args:
            texts (list): List of texts to classify
            
        Returns:
            list: List of theme labels
        """
        return [self.classify_text(text) for text in texts]


class TopicMining:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the TopicMining class with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.topic_model = None
        self.embeddings = None
        self.topics = None
        self.probs = None
        self.theme_classifier = ThemeClassifier(model_name)
        
        # Create directory for saving results
        os.makedirs('topic_mining_res', exist_ok=True)

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

    def fit_model(self, texts, min_topic_size=10, n_neighbors=15, n_components=5):
        """
        Fit the BERTopic model to the texts.
        
        Args:
            texts (list): List of text documents
            min_topic_size (int): Minimum size of topics
            n_neighbors (int): Number of neighbors for UMAP
            n_components (int): Number of components for UMAP
        """
        print("Fitting BERTopic model...")
        
        # Initialize UMAP with parameters for better topic separation
        umap_model = UMAP(
            n_neighbors=5,  # Slightly increased for more stable structure
            n_components=8,  # Reduced to prevent over-separation
            min_dist=0.05,  # Increased for more natural clusters
            metric='cosine',
            random_state=42
        )
        
        # Initialize HDBSCAN with parameters for more granular clusters
        hdbscan_model = HDBSCAN(
            min_cluster_size=40,  # Increased for more stable clusters
            min_samples=5,  # Increased for more robust clusters
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            cluster_selection_epsilon=0.05  # Increased for more inclusive clusters
        )
        
        # Define comprehensive stopwords list
        stop_words = {
            # Basic English stopwords
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'can', 'will', 'just', 'should', 'now', 'get', 'got', 'getting',
            
            # Crypto terms that are too generic
            'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain', 'cryptocurrencies', 
            'token', 'tokens', 'coin', 'coins', 'digital', 'currency', 'currencies', 'market',
            'markets', 'trading', 'trade', 'price', 'prices', 'exchange', 'exchanges',
            'wallet', 'wallets', 'mining', 'miner', 'miners', 'network', 'networks',
            'transaction', 'transactions', 'block', 'blocks', 'chain', 'chains',
            'platform', 'platforms', 'asset', 'assets', 'volume', 'volumes', 'bull', 'bear',
            'bullish', 'bearish', 'rally', 'dip', 'pump', 'dump', 'hodl', 'fud', 'fomo',
            'support', 'resistance', 'analysis', 'trend', 'trends', 'chart', 'charts',
            'technical', 'fundamental', 'indicator', 'indicators', 'pattern', 'patterns',
            'level', 'levels', 'zone', 'zones', 'range', 'ranges', 'term', 'outlook',
            
            # Common contractions and possessives
            "'s", "'ll", "'t", "'re", "'ve", "'m", "'d", "n't", "'ll",
            "s", "t", "m", "re", "ve", "d", "ll", "ve",
            
            # Numbers and units
            'one', 'two', 'three', 'first', 'second', 'third', 'million', 'billion',
            'thousand', 'k', 'm', 'b', 'usd', 'eth', 'btc', '000', 'worth', 'value',
            'amount', 'total', 'number', 'numbers', 'figure', 'figures', 'percent',
            'percentage', 'rate', 'rates', 'fee', 'fees',
            
            # Time-related
            'year', 'years', 'month', 'months', 'week', 'weeks', 'day', 'days',
            'hour', 'hours', 'minute', 'minutes', 'second', 'seconds', 'today',
            'tomorrow', 'yesterday', 'time', 'date', 'daily', 'weekly', 'monthly',
            'quarter', 'quarterly', 'annual', 'annually', 'period', 'periods',
            
            # Common verbs and reporting words
            'says', 'said', 'would', 'could', 'may', 'might', 'must', 'need', 'needs',
            'new', 'according', 'reuters', 'told', 'say', 'report', 'reported',
            'reports', 'announced', 'announces', 'announcement', 'update', 'updates',
            'launch', 'launches', 'launched', 'plan', 'plans', 'planned', 'bite',
            'digest', 'sized', 'cryptoasset', 'get', 'delisting', 'announcements',
            'significant', 'rise', 'risen', 'fall', 'fallen', 'increase', 'decrease',
            'grew', 'dropped', 'jumped', 'plunged', 'surged', 'declined', 'gained',
            'lost', 'moved', 'shifted', 'changed', 'updated', 'modified', 'revised',
            
            # Generic business/tech terms
            'company', 'companies', 'business', 'businesses', 'service', 'services',
            'product', 'products', 'user', 'users', 'customer', 'customers',
            'project', 'projects', 'team', 'teams', 'community', 'communities',
            'industry', 'technology', 'tech', 'solution', 'solutions', 'firm',
            'firms', 'group', 'groups', 'partner', 'partners', 'partnership',
            'investor', 'investors', 'trader', 'traders', 'developer', 'developers',
            'development', 'protocol', 'protocols', 'ecosystem', 'ecosystems',
            'platform', 'platforms', 'system', 'systems', 'network', 'networks',
            'infrastructure', 'framework', 'frameworks', 'tool', 'tools',
            
            # Common adjectives
            'new', 'latest', 'current', 'recent', 'upcoming', 'previous',
            'high', 'low', 'higher', 'lower', 'highest', 'lowest',
            'good', 'bad', 'better', 'best', 'worse', 'worst',
            'big', 'small', 'large', 'larger', 'largest', 'major', 'minor',
            'significant', 'important', 'key', 'main', 'primary', 'secondary',
            'additional', 'extra', 'more', 'less', 'several', 'various', 'different',
            'similar', 'same', 'other', 'another', 'next', 'last', 'previous',
            
            # Articles and common words
            'news', 'update', 'updates', 'updated', 'article', 'articles',
            'read', 'reading', 'writes', 'written', 'related', 'latest',
            'breaking', 'exclusive', 'report', 'reported', 'reports',
            'story', 'stories', 'coverage', 'analysis', 'review', 'preview',
            'guide', 'guides', 'tutorial', 'tutorials', 'how', 'what', 'why',
            'when', 'where', 'who', 'which', 'way', 'ways', 'thing', 'things',
            'something', 'anything', 'everything', 'nothing', 'someone',
            'anyone', 'everyone', 'nobody', 'everybody', 'anybody',
            'elsewhere', 'everywhere', 'anywhere', 'nowhere', 'somewhere',
            'pair', 'listing', 'collection',
            
            # Additional generic terms from current output
            'missed', 'listings', 'found', 'information', 'radar', 'investigating',
            'source', 'adobestock', 'co', 'r', 'money', 'fund', 'funds', 'financial',
            'central', 'spot', 'regulatory', 'activity', 'remains', 'reaching',
            'prediction', 'breakout', 'green', 'gains', 'surge', 'native', 'outage',
            'defense', 'charges', 'stolen', 'hackers', 'play', 'social', 'virtual',
            'brands', 'non', 'yacht',
            
            # Crypto exchange related
            'binance', 'coinbase', 'bitfinex', 'tether', 'stablecoin', 'stablecoins',
            'altcoin', 'altcoins', 'exchange', 'exchanges', 'cex', 'dex', 'trading',
            'trader', 'traders', 'volume', 'liquidity', 'order', 'orders', 'pair',
            'pairs', 'listing', 'listings', 'delisting', 'delistings',
            
            # Additional action words and generic terms
            'pull', 'spend', 'apply', 'upside', 'link', 'meanwhile', 'quickly',
            'test', 'victim', 'allegations', 'connections', 'aware', 'safety',
            'search', 'haven', 'original', 'life', 'age', 'san', 'francisco',
            'sharply', 'roughly', 'weekend', 'example', 'shot', 'ex', 'settle',
            'dao', 'votes', 'organizations', 'controversy', 'power', 'minting',
            'identified', 'situation', 'corporate', 'organization', 'super',
            'registered', 'win', 'deal', 'gas', 'engagement', 'significantly',
            'wake', 'posting', 'broader', 'holding', 'reached', 'immediate',
            'headlines', 'recovery', 'offline', 'noted', 'thanks', 'building',
            'tweets', 'records', 'settle', 'district', 'spent', 'rejected',
            'discusses', 'takes', 'chief', 'conditions', 'officer', 'regarding',
            'ruling', 'allegations', 'sales', 'giant', 'allowing', 'purchase',
            'utility', 'real', 'feature', 'people', 'fork', 'owning', 'algorand',
            
            # Location and organization names
            'south', 'singapore', 'korean', 'korea', 'china', 'chinese', 'japan',
            'japanese', 'us', 'usa', 'american', 'european', 'asia', 'asian',
            'africa', 'african', 'australia', 'australian', 'russia', 'russian',
            'india', 'indian', 'brazil', 'brazilian', 'canada', 'canadian',
            
            # Additional crypto project names
            'matic', 'polygon', 'tezos', 'algorand', 'cosmos', 'polkadot',
            'avalanche', 'chainlink', 'uniswap', 'aave', 'maker', 'compound',
            'yearn', 'sushi', 'pancake', 'curve', 'balancer', 'synthetix',
            
            # NFT related terms
            'nft', 'nfts', 'ape', 'bored', 'fungible', 'non', 'collection',
            'collections', 'collectible', 'collectibles', 'art', 'artist',
            'artists', 'creator', 'creators', 'mint', 'minting', 'opensea',
            
            # Additional finance terms
            'market', 'markets', 'trade', 'trading', 'trader', 'traders',
            'investment', 'investments', 'investor', 'investors', 'fund',
            'funds', 'asset', 'assets', 'portfolio', 'portfolios', 'hedge',
            'institutional', 'retail', 'commercial', 'financial', 'finance',
            
            # Time and measurement terms
            'day', 'days', 'week', 'weeks', 'month', 'months', 'year',
            'years', 'hour', 'hours', 'minute', 'minutes', 'second',
            'seconds', 'today', 'tomorrow', 'yesterday', 'morning',
            'afternoon', 'evening', 'night', 'daily', 'weekly', 'monthly',
            'quarterly', 'annually', 'percent', 'percentage', 'basis',
            'points', 'point', 'level', 'levels'
        }
        
        # Initialize CountVectorizer with extended stopwords
        vectorizer_model = CountVectorizer(
            min_df=5,  # Reduced to allow more terms
            max_df=0.5,  # Increased to allow more common terms
            token_pattern=r'(?u)\b[a-zA-Z]+\b',  # Only keep words, no numbers or contractions
            stop_words=list(stop_words)
        )
        
        # Create and fit BERTopic model with reduced_topics parameter
        self.topic_model = BERTopic(
            embedding_model=self.model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=10,  # Further reduced for more focused topics
            verbose=True,
            min_topic_size=40,  # Increased for more stable topics
            n_gram_range=(1, 3),  # Keep longer phrases
            top_n_words=15  # Reduced to focus on most relevant keywords
        )
        
        # Fit the model using pre-computed embeddings
        self.topics, self.probs = self.topic_model.fit_transform(texts, embeddings=self.embeddings)

    def get_topic_info(self):
        """
        Get information about the topics.
        
        Returns:
            pd.DataFrame: DataFrame containing topic information
        """
        if self.topic_model is None:
            raise ValueError("Model must be fit before getting topic information")
        
        topic_info = self.topic_model.get_topic_info()
        topic_info.to_csv("topic_mining_res/topic_info.csv", index=False)
        return topic_info

    def get_topic_keywords(self, topic_id):
        """
        Get keywords for a specific topic.
        
        Args:
            topic_id (int): ID of the topic
            
        Returns:
            list: List of (word, score) tuples
        """
        if self.topic_model is None:
            raise ValueError("Model must be fit before getting topic keywords")
        
        return self.topic_model.get_topic(topic_id)

    def analyze_documents(self, texts, dates=None):
        """
        Analyze documents and save topic distributions.
        
        Args:
            texts (list): List of text documents
            dates (list, optional): List of dates corresponding to the documents
        """
        if self.topic_model is None:
            raise ValueError("Model must be fit before analyzing documents")
        
        # Get topic distributions
        topics, probs = self.topic_model.transform(texts)
        
        # Create DataFrame with topic assignments
        topic_df = pd.DataFrame({
            'document_idx': range(len(texts)),
            'topic': topics,
            'probability': probs.flatten()  # Flatten the 1D array
        })
        
        # Add dates if provided
        if dates is not None:
            topic_df['date'] = dates
        
        # Save topic distributions
        topic_df.to_csv("topic_mining_res/topic_distributions.csv", index=False)

    def run_analysis(self, texts, dates=None, additional_features=None):
        """
        Run complete topic analysis pipeline.
        
        Args:
            texts (list): List of text documents
            dates (list, optional): List of dates corresponding to the documents
            additional_features (numpy.ndarray, optional): Additional numerical features to consider
        """
        # Create embeddings
        self.create_embeddings(texts)
        
        # If additional features are provided, combine them with embeddings
        if additional_features is not None:
            print("\nCombining text embeddings with additional features...")
            # Normalize additional features
            features_normalized = (additional_features - np.mean(additional_features, axis=0)) / np.std(additional_features, axis=0)
            # Combine with embeddings (weighted combination)
            self.embeddings = np.hstack([
                0.7 * self.embeddings,  # Give more weight to text embeddings
                0.3 * features_normalized  # Give less weight to additional features
            ])
        
        # Fit model
        self.fit_model(texts)
        
        # Get topic information
        topic_info = self.get_topic_info()
        print("\nTop topics:")
        print(topic_info.head(15))
        
        # Analyze documents
        self.analyze_documents(texts, dates)
        
        # Classify texts into macro themes
        print("\nClassifying texts into macro themes...")
        macro_labels = self.theme_classifier.classify_texts(texts)
        
        # Add macro theme labels to the topic distributions
        topic_df = pd.read_csv("topic_mining_res/topic_distributions.csv")
        topic_df['macro_theme'] = macro_labels
        
        # Add processed features to the output if available
        if additional_features is not None:
            feature_cols = [f'feature_{i}' for i in range(additional_features.shape[1])]
            for i, col in enumerate(feature_cols):
                topic_df[col] = additional_features[:, i]
        
        topic_df.to_csv("topic_mining_res/topic_distributions.csv", index=False)
        
        # Print distribution of macro themes
        theme_dist = pd.Series(macro_labels).value_counts()
        print("\nDistribution of macro themes:")
        print(theme_dist)
        
        # Save random examples for each theme to a text file
        print("\nSaving theme examples to topic_mining_res/theme_examples.txt...")
        with open("topic_mining_res/theme_examples.txt", "w", encoding="utf-8") as f:
            f.write("Theme Classification Examples\n")
            f.write("==========================\n\n")
            
            # Convert texts to numpy array if it isn't already
            texts_array = np.array(texts)
            
            for theme in self.theme_classifier.theme_descriptions.keys():
                # Get indices where the theme matches
                theme_indices = np.where(np.array(macro_labels) == theme)[0]
                theme_texts = texts_array[theme_indices]
                
                f.write(f"{theme}\n")
                f.write("-" * len(theme) + "\n")
                f.write(f"Description: {self.theme_classifier.theme_descriptions[theme]}\n")
                f.write(f"Total texts in this category: {len(theme_texts)}\n\n")
                
                if len(theme_texts) > 0:
                    # Sample up to 5 random indices
                    sample_indices = np.random.choice(len(theme_texts), size=min(5, len(theme_texts)), replace=False)
                    samples = theme_texts[sample_indices]
                    original_indices = theme_indices[sample_indices]
                    
                    for i, (sample, orig_idx) in enumerate(zip(samples, original_indices), 1):
                        f.write(f"Example {i}:\n")
                        f.write(f"{sample}\n\n")
                        
                        # Add feature values if available
                        if additional_features is not None:
                            f.write("Associated Features:\n")
                            for j, val in enumerate(additional_features[orig_idx]):
                                f.write(f"- Feature {j}: {val:.4f}\n")
                            f.write("\n")
                else:
                    f.write("No examples found for this theme.\n\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print("Examples have been saved to topic_mining_res/theme_examples.txt")
        return topic_info
