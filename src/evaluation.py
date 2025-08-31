"""
Music Recommendation System - Evaluation Metrics
Implements standard evaluation metrics for recommendation systems
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    def __init__(self):
        """Initialize the evaluator"""
        self.test_interactions = None
        self.train_interactions = None
        
    def split_data(self, interactions_df, test_size=0.2, random_state=42):
        """Split interaction data into train and test sets"""
        # Split by timestamp to simulate real-world scenario
        interactions_sorted = interactions_df.sort_values('timestamp')
        split_idx = int(len(interactions_sorted) * (1 - test_size))
        
        self.train_interactions = interactions_sorted.iloc[:split_idx]
        self.test_interactions = interactions_sorted.iloc[split_idx:]
        
        print(f"Train set: {len(self.train_interactions)} interactions")
        print(f"Test set: {len(self.test_interactions)} interactions")
        
        return self.train_interactions, self.test_interactions
    
    def evaluate_rating_prediction(self, recommender, test_interactions=None):
        """Evaluate rating prediction accuracy"""
        if test_interactions is None:
            test_interactions = self.test_interactions
        
        if test_interactions is None:
            raise ValueError("No test data available. Call split_data first.")
        
        print("Evaluating rating prediction accuracy...")
        
        predictions = []
        actuals = []
        
        for _, interaction in test_interactions.iterrows():
            user_id = interaction['user_id']
            song_id = interaction['song_id']
            actual_rating = interaction['rating']
            
            try:
                predicted_rating = recommender.predict_rating(user_id, song_id)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except Exception as e:
                # Skip if prediction fails
                continue
        
        if len(predictions) == 0:
            return {'error': 'No valid predictions made'}
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        # Calculate coverage (percentage of test cases we could predict)
        coverage = len(predictions) / len(test_interactions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage,
            'num_predictions': len(predictions)
        }
    
    def precision_at_k(self, recommended_items, relevant_items, k):
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = set(recommended_k) & set(relevant_items)
        
        return len(relevant_recommended) / k
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        """Calculate Recall@K"""
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = set(recommended_k) & set(relevant_items)
        
        return len(relevant_recommended) / len(relevant_items)
    
    def ndcg_at_k(self, recommended_items, relevant_items, k):
        """Calculate Normalized Discounted Cumulative Gain@K"""
        def dcg_at_k(r, k):
            r = np.asfarray(r)[:k]
            if r.size:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            return 0.0
        
        # Create relevance scores (1 for relevant, 0 for not relevant)
        relevance_scores = [1 if item in relevant_items else 0 
                          for item in recommended_items[:k]]
        
        # Calculate DCG
        dcg = dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1] * min(len(relevant_items), k)
        idcg = dcg_at_k(ideal_relevance, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_ranking(self, recommender, k_values=[5, 10, 20], rating_threshold=4):
        """Evaluate ranking quality using Precision@K, Recall@K, and NDCG@K"""
        if self.test_interactions is None:
            raise ValueError("No test data available. Call split_data first.")
        
        print(f"Evaluating ranking quality with rating threshold >= {rating_threshold}...")
        
        # Group test interactions by user
        user_test_items = defaultdict(list)
        for _, interaction in self.test_interactions.iterrows():
            if interaction['rating'] >= rating_threshold:
                user_test_items[interaction['user_id']].append(interaction['song_id'])
        
        # Calculate metrics for each user
        metrics = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
        
        evaluated_users = 0
        for user_id, relevant_items in user_test_items.items():
            if len(relevant_items) == 0:
                continue
            
            try:
                # Get recommendations for this user
                recommendations = recommender.recommend_songs(
                    user_id, n_recommendations=max(k_values), exclude_rated=True
                )
                
                # Calculate metrics for each k
                for k in k_values:
                    precision = self.precision_at_k(recommendations, relevant_items, k)
                    recall = self.recall_at_k(recommendations, relevant_items, k)
                    ndcg = self.ndcg_at_k(recommendations, relevant_items, k)
                    
                    metrics[k]['precision'].append(precision)
                    metrics[k]['recall'].append(recall)
                    metrics[k]['ndcg'].append(ndcg)
                
                evaluated_users += 1
                
            except Exception as e:
                # Skip users where recommendation fails
                continue
        
        # Calculate average metrics
        results = {}
        for k in k_values:
            results[f'precision@{k}'] = np.mean(metrics[k]['precision']) if metrics[k]['precision'] else 0
            results[f'recall@{k}'] = np.mean(metrics[k]['recall']) if metrics[k]['recall'] else 0
            results[f'ndcg@{k}'] = np.mean(metrics[k]['ndcg']) if metrics[k]['ndcg'] else 0
        
        results['evaluated_users'] = evaluated_users
        results['total_test_users'] = len(user_test_items)
        
        return results
    
    def evaluate_diversity(self, recommender, user_ids=None, n_recommendations=10):
        """Evaluate recommendation diversity"""
        if user_ids is None:
            # Sample some users for diversity evaluation
            all_users = self.train_interactions['user_id'].unique()
            user_ids = np.random.choice(all_users, size=min(50, len(all_users)), replace=False)
        
        print("Evaluating recommendation diversity...")
        
        # Get song data for diversity calculation
        songs_df = pd.read_csv('../data/songs.csv')
        
        all_recommendations = []
        user_diversities = []
        
        for user_id in user_ids:
            try:
                recommendations = recommender.recommend_songs(user_id, n_recommendations)
                all_recommendations.extend(recommendations)
                
                # Calculate intra-list diversity for this user
                if len(recommendations) > 1:
                    user_songs = songs_df[songs_df['song_id'].isin(recommendations)]
                    
                    # Genre diversity
                    unique_genres = user_songs['genre'].nunique()
                    genre_diversity = unique_genres / len(recommendations)
                    
                    # Audio feature diversity (using standard deviation)
                    audio_features = ['danceability', 'energy', 'valence', 'acousticness']
                    feature_diversity = user_songs[audio_features].std().mean()
                    
                    user_diversities.append({
                        'genre_diversity': genre_diversity,
                        'feature_diversity': feature_diversity
                    })
                
            except Exception as e:
                continue
        
        # Calculate catalog coverage
        unique_recommended = len(set(all_recommendations))
        total_songs = len(songs_df)
        catalog_coverage = unique_recommended / total_songs
        
        # Calculate average diversities
        avg_genre_diversity = np.mean([d['genre_diversity'] for d in user_diversities])
        avg_feature_diversity = np.mean([d['feature_diversity'] for d in user_diversities])
        
        return {
            'catalog_coverage': catalog_coverage,
            'avg_genre_diversity': avg_genre_diversity,
            'avg_feature_diversity': avg_feature_diversity,
            'unique_songs_recommended': unique_recommended,
            'total_songs': total_songs,
            'evaluated_users': len(user_diversities)
        }
    
    def evaluate_novelty(self, recommender, user_ids=None, n_recommendations=10):
        """Evaluate recommendation novelty (how unpopular/surprising the recommendations are)"""
        if user_ids is None:
            all_users = self.train_interactions['user_id'].unique()
            user_ids = np.random.choice(all_users, size=min(50, len(all_users)), replace=False)
        
        print("Evaluating recommendation novelty...")
        
        # Calculate song popularity from training data
        song_popularity = self.train_interactions.groupby('song_id').size()
        total_interactions = len(self.train_interactions)
        song_popularity_norm = song_popularity / total_interactions
        
        novelty_scores = []
        
        for user_id in user_ids:
            try:
                recommendations = recommender.recommend_songs(user_id, n_recommendations)
                
                # Calculate novelty as negative log of popularity
                user_novelty = []
                for song_id in recommendations:
                    if song_id in song_popularity_norm:
                        popularity = song_popularity_norm[song_id]
                        novelty = -np.log2(popularity)
                        user_novelty.append(novelty)
                
                if user_novelty:
                    novelty_scores.append(np.mean(user_novelty))
                
            except Exception as e:
                continue
        
        return {
            'avg_novelty': np.mean(novelty_scores) if novelty_scores else 0,
            'novelty_std': np.std(novelty_scores) if novelty_scores else 0,
            'evaluated_users': len(novelty_scores)
        }
    
    def compare_recommenders(self, recommenders, recommender_names=None):
        """Compare multiple recommenders across all metrics"""
        if recommender_names is None:
            recommender_names = [f"Recommender_{i+1}" for i in range(len(recommenders))]
        
        print("Comparing recommenders...")
        
        results = {}
        
        for name, recommender in zip(recommender_names, recommenders):
            print(f"\nEvaluating {name}...")
            
            # Rating prediction accuracy
            rating_metrics = self.evaluate_rating_prediction(recommender)
            
            # Ranking quality
            ranking_metrics = self.evaluate_ranking(recommender)
            
            # Diversity
            diversity_metrics = self.evaluate_diversity(recommender)
            
            # Novelty
            novelty_metrics = self.evaluate_novelty(recommender)
            
            results[name] = {
                **rating_metrics,
                **ranking_metrics,
                **diversity_metrics,
                **novelty_metrics
            }
        
        return results
    
    def plot_comparison(self, comparison_results, save_path=None):
        """Plot comparison results"""
        # Prepare data for plotting
        metrics_to_plot = [
            'rmse', 'mae', 'precision@10', 'recall@10', 'ndcg@10',
            'catalog_coverage', 'avg_genre_diversity', 'avg_novelty'
        ]
        
        data = []
        for recommender, metrics in comparison_results.items():
            for metric in metrics_to_plot:
                if metric in metrics:
                    data.append({
                        'Recommender': recommender,
                        'Metric': metric,
                        'Value': metrics[metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            metric_data = df[df['Metric'] == metric]
            if not metric_data.empty:
                sns.barplot(data=metric_data, x='Recommender', y='Value', ax=axes[i])
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        return fig
    
    def generate_evaluation_report(self, comparison_results, save_path=None):
        """Generate a comprehensive evaluation report"""
        report = "# Music Recommendation System Evaluation Report\n\n"
        
        # Summary table
        report += "## Performance Summary\n\n"
        report += "| Recommender | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | Novelty |\n"
        report += "|-------------|------|-----|--------------|-----------|---------|----------|-----------|----------|\n"
        
        for name, metrics in comparison_results.items():
            report += f"| {name} | "
            report += f"{metrics.get('rmse', 'N/A'):.3f} | " if 'rmse' in metrics else "N/A | "
            report += f"{metrics.get('mae', 'N/A'):.3f} | " if 'mae' in metrics else "N/A | "
            report += f"{metrics.get('precision@10', 'N/A'):.3f} | " if 'precision@10' in metrics else "N/A | "
            report += f"{metrics.get('recall@10', 'N/A'):.3f} | " if 'recall@10' in metrics else "N/A | "
            report += f"{metrics.get('ndcg@10', 'N/A'):.3f} | " if 'ndcg@10' in metrics else "N/A | "
            report += f"{metrics.get('catalog_coverage', 'N/A'):.3f} | " if 'catalog_coverage' in metrics else "N/A | "
            report += f"{metrics.get('avg_genre_diversity', 'N/A'):.3f} | " if 'avg_genre_diversity' in metrics else "N/A | "
            report += f"{metrics.get('avg_novelty', 'N/A'):.3f} |\n" if 'avg_novelty' in metrics else "N/A |\n"
        
        # Detailed analysis
        report += "\n## Detailed Analysis\n\n"
        
        for name, metrics in comparison_results.items():
            report += f"### {name}\n\n"
            
            if 'rmse' in metrics:
                report += f"**Rating Prediction:**\n"
                report += f"- RMSE: {metrics['rmse']:.3f}\n"
                report += f"- MAE: {metrics['mae']:.3f}\n"
                report += f"- Coverage: {metrics['coverage']:.3f}\n\n"
            
            if 'precision@10' in metrics:
                report += f"**Ranking Quality:**\n"
                report += f"- Precision@10: {metrics['precision@10']:.3f}\n"
                report += f"- Recall@10: {metrics['recall@10']:.3f}\n"
                report += f"- NDCG@10: {metrics['ndcg@10']:.3f}\n\n"
            
            if 'catalog_coverage' in metrics:
                report += f"**Diversity & Novelty:**\n"
                report += f"- Catalog Coverage: {metrics['catalog_coverage']:.3f}\n"
                report += f"- Genre Diversity: {metrics['avg_genre_diversity']:.3f}\n"
                report += f"- Novelty Score: {metrics['avg_novelty']:.3f}\n\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to {save_path}")
        
        return report

def main():
    """Demo of evaluation system"""
    from collaborative_filtering import CollaborativeFilteringRecommender
    from content_based_filtering import ContentBasedRecommender
    from hybrid_recommender import HybridRecommender
    
    # Load data
    interactions_df = pd.read_csv('../data/interactions.csv')
    songs_df = pd.read_csv('../data/songs.csv')
    
    print("=== Recommendation System Evaluation Demo ===")
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator()
    train_data, test_data = evaluator.split_data(interactions_df)
    
    # Train different recommenders on training data
    print("\nTraining recommenders...")
    
    # Collaborative Filtering
    cf_recommender = CollaborativeFilteringRecommender(method='matrix_factorization')
    cf_recommender.fit(train_data)
    
    # Content-Based
    cb_recommender = ContentBasedRecommender()
    cb_recommender.fit(songs_df, train_data)
    
    # Hybrid
    hybrid_recommender = HybridRecommender()
    hybrid_recommender.fit(songs_df, train_data)
    
    # Compare all recommenders
    recommenders = [cf_recommender, cb_recommender, hybrid_recommender]
    names = ['Collaborative Filtering', 'Content-Based', 'Hybrid']
    
    comparison_results = evaluator.compare_recommenders(recommenders, names)
    
    # Print results
    print("\n=== Evaluation Results ===")
    for name, metrics in comparison_results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    # Generate report
    report = evaluator.generate_evaluation_report(
        comparison_results, 
        save_path='../results/evaluation_report.md'
    )
    
    # Create comparison plot
    fig = evaluator.plot_comparison(
        comparison_results, 
        save_path='../results/comparison_plot.png'
    )
    
    print("\nEvaluation completed! Check the results folder for detailed reports.")

if __name__ == "__main__":
    main()

